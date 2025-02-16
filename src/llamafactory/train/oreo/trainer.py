# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import json
import os
import re
from pathlib import Path
import shutil
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.functional import F
from transformers import Trainer
from typing_extensions import override
from tqdm import tqdm
from trl.core import PPODecorators

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixOREOValueHeadModelCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_trainer import CustomPPOTrainer


from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class OREOTrainer(CustomPPOTrainer):
    r"""
    Inherits Trainer
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        eval_dataset = kwargs.pop('eval_dataset')
        super().__init__(*args, **kwargs)
        training_args = kwargs['training_args']
        finetuning_args = kwargs['finetuning_args']
        # train the reward model
        self.reward_model.train()
        self.beta = training_args.reward_beta
        self.kl_reg = training_args.oreo_kl_reg
        self.unbiased_kl = training_args.oreo_unbiased_kl

        default_learning_rate = training_args.learning_rate
        training_args.learning_rate = training_args.reward_learning_rate

        # create optimizer and scheduler for reward model
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = training_args.per_device_train_batch_size \
                * training_args.gradient_accumulation_steps * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(kwargs.get('train_dataset')) / total_train_batch_size
            )

        reward_optimizer = self.create_optimizer(self.reward_model, training_args, finetuning_args)
        reward_scheduler = self.create_scheduler(training_args, num_training_steps, reward_optimizer)
        self.reward_optimizer, self.reward_scheduler = self.accelerator.prepare(reward_optimizer, reward_scheduler)
        # revert the learning rate
        training_args.learning_rate = default_learning_rate

        self.add_callback(FixOREOValueHeadModelCallback(self.accelerator.unwrap_model(self.reward_model)))

    def oreo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the OREO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        self._log_info()

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )

        dataiter = iter(self.dataloader)
        loss_reward_meter = AverageMeter()
        loss_policy_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        max_steps = self.state.max_steps
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            with torch.no_grad():
                accumulated_logps, logps, _ = self.accumulated_logps(self.model, batch["input_ids"], batch["attention_mask"], action_mask)
            _, _, values = self.reward_model(batch["input_ids"], attention_mask=batch["attention_mask"], action_mask=action_mask)
            values_detached: torch.Tensor = values.clone().detach()

            with torch.no_grad():
                with self.optional_peft_ctx(): # with this context, self.model will deactivate its LoRA
                    reference_accumulated_logps, reference_logps, reference_logps_raw = self.accumulated_logps(
                        self.model if self.ref_model is None else self.ref_model,
                        batch["input_ids"],
                        batch["attention_mask"],
                        action_mask
                    )

            # estimate KL
            kls = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
            kl_estimate = torch.mean((kls * action_mask[:, 1:]).sum(dim=-1) / action_mask[:, 1:].sum(dim=-1))

            # # value statistics
            # max_value = values_detached.masked_fill(~state_mask[:, :-1].bool(), float("-inf")).max()
            # min_value = values_detached.masked_fill(~state_mask[:, :-1].bool(), float("inf")).min()

            loss = self.loss(
                accumulated_logps,
                reference_accumulated_logps,
                values,
                state_mask,
                batch["reward_score"]
            )
            self.accelerator.backward(loss)
            self.reward_optimizer.step()
            self.reward_optimizer.zero_grad()
            self.reward_scheduler.step()

            loss_reward_meter.update(loss.detach().mean().item(), n=total_train_batch_size)
            
            # training the policy model
            accumulated_logps, logps, logps_raw = self.accumulated_logps(self.model, batch["input_ids"], batch["attention_mask"], action_mask)

            actor_loss = self.dro_loss(
                logps,
                reference_logps,
                accumulated_logps,
                reference_accumulated_logps,
                values_detached,
                state_mask,
                batch["reward_score"],
            )

            if self.kl_reg is not None:
                if not self.unbiased_kl:
                    kls_actor = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
                else:
                    kls_actor = F.kl_div(
                        logps_raw, reference_logps_raw, reduction="none", log_target=True
                    ).sum(dim=-1)
                kl_estimate_actor = torch.mean(
                    (kls_actor * action_mask[:, 1:]).sum(dim=-1) / action_mask[:, 1:].sum(dim=-1)
                )
                self.accelerator.backward(actor_loss + self.kl_reg * kl_estimate_actor)
            else:
                self.accelerator.backward(actor_loss)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            loss_policy_meter.update(actor_loss.detach().mean().item(), n=total_train_batch_size)

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss_reward=round(loss_reward_meter.avg, 4),
                    policy_learning_rate=self.optimizer.param_groups[0]["lr"],
                    reward_learning_rate=self.reward_optimizer.param_groups[0]["lr"],
                    kl_estimate=kl_estimate.detach().mean().item(),
                    loss_policy=round(loss_policy_meter.avg, 4),

                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_reward_meter.reset()
                loss_policy_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"),
                    is_reward_model=True,
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

                if self.args.should_save:
                    self._rotate_checkpoints(use_mtime=False, output_dir=self.args.output_dir)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @PPODecorators.empty_device_cache()
    def accumulated_logps(self, model, ids: torch.Tensor, masks: torch.Tensor, action_masks: torch.Tensor):
        logits, _, _ = model(ids, attention_mask=masks)

        logits = logits[:, :-1, :]
        labels = ids[:, 1:]  # [bsz, seq_len]
        action_masks = action_masks[:, 1:]

        logps_raw = torch.log_softmax(logits, dim=-1)  # [bsz, seq_len, vocab]
        logps = logps_raw.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        accumulated_logps = (logps * action_masks).flip(-1).cumsum(-1).flip(-1)

        return accumulated_logps, logps, logps_raw

    def loss(
        self,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        rewards: torch.Tensor,
    ):
        policy_rewards = self.beta * (accumulated_logps - reference_accumulated_logps)

        state_masks = state_masks[:, :-1]
        rewards = rewards.unsqueeze(1)
        tmp = torch.square((policy_rewards + values - rewards) * state_masks).sum(dim=-1) / state_masks.sum(dim=-1)
        return tmp.mean()

    def dro_loss(
        self,
        logps: torch.Tensor,
        reference_logps: torch.Tensor,
        accumulated_logps: torch.Tensor,
        reference_accumulated_logps: torch.Tensor,
        values: torch.Tensor,
        state_masks: torch.Tensor,
        rewards: torch.Tensor,
    ):
        policy_rewards = accumulated_logps - reference_accumulated_logps
        policy_rewards = torch.cat([policy_rewards[:, 1:], torch.zeros_like(values[:, -1]).unsqueeze(-1)], dim=-1)
        policy_rewards = policy_rewards.detach() + logps - reference_logps

        policy_rewards = self.beta * policy_rewards

        state_masks = state_masks[:, :-1]
        rewards = rewards.unsqueeze(1)
        tmp = torch.square((policy_rewards + values - rewards) * state_masks).sum(dim=-1) / state_masks.sum(dim=-1)
        return tmp.mean()

    def _log_info(self) -> None:
        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                total_train_batch_size
            )
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {(count_parameters(self.model)[0]+count_parameters(self.reward_model)[0]):,}")

    @override
    def save_model(self, output_dir: Optional[str] = None, is_reward_model: bool = False) -> None:
        if is_reward_model:
            output_dir += "/reward"
            self.save_reward_model(output_dir)
        else:
            output_dir += "/policy"
            self.save_main_model(output_dir)

    def save_reward_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves reward model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.reward_model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.reward_model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithOREOValueHead" = self.accelerator.unwrap_model(self.reward_model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())

    def save_main_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
