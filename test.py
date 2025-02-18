from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    print("This is a test")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    dir = 'saves/test/oreo/lora-qwen-1.5B/oreo/checkpoint-8000'
    model.load_adapter(dir + "/policy")
    inputs = tokenizer("Hello World", return_tensors='pt')
    outputs = model(**inputs)
    policy_logits = outputs["logits"]
    del model
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    model.load_adapter(dir + "/reward")
    outputs = model(**inputs)
    reward_logits = outputs["logits"]
    print(policy_logits - reward_logits)
    # prompt = "Hello there "
    # response = "World"
    # input_token = tokenizer(
    #     prompt + response + " " + tokenizer.eos_token,
    #     max_length=100,
    #     padding=False,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # ids = input_token["input_ids"]
    # attention_mask = input_token["attention_mask"]

    # state_mask = torch.zeros_like(ids)
    # action_mask = torch.zeros_like(ids)
    # idx = input_token.char_to_token(len(prompt))  # first token pos of response
    # state_mask[0][idx - 1 : -1] = 1
    # action_mask[0][idx:] = 1
    # print(state_mask)
    # print(action_mask)
    # print(ids)

    # input_ids = [151644, 8948, 198, 5501, 2874, 3019, 553, 3019, 11, 323, 8193, 264, 1590, 4226, 2701, 220, 19, 5875, 516, 1075, 364, 820, 220, 15, 6, 151645, 198, 151644, 872, 198, 45, 4212, 685, 6088, 26111, 311, 220, 19, 23, 315, 1059, 4780, 304, 5813, 11, 323, 1221, 1340, 6088, 4279, 438, 1657, 26111, 304, 3217, 13, 2585, 1657, 26111, 1521, 41601, 685, 4559, 30055, 304, 5813, 323, 3217, 30, 151645, 198, 151644, 77091, 198, 641, 3217, 11, 41601, 685, 6088, 220, 19, 23, 608, 220, 17, 284, 1115, 19, 23, 14, 17, 28, 17, 19, 2452, 17, 19, 26111, 624, 45, 4212, 685, 6088, 220, 19, 23, 488, 220, 17, 19, 284, 1115, 19, 23, 10, 17, 19, 28, 22, 17, 2452, 22, 17, 26111, 30055, 304, 5813, 323, 3217, 624, 820, 220, 22, 17, 220, 151643]
    # action_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # state_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # reward_score = [1]
    # label_ids = [151643, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 641, 3217, 11, 41601, 685, 6088, 220, 19, 23, 608, 220, 17, 284, 1115, 19, 23, 14, 17, 28, 17, 19, 2452, 17, 19, 26111, 624, 45, 4212, 685, 6088, 220, 19, 23, 488, 220, 17, 19, 284, 1115, 19, 23, 10, 17, 19, 28, 22, 17, 2452, 22, 17, 26111, 30055, 304, 5813, 323, 3217, 624, 820, 220, 22, 17, 220, 151643]

    # example = {
    #     "input_ids": input_ids,
    #     "action_mask": action_mask,
    #     "state_mask": state_mask,
    #     "reward_score": reward_score,
    #     "labels": label_ids
    # }
    # collator = MultiModalDataCollatorForSeq2Seq()



    # cnt = 0
    # for l in label_ids:
    #     if l == -100:
    #         cnt += 1
    # print(cnt)