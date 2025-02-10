import os
import requests
import re

# Function to read the last 10 lines of a file
def read_last_10_lines(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()[-10:]
    return lines

def get_url() -> str:
    with open('.env', 'r') as file:
        for line in file:
            line = re.sub('\n', '', line)

    return line

# Function to post messages to Slack API
def post_to_slack(lines):
    url = get_url()
    content_txt = "".join(lines)
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"message": content_txt}
    )
    if response.status_code != 200:
        print(f"Failed to post message: {response.text}")

# Main function
def main():
    id_file = 'log.txt'
    directory = '/root/slurm_log/'

    with open(id_file, 'r') as file:
        for line in file:
            line = re.sub("\n", "", line)
            id_num = line[-6:]
    for filename in os.listdir(directory):
        if id_num in filename:
            file_path = os.path.join(directory, filename)
            lines = read_last_10_lines(file_path)
            post_to_slack(lines)

if __name__ == "__main__":
    main()
