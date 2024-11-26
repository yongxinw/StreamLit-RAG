"""
Take the markdown file and output
    (1) question only md file and 
    (2) question to answer mapping json file.

Make sure the content follows the pattern:

1. 问题123456
回复：回答123456

2. 问题1234567
回复：回答1234567

The file name should be xxx_qa.md
"""

import json
import os

# Load the content of the uploaded Markdown file to examine its format
file_names = [
    "cont_edu_qa.md",
    "individual_qa.md",
    "supervisory_dept_qa.md",
    "employing_unit_qa.md",
    "jining_qa.md",
    "login_problems_details_qa.md",
    "forgot_password_qa.md",
]

root_path = "policies_v2"

for file_name in file_names:

    file_path = os.path.join(root_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()

    # print(content)

    # Initialize containers for the questions and the question-answer map
    questions = []
    qa_map = {}

    # Variable to keep track of the last question for mapping answers
    last_question = None
    last_question_key = None
    all_question_keys = []
    for line in content:
        if line.strip().startswith("回复"):
            # Process answer lines
            answer = line.strip()[3:]  # Remove '回复：' part
            if last_question:
                qa_map[last_question_key] = f"问题：{last_question_key}\t回复: {answer}"
                # print(len(qa_map))
                # print('===============')
        elif line.strip() and line.strip()[0].isdigit():
            # Process question lines
            last_question = line.strip()
            last_question_key = " ".join(line.strip().split()[1:])
            if last_question_key not in all_question_keys:
                questions.append(last_question)
                all_question_keys.append(last_question_key)
                # print(len(questions))

    # Save the questions-only file
    output_q_name = "_".join(file_name.split("_")[:-1]) + "_q.md"
    questions_file_path = os.path.join(root_path, output_q_name)
    with open(questions_file_path, "w", encoding="utf-8") as q_file:
        q_file.write("\n\n".join(questions))

    # Save the question-answer map file
    output_qa_map_name = "_".join(file_name.split("_")[:-1]) + "_qa_map.json"
    qa_map_file_path = os.path.join(root_path, output_qa_map_name)
    with open(qa_map_file_path, "w", encoding="utf-8") as qa_file:
        json.dump(qa_map, qa_file, ensure_ascii=False, indent=2)
