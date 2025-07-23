import argparse
import os
import json
import re

from glob import glob
from libs.llm.base_model import ApiChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default='evaluation', help='The directory of inputs.')
    parser.add_argument('-u', '--url', type=str, default='http://0.0.0.0:8000/v1', help='The IP address of the LLM server.')
    parser.add_argument('-m', '--model', type=str, default='gpt-4.1', help='The model name of the LLM server.')
    parser.add_argument('-k', '--key', type=str, default='EMPTY', help='The API key of the LLM server.')

    return parser.parse_args()

def get_prompt_template(content_A, content_B, role_name, role_series):
    # Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents
    # https://arxiv.org/abs/2408.04203

    system_prompt = 'You are an objective and precise evaluator, specializing in rigorously assessing the role-playing and multimodal understanding abilities of various models.'

    prompt = f'''## [Model A's Response Start]\n\n{content_A}\n\n## [Model A's Response End]\n\n\n
                ## [Model B's Response Start]\n\n{content_B}\n\n## [Model B's Response End]\n\n\n
                ## [Instruction]\n\n
                The task instruction of the two models is to directly role-play as {role_name} from {role_series} and talk with a curious human 
                using the distinctive tone, manner and vocabulary of {role_name}. \n\n
                Please evaluate the following three aspects of each model's response:\n
                Consistency of Persona: Do the conversational style and behavioral traits exhibited by the agent align with {role_name}?\n
                Accuracy of Knowledge: Do the agent demonstrates accurate and contextually appropriate knowledge reflective of the character’s background, 
                including both the correct articulation of relevant information and the avoidance of statements inconsistent with the character’s known attributes or setting?\n
                Quality of Conversation: The overall impression of the conversation, encompassing aspects such as fluency, engagement.\n\n
                Please provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from
                1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.\n\n
                Only provide ratings, do not give any other evaluations. The output should be in the following format:\n
                [Consistency of Persona]: the score of Model A, the score of Model B\n\n
                [Accuracy of Knowledge]: the score of Model A, the score of Model B\n
                [Quality of Conversation]: the score of Model A, the score of Model B\n\n
                Please ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment.'''

    message = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

    return message

if __name__ == '__main__':

    args = parse()

    model = ApiChatModel(args.model, args.key, args.url, max_tokens=2048, temperature=0.2, top_p=0.8)

    subjects = os.listdir(args.input)
    subjects = [s for s in subjects if os.path.isdir(os.path.join(args.input, s))]

    # evaluate
    for subject in subjects:
        print(f'Evaluating {subject}...')

        if os.path.exists(os.path.join(args.input, subject, 'evaluation_records.json')):
            continue

        role_name = subject.split('_')[3]
        role_series = subject.split('_')[4]

        files = glob(os.path.join(args.input, subject, '*.txt'))

        records = {}
        contents = {}

        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                contents[os.path.basename(file)] = f.read()

        keys = list(contents.keys())

        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                content_A = contents[keys[i]]
                content_B = contents[keys[j]]

                prompt = get_prompt_template(content_A, content_B, role_name, role_series)
                response = model.invoke(prompt)
                records[f'{keys[i]}-{keys[j]}'] = response.content

                print(keys[i], keys[j])
                print(response.content)

                # inverse
                prompt = get_prompt_template(content_B, content_A, role_name, role_series)
                response = model.invoke(prompt)
                records[f'{keys[j]}-{keys[i]}'] = response.content

                print(keys[j], keys[i])
                print(response.content)

        with open(os.path.join(args.input, subject, 'evaluation_records.json'), 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)


    C_P_scores = {}
    A_K_scores = {}
    Q_C_scores = {}

    C_P_zh_scores = {}
    A_K_zh_scores = {}
    Q_C_zh_scores = {}

    C_P_en_scores = {}
    A_K_en_scores = {}
    Q_C_en_scores = {}

    # scoring
    for subject in subjects:
        print(f'Scoring {subject}...')

        with open(os.path.join(args.input, subject, 'evaluation_records.json'), 'r', encoding='utf-8') as f:
            records = json.load(f)

        for key in records:
            content = records[key]

            pattern = r'\[(.*?)\]:\s*(\d+,\s*\d+)'
            matches = re.findall(pattern, content)

            scores = {}
            for k, v in matches:
                score_list = [int(num.strip()) for num in v.split(',')]
                scores[k] = score_list

            C_P_scores.setdefault(key.split('-')[0][0], []).append(scores['Consistency of Persona'][0])
            C_P_scores.setdefault(key.split('-')[1][0], []).append(scores['Consistency of Persona'][1])

            A_K_scores.setdefault(key.split('-')[0][0], []).append(scores['Accuracy of Knowledge'][0])
            A_K_scores.setdefault(key.split('-')[1][0], []).append(scores['Accuracy of Knowledge'][1])

            Q_C_scores.setdefault(key.split('-')[0][0], []).append(scores['Quality of Conversation'][0])
            Q_C_scores.setdefault(key.split('-')[1][0], []).append(scores['Quality of Conversation'][1])

            if 'zh' in key:
                C_P_zh_scores.setdefault(key.split('-')[0][0], []).append(scores['Consistency of Persona'][0])
                C_P_zh_scores.setdefault(key.split('-')[1][0], []).append(scores['Consistency of Persona'][1])

                A_K_zh_scores.setdefault(key.split('-')[0][0], []).append(scores['Accuracy of Knowledge'][0])
                A_K_zh_scores.setdefault(key.split('-')[1][0], []).append(scores['Accuracy of Knowledge'][1])

                Q_C_zh_scores.setdefault(key.split('-')[0][0], []).append(scores['Quality of Conversation'][0])
                Q_C_zh_scores.setdefault(key.split('-')[1][0], []).append(scores['Quality of Conversation'][1])

            if 'en' in key:
                C_P_en_scores.setdefault(key.split('-')[0][0], []).append(scores['Consistency of Persona'][0])
                C_P_en_scores.setdefault(key.split('-')[1][0], []).append(scores['Consistency of Persona'][1])

                A_K_en_scores.setdefault(key.split('-')[0][0], []).append(scores['Accuracy of Knowledge'][0])
                A_K_en_scores.setdefault(key.split('-')[1][0], []).append(scores['Accuracy of Knowledge'][1])

                Q_C_en_scores.setdefault(key.split('-')[0][0], []).append(scores['Quality of Conversation'][0])
                Q_C_en_scores.setdefault(key.split('-')[1][0], []).append(scores['Quality of Conversation'][1])

    # averaging
    for key in C_P_zh_scores:
        C_P_zh_scores[key] = sum(C_P_zh_scores[key]) / len(C_P_zh_scores[key])
    C_P_zh_scores = sorted(C_P_zh_scores.items(), key=lambda x: x[1], reverse=True)

    for key in A_K_zh_scores:
        A_K_zh_scores[key] = sum(A_K_zh_scores[key]) / len(A_K_zh_scores[key])
    A_K_zh_scores = sorted(A_K_zh_scores.items(), key=lambda x: x[1], reverse=True)

    for key in Q_C_zh_scores:
        Q_C_zh_scores[key] = sum(Q_C_zh_scores[key]) / len(Q_C_zh_scores[key])
    Q_C_zh_scores = sorted(Q_C_zh_scores.items(), key=lambda x: x[1], reverse=True)

    print('Consistency of Persona scores (zh):', C_P_zh_scores)
    print('Accuracy of Knowledge scores (zh):', A_K_zh_scores)
    print('Quality of Conversation scores (zh):', Q_C_zh_scores)

    for key in C_P_en_scores:
        C_P_en_scores[key] = sum(C_P_en_scores[key]) / len(C_P_en_scores[key])
    C_P_en_scores = sorted(C_P_en_scores.items(), key=lambda x: x[1], reverse=True)

    for key in A_K_en_scores:
        A_K_en_scores[key] = sum(A_K_en_scores[key]) / len(A_K_en_scores[key])
    A_K_en_scores = sorted(A_K_en_scores.items(), key=lambda x: x[1], reverse=True)

    for key in Q_C_en_scores:
        Q_C_en_scores[key] = sum(Q_C_en_scores[key]) / len(Q_C_en_scores[key])
    Q_C_en_scores = sorted(Q_C_en_scores.items(), key=lambda x: x[1], reverse=True)

    print('Consistency of Persona scores (en):', C_P_en_scores)
    print('Accuracy of Knowledge scores (en):', A_K_en_scores)
    print('Quality of Conversation scores (en):', Q_C_en_scores)

    for key in C_P_scores:
        C_P_scores[key] = sum(C_P_scores[key]) / len(C_P_scores[key])
    C_P_scores = sorted(C_P_scores.items(), key=lambda x: x[1], reverse=True)

    for key in A_K_scores:
        A_K_scores[key] = sum(A_K_scores[key]) / len(A_K_scores[key])
    A_K_scores = sorted(A_K_scores.items(), key=lambda x: x[1], reverse=True)

    for key in Q_C_scores:
        Q_C_scores[key] = sum(Q_C_scores[key]) / len(Q_C_scores[key])
    Q_C_scores = sorted(Q_C_scores.items(), key=lambda x: x[1], reverse=True)

    print('Consistency of Persona scores:', C_P_scores)
    print('Accuracy of Knowledge scores:', A_K_scores)
    print('Quality of Conversation scores:', Q_C_scores)

    with open(os.path.join(args.input, 'evaluation_scores_zh.json'), 'w', encoding='utf-8') as f:
        json.dump({'C_P_zh_scores': C_P_zh_scores, 'A_K_zh_scores': A_K_zh_scores, 'Q_C_zh_scores': Q_C_zh_scores}, f, indent=4)

    with open(os.path.join(args.input, 'evaluation_scores_en.json'), 'w', encoding='utf-8') as f:
        json.dump({'C_P_en_scores': C_P_en_scores, 'A_K_en_scores': A_K_en_scores, 'Q_C_en_scores': Q_C_en_scores}, f, indent=4)

    with open(os.path.join(args.input, 'evaluation_scores.json'), 'w', encoding='utf-8') as f:
        json.dump({'C_P_scores': C_P_scores, 'A_K_scores': A_K_scores, 'Q_C_scores': Q_C_scores}, f, indent=4)











        
