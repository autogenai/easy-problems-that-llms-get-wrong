#%%
from utils import message_parse, model_clean
from collections.abc import Callable
from llm_service import runner
import pandas as pd
import numpy as np
import json
import copy
import re
import os

def create_eval_prompt(question:str, human_response:str, model_response:str) -> str:
    return f"""SCORING CRITERIA
100%: The response contains the correct answer only with a correct thought process and no logical inconsistencies.
80%: The response contains the correct answer only with a correct thought process with some logical inconsistencies.
60%: The response contains the correct answer only but with an incorrect thought process.
40%: The response contains an incorrect answer anywhere but also provides a cor-rect answer or correct thought process with minimal logical inconsistencies.
20%: The response contains an incorrect answer anywhere but provides enough helpful information to plausibly reach a correct answer.
0% The response contains an incorrect answer, too much unhelpful information, or not enough helpful information to plausibly reach a correct answer

QUESTION
{question}

PERFECT RESPONSE
{human_response}

STUDENT RESPONSE
{model_response} 

TASK
Does the STUDENT RESPONSE cointain the PERFECT RESPONSE only! Use the SCORING CRITERIA. Provide a full explanation and finally return a JSON object with the score as a percentage. Example:
{{"score": 20}}
"""
#%%
#print(create_eval_prompt(question='#', human_response='#', model_response='#'))


#%%
def extract_valid_json(s:str) -> dict:
    # Regex pattern for basic JSON structure: objects {} and arrays []
    json_pattern = re.compile(r'\{.*?\}|\[.*?\]', re.DOTALL)
    # Finding all matches that look like JSON
    potential_jsons = json_pattern.findall(s)
    if not potential_jsons:
        return None
    for pj in potential_jsons:
        try:
            # Attempt to parse the JSON
            valid_json = json.loads(pj)
            # Returning the first valid JSON found
            return valid_json
        except json.JSONDecodeError:
            return None
        
#%%
# print(extract_valid_json('{"score": 40}'))

# %%
def create_all_llm_eval_messages(all_llm_answers:dict[pd.DataFrame], benchmark_questions:dict) -> dict[str: list]:
    all_llm_eval_messages = {}
    for model, answers_df in all_llm_answers.items():
        answers_series = answers_df['model_answer']
        all_eval_prompts = []
        for idx, question in enumerate(benchmark_questions):
            eval_prompt = create_eval_prompt(
                question=question['question'],
                human_response=question['human_answer'],
                model_response=answers_series.iloc[idx],
            )
            all_eval_prompts.append(eval_prompt)
        eval_messages = [[{"role": "user", "content": p}] for p in all_eval_prompts]
        all_llm_eval_messages[model] = eval_messages
    return all_llm_eval_messages


def extract_all_scores(all_llm_eval_responses:dict) -> dict[list]:
    all_llm_eval_scores = {}
    for model, eval_responses in all_llm_eval_responses.items():
        llm_scores = []
        for eval in eval_responses:
            eval_response = message_parse(eval)
            score_json = extract_valid_json(eval_response)
            if score_json is None or 'score' not in score_json.keys():
                score_json = {'score': np.nan}
            elif isinstance(score_json['score'], str):
                score_json['score'] = score_json['score'].replace('%', '')
                try:
                    score_json['score'] = int(score_json['score'])
                except ValueError:
                    score_json['score'] = np.nan
            else:
                score_json['score'] = int(score_json['score'])
            llm_scores.append(score_json)
        all_llm_eval_scores[model] = llm_scores
    return all_llm_eval_scores


async def get_llm_eval_responses(
        llm_service:Callable, 
        all_llm_eval_messages:dict, 
        model:str, 
        hyperparams:dict,
    ) -> dict[list]:
    all_llm_eval_responses = {}
    for _model, eval_messages in all_llm_eval_messages.items():
        print(f"Running {_model} evaluation...")
        eval_responses = await runner(
            llm_service.completion,
            messages=eval_messages,
            model=model,
            **hyperparams,
        )
        all_llm_eval_responses[_model] = eval_responses
    return all_llm_eval_responses


def create_auto_eval_json(
        all_llm_scores:dict[list], 
        all_llm_eval_responses:dict[list], 
        all_llm_answers:dict[pd.DataFrame], 
        benchmark_questions:dict, 
        auto_eval_save_path:str,
    ):
    # Join the eval responses and scores with the benchmark questions
    all_auto_eval_results = {}
    for model, eval_responses in all_llm_eval_responses.items():
        final_results = copy.deepcopy(benchmark_questions)
        answers = all_llm_answers[model]['model_answer']
        scores = all_llm_scores[model]
        for idx, question in enumerate(final_results):
            question.update({'model_answer': answers.iloc[idx]})
            question.update({'eval_response': message_parse(eval_responses[idx])})
            question.update(scores[idx])

        final_df = pd.DataFrame(final_results)
        print(f"Auto Eval ->> Model: {model} | Mean score: {final_df['score'].mean()} | Std dev: {final_df['score'].std()}")
        os.makedirs(auto_eval_save_path, exist_ok=True)
        model_name = model_clean(model)
        final_df.set_index('index').to_json(f'{auto_eval_save_path}/auto_eval-{model_name}.json', orient='index')
        all_auto_eval_results[model] = final_df
    return all_auto_eval_results



