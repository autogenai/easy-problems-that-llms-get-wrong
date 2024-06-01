# %%
from collections.abc import Callable
from llm_service import message_parse
from llm_service import runner
import pandas as pd
import numpy as np
import copy
import os


def model_clean(model_str:str) -> str:
    model_str = model_str.split('/')[-1]
    model_str = model_str.replace('.', '_').replace(':', '_')
    return model_str

def save_answers_as_json(answers:list, benchmark_questions:dict, model:str, answers_save_path:str) -> pd.DataFrame:
    if len(answers) != len(benchmark_questions):
        print(f"Error: Number of answers {len(answers)} does not match number of questions {len(benchmark_questions)}")
        return pd.DataFrame()
    final_answers = copy.deepcopy(benchmark_questions)
    for idx, question in enumerate(final_answers):
        question.update({'model_answer': message_parse(answers[idx])})
        question.update({'score': ''})
    answers_df = pd.DataFrame(final_answers)
    model_name = model_clean(model)
    os.makedirs(answers_save_path, exist_ok=True)
    answers_df.set_index('index').to_json(f'{answers_save_path}/final_answers-{model_name}.json', orient='index')
    return answers_df


async def get_llm_answers(
        llm_service:Callable, 
        benchmark_questions:dict, 
        models:list, 
        hyperparams:dict, 
        answers_save_path:str
    ) -> dict[pd.DataFrame]:
    messages = [[{"role": "user", "content": q['question']}] 
                for q in benchmark_questions]
    all_llm_answers = {}
    for model in models:
        print(f"Running  Benchmark for {model}")
        answers = await runner(
            llm_service.completion,
            messages=messages,
            model=model,
            **hyperparams,
        )
        answers_df = save_answers_as_json(answers, benchmark_questions, model, answers_save_path)
        all_llm_answers[model] = answers_df
    return all_llm_answers

# %%
# from llm_service import litellm_service
# import json

# litellm_query = litellm_service()
# benchmark_questions = json.load(open('linguistic_benchmark.json', 'r'))
# all_llm_answers = await get_llm_answers(
#         llm_service=litellm_query, 
#         benchmark_questions=benchmark_questions, 
#         models=["mistral/open-mixtral-8x22b"], 
#         hyperparams={'batch_size': 10, 'temperature': 0, 'max_tokens': 2048},
#         answers_save_path='./answers-test'
# )
# print(all_llm_answers)


# %%
def load_all_llm_answers_from_json(answers_save_path:str, prefix_replace='final_answers-') -> dict[pd.DataFrame]:
    # reload all the scored answers from json files
    if not os.path.exists(answers_save_path):
        return {}
    all_llm_answers = {}
    for output_file in os.listdir(f"{answers_save_path}/"):
        if output_file.endswith(".json"):
            outputs_df = pd.read_json(f"{answers_save_path}/{output_file}", orient='index')
            model = output_file.replace(prefix_replace, '').replace('.json', '')
            all_llm_answers[model] = outputs_df
    return all_llm_answers


def calculate_llm_stats(all_llm_answers:dict, bootstrap_n=10000) -> dict:
    all_llm_stats = {}
    for model, outputs in all_llm_answers.items():
        print(f"Calculating stats for {model}")
        mean_score = outputs['score'].mean()
        std_dev_score = outputs['score'].std()
        # do a n(10,000) bootstrap to get the 95% CI
        bootstrap_scores = []
        for _ in range(bootstrap_n):
            bootstrap_scores.append(outputs['score'].sample(frac=1, replace=True).mean())
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        # caculate z-interval 95%
        z = 1.96
        z_interval_error = z * (std_dev_score / np.sqrt(len(outputs)))
        all_llm_stats[model] = {
            'mean_score': mean_score, 
            'std_dev_score': std_dev_score, 
            'z_interval_error': z_interval_error, 
            'ci_lower': ci_lower, 
            'ci_upper': ci_upper,
        }
    return all_llm_stats


def get_llm_stats(all_llm_answers:dict, stats_save_path:str, bootstrap_n=10000) -> pd.DataFrame:
    all_llm_stats = calculate_llm_stats(all_llm_answers, bootstrap_n)
    stats_df = pd.DataFrame(all_llm_stats).transpose().sort_values('mean_score', ascending=False).round(0)
    stats_df.index.name = 'model'
    os.makedirs(stats_save_path, exist_ok=True)
    stats_df.to_csv(f'./{stats_save_path}/final_stats.csv')
    return stats_df

# %%
