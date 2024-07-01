# %%
from IPython.display import display
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
        question.update({'model_answer': message_parse(answers[idx], model)})
        question.update({'score': ''})
    answers_df = pd.DataFrame(final_answers)
    model_name = model_clean(model)
    os.makedirs(answers_save_path, exist_ok=True)
    if 'index' not in answers_df.columns:
        answers_df.reset_index(inplace=True)
    answers_df.set_index('index').to_json(f'{answers_save_path}/final_answers-{model_name}.json', orient='index')
    return answers_df


async def get_llm_answers(
    llm_service:Callable, 
    benchmark_questions:dict[list[dict]], 
    models:list, 
    hyperparams:dict, 
    answers_save_path:str,
    multiple_choice=False,
    validation_func=lambda x: True,
) -> dict[pd.DataFrame]:
    question_str = 'question' if not multiple_choice else 'multi_choice_question'
    all_llm_answers = {}
    for model in models:
        print(f"Running  Benchmark for {model}")
        messages = [[{"role": "user", "content": q[question_str]}] 
                    for q in benchmark_questions[model_clean(model)]]
        answers = await runner(
            llm_service.completion,
            messages=messages,
            model=model,
            validation_func=validation_func,
            **hyperparams,
        )
        answers_df = save_answers_as_json(answers, benchmark_questions[model_clean(model)], model, answers_save_path)
        all_llm_answers[model] = answers_df
    return all_llm_answers

# %%
# from llm_service import litellm_service
# import json

# litellm_query = litellm_service()
# benchmark_questions = json.load(open('linguistic_benchmark.json', 'r'))
# models = ["mistral/open-mixtral-8x22b"]
# questions = {m: benchmark_questions for m in models]
# all_llm_answers = await get_llm_answers(
#         llm_service=litellm_query, 
#         benchmark_questions=questions, 
#         models=["mistral/open-mixtral-8x22b"], 
#         hyperparams={'batch_size': 10, 'temperature': 0, 'max_tokens': 50, 'num_retries': 2},
#         answers_save_path='./answers-test'
# )
# print(all_llm_answers)


# %%
def load_all_llm_answers_from_json(
        answers_save_path:str, 
        prefix_replace='final_answers-',
        sub_folders=[''],
) -> list[dict[pd.DataFrame]]:
    # reload all the scored answers from json files
    all_llm_answers = {}
    for sub_folder in sub_folders:
        answers_save_path_sub = f"{answers_save_path}{sub_folder}"
        if not os.path.exists(answers_save_path_sub):
            continue
        for output_file in os.listdir(f"{answers_save_path_sub}/"):
            if output_file.endswith(".json"):
                outputs_df = pd.read_json(f"{answers_save_path_sub}/{output_file}", orient='index')
                model = output_file.replace(prefix_replace, '').replace('.json', '')
                all_llm_answers.setdefault(model, pd.DataFrame())
                all_llm_answers[model] = pd.concat([all_llm_answers[model], outputs_df])
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
            'output_count': len(outputs),
        }
    return all_llm_stats


def get_llm_stats(all_llm_answers:dict, stats_save_path:str, bootstrap_n=10000) -> pd.DataFrame:
    all_llm_stats = calculate_llm_stats(all_llm_answers, bootstrap_n)
    stats_df = pd.DataFrame(all_llm_stats).transpose().sort_values('mean_score', ascending=False)
    stats_df.index.name = 'model'
    os.makedirs(stats_save_path, exist_ok=True)
    stats_df.to_csv(f'./{stats_save_path}/final_stats.csv')
    return stats_df

# %%
