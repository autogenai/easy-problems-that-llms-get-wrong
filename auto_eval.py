# %%
from llm_service import message_parse, runner, litellm_service, custom_llm_service
from utils import message_parse, model_clean
from IPython.display import display
import pandas as pd
import numpy as np
import json
import copy
import re
import os


def create_eval_prompt(question: str, human_response: str, model_response: str) -> str:
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
Does the STUDENT RESPONSE contains the PERFECT RESPONSE only! Use the SCORING CRITERIA. Provide a full explanation and finally return a JSON object with the score as a percentage. Example:
{{"score": 20}}
"""


# %%
# print(create_eval_prompt(question='#', human_response='#', model_response='#'))


# %%
def extract_valid_json(string: str) -> dict:
    string_clean = string.replace("\n", "")
    # Regex pattern for basic JSON structure: objects {} and arrays []
    json_pattern = re.compile(r"\{.*?\}|\[.*?\]", re.DOTALL)
    # Finding all matches that look like JSON
    potential_jsons = json_pattern.findall(string_clean)
    if not potential_jsons:
        return None
    for pj in potential_jsons:
        try:
            # Attempt to parse the JSON
            valid_json = json.loads(pj)
            # Returning the first valid JSON found
            return valid_json
        except json.JSONDecodeError:
            continue
    return None


# %%
# print(extract_valid_json('{"score": 40}'))


# %%
def create_all_llm_eval_messages(
    all_llm_answers: dict[pd.DataFrame], benchmark_questions: dict
) -> dict[str:list]:
    all_llm_eval_messages = {}
    for model, answers_df in all_llm_answers.items():
        answers_series = answers_df["model_answer"]
        all_eval_prompts = []
        for idx, question in enumerate(benchmark_questions):
            eval_prompt = create_eval_prompt(
                question=question["question"],
                human_response=question["human_answer"],
                model_response=answers_series.iloc[idx],
            )
            all_eval_prompts.append(eval_prompt)
        eval_messages = [[{"role": "user", "content": p}] for p in all_eval_prompts]
        all_llm_eval_messages[model] = eval_messages
    return all_llm_eval_messages


def validation_func(response: str, json_key="score", list_of_values=None) -> bool:
    score_json = extract_valid_json(response)
    if (
        score_json is not None
        and isinstance(score_json, dict)
        and json_key in score_json.keys()
    ):
        if list_of_values is not None:
            correct_value_bool = score_json[json_key] in list_of_values
            return correct_value_bool
        return True
    return False


def extract_all_scores(all_llm_eval_responses: dict) -> dict[list]:
    all_llm_eval_scores = {}
    for model, eval_responses in all_llm_eval_responses.items():
        llm_scores = []
        for eval in eval_responses:
            eval_response = message_parse(eval, model)
            score_json = extract_valid_json(eval_response)
            if score_json is None or "score" not in score_json.keys():
                score_json = {"score": np.nan}
            elif isinstance(score_json["score"], str):
                score_json["score"] = score_json["score"].replace("%", "")
                try:
                    score_json["score"] = int(score_json["score"])
                except ValueError:
                    score_json["score"] = np.nan
            else:
                score_json["score"] = int(score_json["score"])
            llm_scores.append(score_json)
        all_llm_eval_scores[model] = llm_scores
    return all_llm_eval_scores


async def get_llm_eval_responses(
    all_llm_eval_messages: dict,
    model_info: str,
    hyperparams: dict,
    validation_func: lambda x: True,
) -> dict[list]:
    all_llm_eval_responses = {}
    for _model, eval_messages in all_llm_eval_messages.items():
        print(f"Running {_model} evaluation...")
        llm_service_func = (
            litellm_service() if model_info[1] == "llmlite" else custom_llm_service()
        )
        eval_responses = await runner(
            llm_service_func.completion,
            messages=eval_messages,
            model=model_info[0],
            validation_func=validation_func,
            **hyperparams,
        )
        all_llm_eval_responses[_model] = eval_responses
    return all_llm_eval_responses


def create_auto_eval_json(
    all_llm_scores: dict[list],
    all_llm_eval_responses: dict[list],
    all_llm_answers: dict[pd.DataFrame],
    benchmark_questions: dict,
    auto_eval_save_path: str,
):
    # Join the eval responses and scores with the benchmark questions
    all_auto_eval_results = {}
    for model, eval_responses in all_llm_eval_responses.items():
        final_results = copy.deepcopy(benchmark_questions)
        answers = all_llm_answers[model]["model_answer"]
        scores = all_llm_scores[model]
        for idx, question in enumerate(final_results):
            question.update({"model_answer": answers.iloc[idx]})
            question.update(
                {"eval_response": message_parse(eval_responses[idx], model)}
            )
            question.update(scores[idx])

        final_df = pd.DataFrame(final_results)
        print(
            f"Auto Eval ->> Model: {model} | Mean score: {final_df['score'].mean()} | Std dev: {final_df['score'].std()}"
        )
        os.makedirs(auto_eval_save_path, exist_ok=True)
        model_name = model_clean(model)
        final_df.set_index("index").to_json(
            f"{auto_eval_save_path}/auto_eval-{model_name}.json",
            orient="index",
        )
        all_auto_eval_results[model] = final_df
    return all_auto_eval_results


def score_multiple_choice_answers(
    all_llm_answers: dict[pd.DataFrame], auto_eval_save_path: str
):
    all_llm_answers = {
        model: data.reset_index() for model, data in all_llm_answers.items()
    }
    for model, answers_df in all_llm_answers.items():
        for idx, answer_row in answers_df.iterrows():
            json_answer = extract_valid_json(str(answer_row["model_answer"]))
            json_answer_letter = None
            if json_answer is None or "ANSWER" not in json_answer.keys():
                score = 0
            else:
                json_answer_letter = json_answer["ANSWER"]
                correct = json_answer["ANSWER"] == answer_row["correct_letter"]
                score = 100 if correct else 0
            all_llm_answers[model].loc[idx, "json_answer"] = str(json_answer)
            all_llm_answers[model].loc[idx, "json_answer_letter"] = json_answer_letter
            all_llm_answers[model].loc[idx, "invalid_answer_letter"] = int(
                json_answer_letter not in ["A", "B", "C", "D"]
            )
            all_llm_answers[model].loc[idx, "score"] = score
        final_df = copy.deepcopy(all_llm_answers[model])
        # display(final_df)
        print(
            f"Auto Eval ->> Model: {model} | Mean score: {final_df['score'].mean()} | Std dev: {final_df['score'].std()}"
        )
        os.makedirs(auto_eval_save_path, exist_ok=True)
        model_name = model_clean(model)
        final_df.reset_index().to_json(
            f"{auto_eval_save_path}/auto_eval-{model_name}.json", orient="index"
        )
    return all_llm_answers
