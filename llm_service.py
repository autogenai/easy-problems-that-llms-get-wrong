# %%
import os
import time
import asyncio
import litellm
import requests
from decouple import AutoConfig
from concurrent.futures import ThreadPoolExecutor

config = AutoConfig(search_path=".env")

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = config("COHERE_API_KEY")
os.environ["MISTRAL_API_KEY"] = config("MISTRAL_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = config("ANTHROPIC_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = config("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = config("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_REGION_NAME"] = "us-west-2"
litellm.vertex_project = config("VERTEX_PROJECT")
litellm.vertex_location = config("VERTEX_LOCATION")


# %%
def openai_message_parse(response:dict):
    try:
        messages = [m["message"]["content"] for m in response["choices"]]
    except Exception as e:
        if response == {}:
            return "None"
        else:
            raise ValueError(f"Response does not contain the expected key 'choices'. Error: {e}. Response: {response}")
    if len(messages) == 1:
        messages = messages[0]
    return messages


def anthropic_message_parse(response:dict):
    try:
        messages = response['content'][0]['text']
    except Exception as e:
        if response == {}:
            return "None"
        else:
            raise ValueError(f"Response does not contain the expected key 'choices'. Error: {e}. Response: {response}")
    if len(messages) == 1:
        messages = messages[0]
    return messages


def message_parse(response:dict, model:str):
    if response == {}:
        return ""
    if str(response)[:17] == 'ModelResponse(id=':
        messages = openai_message_parse(response)
    elif model in ['gpt-4-turbo-preview', 'gpt-4-turbo', 'gpt-4o']:
        messages = openai_message_parse(response)
    elif 'claude' in model:
        messages = anthropic_message_parse(response)
    else:
        print(f"!!!!!! Could not determing model in `message_parse`: '{model}'")
        messages = ""
    return messages


def litellm_service():
    return litellm


# %%
## Test LiteLLM Service
# litellm_query = litellm_service()

# messages = [{ "content": "Write a sentence where every word starts with the letter A.","role": "user"}]

# models = ["gpt-4-turbo-preview", "meta.llama3-70b-instruct-v1:0", "command-r", "mistral/mistral-large-latest", "mistral/open-mixtral-8x22b", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "vertex_ai/gemini-1.5-pro", "vertex_ai/gemini-1.0-pro"]

# model = "claude-3-5-sonnet-20240620"

# response = litellm_query.completion(model=model, messages=messages, num_retries=2)
# print(response)

# message_parse(response, model)


# %%
class custom_llm_service:
    def __init__(self):
        pass

    def openai_query(self, messages:list, model="gpt-4-turbo-preview", max_tokens=1000, temperature=0, n=1):
        """
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            },
        ]
        """
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": n,
            },
        )
        return response.json()
    

    def anthropic_query(self, messages:list, model="claude-3-5-sonnet-20240620", max_tokens=1000, temperature=0, n=1):
        """
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            },
        ]
        """
        if n > 1:
            print('The "n" variable is not supported by Anthropic')
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": f"{config('ANTHROPIC_API_KEY')}",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.json()


    def completion(self, messages: list, model="gpt-4-turbo-preview", num_retries=2, **kwargs):
        # Add in num_retry logic to match the LiteLLM service
        if model in ['gpt-4-turbo-preview', 'gpt-4-turbo', 'gpt-4o']:
            response = custom_llm_service.openai_query(self, messages=messages, model=model, **kwargs)
        elif 'claude' in model:
            response = custom_llm_service.anthropic_query(self, messages=messages, model=model, **kwargs)
        else:
            raise ValueError(f"Model '{model}' not supported")
        return response


#%%
# ##Test Custom LLM Service
# custom_llm_service_obj = custom_llm_service()
# response = custom_llm_service_obj.completion(
#     messages=[{"role": "user","content": "What is the meaning of life?"}],
#     model="claude-3-5-sonnet-20240620",
#     max_tokens=4092,
#     temperature=0,
#     n=1,
# )
# print(response)
# print(message_parse(response, "claude-3-5-sonnet-20240620"))


# %%
async def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: func(*args, **kwargs))
    return result


async def runner(func, messages:list, batch_size=1, validation_func=lambda x: True, **kwargs):
    all_responses = [{}] * len(messages)
    messages_copy = {i: message for i, message in enumerate(messages)}
    for idx in range(0, len(messages), batch_size):
        print(f"> Processing batch {idx + 1}-{idx + batch_size} ex {len(messages)}")
        for _retry in range(1, 4):
            batch_messages = {i: message for i, message in messages_copy.items() 
                              if i in list(range(idx, idx + batch_size))}
            _responses = await asyncio.gather(
                *(
                    run_in_executor(func, messages=_messages, **kwargs)
                    for _messages in batch_messages.values()
                )
            )
            responses = dict(zip(batch_messages.keys(), _responses))
            for _idx, response in responses.items():
                if validation_func(message_parse(response, kwargs['model'])):
                    all_responses[_idx] = response
                    del messages_copy[_idx]
                else:
                    print(f"Validation failed on response {_idx}. Retry #{_retry}")

            if len(messages_copy) == 0:
                break
            else:
                time.sleep(5)

    return all_responses




#%%
# messages = [{"role": "user", "content": "What is the meaning of life?"}]
# hyperparams = {
#     "max_tokens": 500, 
#     "temperature": 0, 
#     "num_retries": 1
# }
# responses = await runner(
#     litellm.completion, 
#     messages=[messages] * 5, 
#     batch_size=5,
#     model="vertex_ai/gemini-1.5-pro", 
#     **hyperparams,
# )

# for response in responses:
#     print(message_parse(response))
#     print('\n------------------\n')

# %%
