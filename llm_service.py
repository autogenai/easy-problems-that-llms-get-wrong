# %%
import os
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
def message_parse(response:dict):
    messages = [m["message"]["content"] for m in response["choices"]]
    if len(messages) == 1:
        messages = messages[0]
    return messages


def litellm_service():
    return litellm


# %%
### Test LiteLLM Service
# litellm_query = litellm_service()

# messages = [{ "content": "Write a sentence where every word starts with the letter A.","role": "user"}]

# models = ["gpt-4-turbo-preview", "meta.llama3-70b-instruct-v1:0", "command-r", "mistral/mistral-large-latest", "mistral/open-mixtral-8x22b", "claude-3-opus-20240229", "vertex_ai/gemini-1.5-pro", "vertex_ai/gemini-1.0-pro"]

# response = litellm_query.completion(model="vertex_ai/gemini-1.5-pro", messages=messages, num_retries=2)

# message_parse(response)


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

    def completion(self, messages: list, model="gpt-4-turbo-preview", num_retries=2, **kwargs):
        # Add in num_retry logic to match the LiteLLM service
        if model in ['gpt-4-turbo-preview', 'gpt-4-turbo']:
            response = custom_llm_service.openai_query(self, messages=messages, model=model, **kwargs)
        return response


# %%
# Test OpenAI Service
# response = custom_llm_service.completion(
#     messages=[{"role": "user","content": "What is the meaning of life?"}],
#     model="gpt-4-turbo-preview",
#     max_tokens=10,
#     temperature=0,
#     n=1,
# )
# print(response)


# %%
async def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: func(*args, **kwargs))
    return result


async def runner(func, messages:list, batch_size=1, **kwargs):
    all_responses = []
    for idx in range(0, len(messages), batch_size):
        print(f"> Processing batch {idx + 1}-{idx + batch_size} ex {len(messages)}")
        batch_messages = messages[idx : idx + batch_size]
        responses = await asyncio.gather(
            *(
                run_in_executor(func, messages=_messages, **kwargs)
                for _messages in batch_messages
            )
        )
        all_responses.extend(responses)
    return all_responses


#%%
# messages = [{"role": "user", "content": "What is the meaning of life?"}]
# hyperparams = {
#     "max_tokens": 50, 
#     "temperature": 0.5, 
#     "num_retries": 1
# }
# responses = await runner(
#     litellm.completion, 
#     messages=[messages] * 2, 
#     batch_size=5,
#     model="claude-3-opus-20240229", 
#     **hyperparams,
# )

# for response in responses:
#     print(message_parse(response))
#     print('\n------------------\n')

# %%
