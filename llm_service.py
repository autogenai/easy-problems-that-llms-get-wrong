# %%
import os
import json
import time
import boto3
import asyncio
import litellm
import requests
import subprocess
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
os.environ["AZURE_AI_KEYS"] = config("AZURE_AI_KEYS")
os.environ["VERTEX_PROJECT"] = config("VERTEX_PROJECT")
os.environ["VERTEX_LOCATION"] = config("VERTEX_LOCATION")
os.environ["GOOGLE_AI_STUDIO"] = config("GOOGLE_AI_STUDIO")

litellm.vertex_project = config("VERTEX_PROJECT")
litellm.vertex_location = config("VERTEX_LOCATION")
litellm.set_verbose = True


# %%
def openai_message_parse(response: dict):
    try:
        messages = [m["message"]["content"] for m in response["choices"]]
    except Exception as e:
        if response == {}:
            return "None"
        else:
            raise ValueError(
                f"Response does not contain the expected key 'choices'. Error: {e}. Response: {response}"
            )
    if len(messages) == 1:
        messages = messages[0]
    return messages


def anthropic_message_parse(response: dict):
    try:
        messages = response["content"][0]["text"]
    except Exception as e:
        if response == {}:
            return "None"
        else:
            raise ValueError(
                f"Response does not contain the expected key 'choices'. Error: {e}. Response: {response}"
            )
    if len(messages) == 1:
        messages = messages[0]
    return messages


def google_message_parse(response: dict | list):
    try:
        if isinstance(response, dict):
            text_response = response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            text_response = "".join(
                [
                    line["candidates"][0]["content"]["parts"][0]["text"]
                    for line in response
                    if "candidates" in line and "content" in line["candidates"][0]
                ]
            )
        return text_response
    except Exception as e:
        if response in [{}, []]:
            return "None"
        else:
            raise ValueError(
                f"Response does not contain the expected key 'candidates'. Error: {e}. Response: {response}"
            )


def message_parse(response: dict, model: str):
    if response == {}:
        return ""
    if str(response)[:17] == "ModelResponse(id=":
        messages = openai_message_parse(response)
    elif model in ["gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4o", "o1-preview"]:
        messages = openai_message_parse(response)
    elif "azure" in model or "mistral" in model or "mixtral" in model:
        messages = openai_message_parse(response)
    elif "gemini" in model:
        messages = google_message_parse(response)
    elif "llama" in model:
        messages = response
    elif "claude" in model:
        messages = anthropic_message_parse(response)
    else:
        print(f"!!!!!! Could not determine model in `message_parse`: '{model}'")
        messages = ""
    return messages


def litellm_service():
    return litellm


# %%
## Test LiteLLM Service
# litellm_query = litellm_service()

# messages = [{ "content": "Write a sentence where every word starts with the letter A.","role": "user"}]

# models = ["gpt-4-turbo-preview", "meta.llama3-70b-instruct-v1:0", "command-r-plus", "mistral/mistral-large-latest", "mistral/open-mixtral-8x22b", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "vertex_ai/gemini-1.5-pro", "vertex_ai/gemini-1.0-pro"]

# model = "command-r-plus"

# response = litellm_query.completion(model=model, messages=messages, num_retries=2)
# print(response)

# message_parse(response, model)


# %%
class custom_llm_service:
    def __init__(self):
        pass

    def openai_query(
        self,
        messages: list,
        model="gpt-4-turbo-preview",
        max_tokens=1000,
        temperature=0,
        n=1,
        stream=False,
    ):
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
                "max_completion_tokens": max_tokens,
                "temperature": temperature if model not in ["o1-preview"] else 1,
                "n": n,
                "stream": stream,
            },
            timeout=300,
        )
        return response.json()

    def anthropic_query(
        self,
        messages: list,
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        n=1,
        stream=False,
    ):
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
                "stream": stream,
            },
        )
        return response.json()

    def bedrock_query(
        self,
        messages: list,
        model="meta.llama3-8b-instruct-v1:0",
        max_tokens=2048,
        temperature=0,
        n=1,
        stream=False,
    ):
        # Initialize a session using your AWS credentials
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-west-2",
        )

        # Initialize the Bedrock client
        bedrock_client = session.client("bedrock-runtime", region_name="us-west-2")

        # Embed the prompt in Llama 3's instruction format.
        formatted_prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {messages[0]['content']}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        try:
            # Invoke the model with the request.
            response = bedrock_client.invoke_model(modelId=model, body=request)
            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            response_text = model_response["generation"]
            # print(response_text)
            return response_text

        except Exception as e:
            print(f"ERROR: Can't invoke '{model}'. Reason: {e}")

    def azure_query(
        self,
        messages: list,
        model="Meta-Llama-3-1-405B-Instruct-jjo.eastus.models.ai.azure.com",
        max_tokens=1000,
        temperature=0,
        n=1,
        stream=False,
    ):
        ###
        stream = False  ###!!!!
        ###

        azure_ai_keys = json.loads(os.environ["AZURE_AI_KEYS"])
        url = f"https://{model}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": azure_ai_keys[model],
        }
        headers.update({"Accept": "text/event-stream"}) if stream else None

        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def google_vertex_query(
        self,
        messages: list,
        model="gemini-experimental",
        max_tokens=8192,
        temperature=0,
        n=1,
        stream=False,
    ):
        # Define the request payload
        """
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            },
        ]
        """
        messages_google_format = [
            {"role": message["role"], "parts": [{"text": message["content"]}]}
            for message in messages
        ]
        payload = {
            "contents": messages_google_format,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.95,
            },
        }

        # Project and API details
        PROJECT_ID = config("VERTEX_PROJECT")
        LOCATION_ID = config("VERTEX_LOCATION")
        API_ENDPOINT = f'{config("VERTEX_LOCATION")}-aiplatform.googleapis.com'
        MODEL_ID = model

        # Construct the URL
        url = f"https://{API_ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{LOCATION_ID}/publishers/google/models/{MODEL_ID}:streamGenerateContent"

        # Get the access token
        def get_access_token():
            token = (
                subprocess.check_output("gcloud auth print-access-token", shell=True)
                .decode()
                .strip()
            )
            return token

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_access_token()}",
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        return response.json()

    def google_ai_studio_query(
        self,
        messages: list,
        model="gemini-1.5-pro-exp-0801",
        max_tokens=8192,
        temperature=0,
        n=1,
        stream=False,
    ):
        """
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            },
        ]
        """
        messages_google_format = [
            {"role": message["role"], "parts": [{"text": message["content"]}]}
            for message in messages
        ]

        API_KEY = config("GOOGLE_AI_STUDIO")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "contents": messages_google_format,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "text/plain",
            },
        }

        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        if "error" in response_json:
            raise ValueError(f"Error in google ai studio response: {response_json}")
        return response_json

    def mistral_query(
        self,
        messages: list,
        model="mistral-large-latest",
        max_tokens=1000,
        temperature=0,
        n=1,
        stream=False,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers
        )
        return response.json()

    def completion(
        self, messages: list, model="gpt-4-turbo-preview", num_retries=2, **kwargs
    ):
        # Add in num_retry logic to match the LiteLLM service
        if model in ["gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4o", "o1-preview"]:
            response = custom_llm_service.openai_query(
                self, messages=messages, model=model, **kwargs
            )
        elif "claude" in model:
            response = custom_llm_service.anthropic_query(
                self, messages=messages, model=model, **kwargs
            )
        elif "gemini" in model:
            response = custom_llm_service.google_ai_studio_query(
                self, messages=messages, model=model, **kwargs
            )
        elif "azure" in model:
            response = custom_llm_service.azure_query(
                self, messages=messages, model=model, **kwargs
            )
        elif "mistral" in model or "mixtral" in model:
            response = custom_llm_service.mistral_query(
                self, messages=messages, model=model, **kwargs
            )
        elif "llama" in model:
            response = custom_llm_service.bedrock_query(
                self, messages=messages, model=model, **kwargs
            )
        else:
            raise ValueError(f"Model '{model}' not supported")
        return response


# #%%
# ##Test Custom LLM Service
# model = "gemini-1.5-pro-exp-0801"
# custom_llm_service_obj = custom_llm_service()
# response = custom_llm_service_obj.completion(
#     messages=[{"role": "user","content": "10 + 10 ="}],
#     model=model,
#     max_tokens=100,
#     temperature=0,
#     n=1,
# )
# print(response)
# print('\n-----------\n')
# print(message_parse(response, model))


# %%
async def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: func(*args, **kwargs))
    return result


async def runner(
    func, messages: list, batch_size=1, validation_func=lambda x: True, **kwargs
):
    all_responses = [{}] * len(messages)
    messages_copy = {i: message for i, message in enumerate(messages)}
    for idx in range(0, len(messages), batch_size):
        print(f"> Processing batch {idx + 1}-{idx + batch_size} ex {len(messages)}")
        for _retry in range(1, 4):
            batch_messages = {
                i: message
                for i, message in messages_copy.items()
                if i in list(range(idx, idx + batch_size))
            }
            try:
                _responses = await asyncio.gather(
                    *(
                        run_in_executor(func, messages=_messages, **kwargs)
                        for _messages in batch_messages.values()
                    )
                )
            except Exception as e:
                print(f"Error getting response: {e}. Retry #{_retry}")
                time.sleep(30)
                continue
            responses = dict(zip(batch_messages.keys(), _responses))
            for _idx, response in responses.items():
                if validation_func(message_parse(response, kwargs["model"])):
                    all_responses[_idx] = response
                    del messages_copy[_idx]
                else:
                    print(f"Validation failed on response {_idx}. Retry #{_retry}")
                    print(f"Invalid Response: {response}")

            if len(messages_copy) == 0:
                break
            else:
                time.sleep(30)

    return all_responses


# %%
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
