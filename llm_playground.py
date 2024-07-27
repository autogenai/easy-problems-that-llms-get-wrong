import streamlit as st
import openai
import json
from llm_service import message_parse, runner, litellm_service, custom_llm_service
from utils import model_clean

answer_models = {
    "gpt-4-turbo-preview": "litellm",
    "gpt-4o": "litellm",
    "gpt-4o-mini-2024-07-18": "litellm",
    "bedrock/meta.llama3-70b-instruct-v1:0": "litellm",
    "Meta-Llama-3-1-70B-Instruct-ostu.eastus.models.ai.azure.com": "custom",
    "Meta-Llama-3-1-405B-Instruct-jjo.eastus.models.ai.azure.com": "custom",
    "mistral-large-latest": "custom",
    "mistral/open-mixtral-8x22b": "litellm", 
    "claude-3-opus-20240229": "litellm", 
    "vertex_ai/gemini-1.5-pro": "litellm", 
    "command-r-plus": "litellm",
    "claude-3-5-sonnet-20240620": "litellm",
}

mapper = {
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini-2024-07-18': 'GPT-4o Mini',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet',
    'gemini-1_5-pro': 'Gemini 1.5 Pro',
    'gemini-1_0-pro': 'Gemini 1.0 Pro',
    'mistral-large-latest': 'Mistral Large 2',
    'open-mixtral-8x22b': 'Mistral 8x22B',
    'meta_llama3-70b-instruct-v1_0': 'Llama 3 70B',
    'meta_llama3-1-70b-instruct-v1_0': 'Llama 3.1 70B',
    'command-r': 'Command R',
    'command-r-plus': 'Command R Pro',
    'Meta-Llama-3-1-405B-Instruct-jjo_eastus_models_ai_azure_com': 'Llama 3.1 405B',
    'Meta-Llama-3-1-70B-Instruct-ostu_eastus_models_ai_azure_com': 'Llama 3.1 70B',
}

# Streamlit app layout
from typing import List, Dict


# Available models
MODELS = list(answer_models.keys())  # Add more models as needed
st.session_state['rerun'] = False

def get_assistant_response(model: str, messages: List[Dict[str, str]], temperature=0.5, max_tokens=2048, stream=True) -> str:
    execution_func = litellm_service() if answer_models[model] == "litellm" else custom_llm_service()
    response = execution_func.completion(model=model, messages=messages, num_retries=2, temperature=temperature, max_tokens=max_tokens, stream=stream)
    if 'azure' in model:
        stream = False
    if stream is False:
        clean_response = message_parse(response, model)
        yield clean_response
        return

    collected_chunks = []
    collected_messages = []

    print('\n\n-----------------------------------\n\n')
    print('response (playground):', response)
    for chunk in response:
        print('chunk:', chunk)
        collected_chunks.append(chunk)
        chunk_message = chunk['choices'][0]['delta']
        if chunk_message.get('content') is not None:
            collected_messages.append(chunk_message)
            full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        yield full_reply_content


st.title("LLM Chat Playground")

# Sidebar for model selection
selected_model = st.sidebar.selectbox("Choose a model", MODELS)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
max_tokens = st.sidebar.slider("Max tokens", min_value=64, max_value=4096, value=2048, step=64)
stream = st.sidebar.checkbox("Stream response", value=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to delete a message
def delete_message(index: int):
    if 0 <= index < len(st.session_state.messages):
        del st.session_state.messages[index]
        st.rerun()

def rerun_from_message(index):
    st.session_state.messages = st.session_state.messages[:index+1]
    st.session_state['rerun'] = True

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
       # Create columns for buttons
        col1, col2 = st.columns([0.07, 0.5])
        with col1:
            if st.button("Delete", key=f"delete_{i}"):
                delete_message(i)
        with col2:
            if st.button("Rerun", key=f"rerun_{i}"):
                rerun_from_message(i)


# Accept user input
prompt = st.chat_input("What is your question?")
if prompt or st.session_state['rerun']:
    if st.session_state.rerun is False:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
    else:
        st.session_state.rerun = False

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in get_assistant_response(selected_model, st.session_state.messages, temperature=temperature, max_tokens=max_tokens, stream=stream):
            full_response = response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()