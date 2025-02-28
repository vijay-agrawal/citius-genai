import os
import openai
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env
load_dotenv()

# Set page config
st.set_page_config(page_title="Mudra Saarthi", page_icon="💬")
st.title("Welcome to Mudra Saarthi")

azure_config = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_MODEL_NAME": os.getenv("AZURE_OPENAI_MODEL_NAME"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION")
    }

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_config["AZURE_OPENAI_ENDPOINT"],
    api_key=azure_config["AZURE_OPENAI_API_KEY"],
    api_version=azure_config["AZURE_OPENAI_API_VERSION"]
)

# Define a system prompt that returns a welcome message + instructions.
system_prompt = """
You are a helpful chatbot.
In your very first response, greet the user with the following welcome message: 
Hello Vijay, how may I help you?
"""
# Called when user submits a prompt:
# 1) Appends the user input to history
# 2) Calls Azure OpenAI
# 3) Appends the assistant response
# 4) Resets the text input
def submit():
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=azure_config["AZURE_OPENAI_MODEL_NAME"],
            messages=st.session_state.messages
        )
        assistant_output = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_output})
    
    # Clear the text input for the next user prompt
    st.session_state.user_input = ""

# Initialize session state for storing the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# If it's the first run, call the system prompt and store the welcome message
if not st.session_state.messages:
    response = client.chat.completions.create(
        model=azure_config["AZURE_OPENAI_MODEL_NAME"],
        messages=[{"role": "system", "content": system_prompt}]
    )
    system_output = response.choices[0].message.content
    st.session_state.messages.append({"role": "system", "content": system_prompt})
    st.session_state.messages.append({"role": "assistant", "content": system_output})

# Define custom labels for roles
role_labels = {
    "system": "System",
    "user": "You",
    "assistant": "Mudra Saarthi"  # <-- Custom label for assistant messages
}

# Display messages from the history/previous conversations
for msg in st.session_state.messages:
    # Only display user and assistant messages
    if msg["role"] in ["user", "assistant"]:    
        role = role_labels.get(msg["role"], msg["role"].capitalize())
        st.markdown(f"**{role}:** {msg['content']}")

# Text input with on_change callback
st.text_input(
    "Your answer:",
    value="",
    key="user_input",
    on_change=submit
)