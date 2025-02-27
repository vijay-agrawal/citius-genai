import streamlit as st
from dotenv import load_dotenv
import os
from openai import AzureOpenAI


# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title="Azure OpenAI Chatbot", page_icon="💬")

# Configure OpenAI with Azure credentials
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

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_completion(messages):
    """Get completion from Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Chat interface
st.title("My Chatbot")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Input for new message
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get messages in format for API
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    # Display assistant thinking indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Get response from API
        response = get_completion(api_messages)
        
        if response:
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Update message placeholder with response
            message_placeholder.markdown(response)
        else:
            message_placeholder.markdown("Sorry, I encountered an error. Please check your .env file and try again.")