import os
import warnings
import streamlit as st
import subprocess
import tempfile
import re
import ast
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

def initialize_model(model_name: str = "gemma2-9b-it"):
    """Initializes and returns a chat model with API key from environment variables."""
    os.environ["GROQ_API_KEY"] = "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"
    os.environ["OPENAI_API_KEY"] = "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"
    warnings.filterwarnings('ignore')
    return init_chat_model(model_name, model_provider="groq")

def initialize_chain(model):
    """Initializes an LLMChain with memory and structured prompting."""
    memory = ConversationBufferWindowMemory(k=50)
    
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=(
            "\U0001F916 You are an advanced AI assistant skilled in helping users write efficient, "
            "well-structured, and professional code. \U0001F4DA You understand various frameworks, "
            "libraries, and best practices. Below is the conversation history:\n"
            "{history}\n"
            "\U0001F4AC User's request: {input}\n"
            "\U0001F680 Provide a clear, optimized, and professional response including necessary "
            "frameworks, libraries, and best practices. If applicable, give examples. Use appropriate emojis based on context."
        )
    )

    chain = LLMChain(
        llm=model,
        prompt=prompt_template,
        memory=memory
    )
    return chain

def extract_code(response: str):
    """Extracts the first Python code block from a response."""
    match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    return match.group(1) if match else None

def is_valid_python(code: str):
    """Checks if the given code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def execute_code(code: str):
    """Executes the given Python code and returns the output."""
    if not is_valid_python(code):
        return "⚠️ Syntax Error: The generated code has a syntax issue. Please review and try again."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=5)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)
    finally:
        os.remove(temp_file_path)

# Initialize Streamlit app
st.title("Conversational AI Code Assistant")

# Initialize model and memory in session state
if "model" not in st.session_state:
    st.session_state.model = initialize_model()

if "chain" not in st.session_state:
    st.session_state.chain = initialize_chain(st.session_state.model)

# Display chat history
for msg in st.session_state.chain.memory.chat_memory.messages:
    emoji = "\U0001F464" if msg.type == "human" else "\U0001F916"
    st.chat_message("user" if msg.type == "human" else "assistant").write(f"{emoji} {msg.content}")

# User input handling
user_input = st.chat_input("\U0001F50D Ask me anything about coding...")
if user_input:
    response = st.session_state.chain.run(input=user_input)
    st.chat_message("user").write(f"\U0001F464 {user_input}")
    st.chat_message("assistant").write(f"\U0001F916 {response}")
    
    # Extract and execute Python code
    code_block = extract_code(response)
    if code_block:
        with st.expander("📋 Copy Code"):
            st.code(code_block, language="python")
        
        execution_result = execute_code(code_block)
        st.subheader("\U0001F4BB Execution Output:")
        st.code(execution_result, language="plaintext")
