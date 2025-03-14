import os
import warnings
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
    memory = ConversationBufferWindowMemory(k=15)
    
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=(
            "You are an advanced AI assistant skilled in helping users write efficient, "
            "well-structured, and professional code. You understand various frameworks, "
            "libraries, and best practices. Below is the conversation history:\n"
            "{history}\n"
            "User's request: {input}\n"
            "Provide a clear, optimized, and professional response including necessary "
            "frameworks, libraries, and best practices. If applicable, give examples."
        )
    )

    chain = LLMChain(
        llm=model,
        prompt=prompt_template,
        memory=memory
    )
    return chain

def chat_loop(chain):
    """Runs an interactive chat loop with LLMChain memory."""
    while True:
        message = input("Enter a message for LLM (type 'bye' to exit): ")
        if message.lower() == "bye":
            print("Goodbye!")
            break
        response = chain.run(input=message)
        print("LLM:", response)

# Example usage
model = initialize_model()
chain = initialize_chain(model)
chat_loop(chain)
