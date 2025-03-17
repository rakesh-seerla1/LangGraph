import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
import warnings
from langchain.chat_models import init_chat_model

# Hardcoded API Keys (as per your request)
GROQ_API_KEY = "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"
HUGGINGFACE_API_KEY = "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"

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to initialize vectorstore
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Function to generate a human-like prompt
def generate_prompt():
    return PromptTemplate(
        template="""
        You are a highly intelligent and friendly AI assistant. You are designed to:
        ✅ Help users analyze and extract information from PDFs.
        ✅ Engage in human-like, warm, and engaging conversations.
        ✅ Maintain memory of the conversation to make responses feel natural.
        
        🔥 How to Respond:
        1️⃣ When PDFs are uploaded:  
           - Always prioritize PDF content first when answering.  
           - Provide clear, structured, and concise responses.  
           - Use bullet points, numbered lists, and summaries where needed.  
           - If the PDF doesn't have enough details, say so and use external knowledge.  
           
        2️⃣ When No PDFs are uploaded:  
           - Act as a smart, engaging chatbot that remembers context.  
           - Respond in a human-like, fun, and friendly way with some expressions & emojis 😊.  
           - Keep answers concise yet informative and avoid robotic tone.  
           
        3️⃣ For All Responses:  
           - Use simple, easy-to-understand language—avoid complex jargon.  
           - Always keep context in mind from the chat history to avoid repeating info.  
           - If asked about personal opinions, reply with an engaging response.  

        **Chat History:** {chat_history}  
        **Context (from PDFs or External Knowledge):** {context}  
        **User's Question:** {question}  
        """,
        input_variables=["chat_history", "context", "question"]
    )

# Function to initialize model
def initialize_model(model_name="gemma2-9b-it"):
    """Initializes and returns a chat model with API key from environment variables."""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    warnings.filterwarnings('ignore')
    return init_chat_model(model_name, model_provider="groq")

LLm = initialize_model()

# Function to set up the conversational chain
def get_conversation_chain(vectorstore=None):
    llm = LLm

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=15,  # Keeps the last 15 messages
        return_messages=True
    )

    prompt_template = generate_prompt()

    # Load QA Chain with error handling for missing input keys
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt_template,
        document_variable_name="context"
    )

    # Question rephrasing chain
    question_generator = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template="Rephrase the following question: {question}", input_variables=["question"])
    )

    if vectorstore:  # Use retriever if PDFs exist
        return ConversationalRetrievalChain(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=qa_chain,
            memory=memory,
            question_generator=question_generator
        )
    else:
        return LLMChain(  # Acts as a chatbot when no PDFs are available
            llm=llm,
            prompt=PromptTemplate(
                template="""
                You are a friendly AI assistant. Keep responses engaging and structured.
                
                **Chat History:** {chat_history}  
                **User's Question:** {question}  
                """,
                input_variables=["chat_history", "question"]
            )
        )

# Function to handle user input
def handle_userinput(user_question):
    if not user_question or user_question.strip() == "":
        st.warning("⚠ Please enter a valid question.")
        return

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.info("🤖 No PDFs uploaded. Switching to chatbot mode!")
        st.session_state.conversation = get_conversation_chain(None)

    # Ensure chat history is initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Add user question to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Prepare input for conversation chain
    conversation_input = {
        "chat_history": st.session_state.chat_history,
        "question": user_question
    }

    # Call the conversation chain
    response = st.session_state.conversation(conversation_input)

    # Add AI response to the chat history
    if isinstance(response, dict) and "answer" in response:
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    elif isinstance(response, str):
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    elif hasattr(response, "content"):
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})

    # Display chat history
    for message in st.session_state.chat_history[-15:]:  # Show the last 15 messages
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.set_page_config(page_title="Multi PDF Chat + Conversational AI", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Ensure session state variables are initialized
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("📖 Chat with AI about your PDFs & More!")
    user_question = st.chat_input("Ask me anything...")

    with st.sidebar:
        st.subheader("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if pdf_docs and st.button("🔄 Process PDFs"):
            with st.spinner("Processing... ⏳"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("✅ PDFs processed! Ask me anything.")

    if user_question:
        handle_userinput(user_question)

if __name__ == "__main__":
    main()
