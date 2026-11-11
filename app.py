import os
import time
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from pdf_qa import (
    create_vector_store,
    get_answer,
    save_result
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
UPLOAD_PATH = os.path.join(DATASET_PATH, "uploaded_files")
ASSETS_PATH = os.path.join(BASE_DIR, "assets")

SAMPLE_PATH = os.path.join(DATASET_PATH, "sample.pdf")
LOGO_PATH = os.path.join(ASSETS_PATH, "logo.png")

os.makedirs(UPLOAD_PATH, exist_ok=True)

st.set_page_config(
    page_title="Education AI Assistant",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3.2:3b")


# Custom title and sidebar
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=130)
st.sidebar.write("Logo Path:", LOGO_PATH)
st.sidebar.write("Logo Exists:", os.path.exists(LOGO_PATH))

st.sidebar.title("Education AI Assistant")
st.sidebar.write("Ask questions from PDF documents.")
st.sidebar.divider()

st.title("AI PDF Question Answering System")
st.write("Upload PDF documents and ask questions using AI.")

# Multiple PDF upload
uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        if not pdf_bytes.startswith(b"%PDF"):
            st.error("Invalid PDF file")
            continue

        save_path = os.path.join(
            UPLOAD_PATH,
            os.path.basename(uploaded_file.name)
        )

        with open(save_path, "wb") as file:
            file.write(pdf_bytes)

# Get PDF files
pdf_names = []

if os.path.exists(SAMPLE_PATH):
    pdf_names.append("sample.pdf")

for filename in os.listdir(UPLOAD_PATH):
    if filename.lower().endswith(".pdf"):
        pdf_names.append(filename)

pdf_names = sorted(set(pdf_names))

if not pdf_names:
    st.info("Please upload a PDF document.")
    st.stop()

selected_pdf = st.selectbox(
    "Select PDF Document",
    pdf_names
)

if selected_pdf == "sample.pdf":
    with open(SAMPLE_PATH, "rb") as file:
        pdf_bytes = file.read()
else:
    selected_path = os.path.join(UPLOAD_PATH, selected_pdf)
    with open(selected_path, "rb") as file:
        pdf_bytes = file.read()

# Load PDF and models
try:
    with st.spinner("Loading PDF and AI models..."):
        embeddings = load_embeddings()
        llm = load_llm()

        vector_store, page_count, chunk_count, file_hash = (
            create_vector_store(pdf_bytes, embeddings)
        )

    st.success("PDF loaded successfully!")

except Exception as error:
    st.error(f"Error: {error}")
    st.stop()

# Display filename and page count
st.subheader("Document Information")
col1, col2 = st.columns(2)

col1.metric("PDF Pages", page_count)
col2.metric("Text Chunks", chunk_count)

st.write("Uploaded PDF:", selected_pdf)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if file_hash not in st.session_state.chat_history:
    st.session_state.chat_history[file_hash] = []

history = st.session_state.chat_history[file_hash]

# Clear Chat History
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history[file_hash] = []
    st.rerun()

st.sidebar.success(f"Selected PDF: {selected_pdf}")

# Ask question
st.subheader("Ask Your Question")

question = st.chat_input("Type your question here...")

if question:

    history.append({
        "role": "user",
        "content": question
    })

    start_time = time.time()

    try:
        with st.spinner("Generating answer..."):
            answer, retrieved_docs = get_answer(
                question,
                vector_store,
                llm
            )

        response_time = round(time.time() - start_time, 2)

        sources = []

        for doc in retrieved_docs:
            sources.append({
                "page": doc.metadata.get("page", 0) + 1,
                "content": doc.page_content
            })

        history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "response_time": response_time
        })

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        save_result("queries.csv", {
            "timestamp": timestamp,
            "document": selected_pdf,
            "question": question
        })

        save_result("answers.csv", {
            "timestamp": timestamp,
            "document": selected_pdf,
            "question": question,
            "answer": answer,
            "response_time_seconds": response_time
        })

    except Exception as error:
        history.append({
            "role": "assistant",
            "content": f"Error: {error}",
            "sources": [],
            "response_time": 0
        })

# Display conversation history
st.divider()
st.subheader("Conversation History")

for message in history:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":

            st.caption(
                f"Response Time: "
                f"{message.get('response_time', 0)} seconds"
            )

            sources = message.get("sources", [])

            if sources:
                with st.expander("View Retrieved PDF Sections"):

                    for index, source in enumerate(sources, 1):
                        st.markdown(
                            f"**Chunk {index} - Page {source['page']}**"
                        )
                        st.write(source["content"])
                        st.divider()

st.sidebar.caption("Powered by Streamlit and Ollama")