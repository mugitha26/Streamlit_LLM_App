import os
import hashlib
import tempfile
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


def create_vector_store(pdf_bytes, embeddings):

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Invalid PDF file")

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    collection_name = "pdf_" + file_hash[:20]

    os.makedirs(DB_PATH, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    ) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not documents:
        raise ValueError("No readable text found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks found")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    if vector_store._collection.count() == 0:
        ids = [
            f"chunk_{i}_{file_hash[:8]}"
            for i in range(len(chunks))
        ]

        vector_store.add_documents(
            documents=chunks,
            ids=ids
        )

    return vector_store, len(documents), len(chunks), file_hash


def get_answer(question, vector_store, llm):

    retrieved_docs = vector_store.similarity_search(
        question, k=5
    )

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an educational AI assistant.
Answer using only the PDF context.
If the answer is unavailable, say:
Answer not found in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    answer = llm.invoke(final_prompt)

    return str(answer), retrieved_docs


def save_result(filename, record):

    os.makedirs(DATASET_PATH, exist_ok=True)
    file_path = os.path.join(DATASET_PATH, filename)

    df = pd.DataFrame([record])

    df.to_csv(
        file_path,
        mode="a",
        header=not os.path.exists(file_path),
        index=False
    )