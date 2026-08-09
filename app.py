import os
import tempfile
import streamlit as st
import chromadb

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Simple PDF RAG Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG using ChromaDB + Groq")

st.write("Upload a PDF and ask questions from it.")

# ==========================================================
# GROQ CLIENT (LLM INTEGRATION)
# ==========================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ==========================================================
# CHROMADB
# ==========================================================

@st.cache_resource
def load_chromadb():

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        "pdf_collection"
    )

    return collection

collection = load_chromadb()

# ==========================================================
# PDF UPLOAD
# ==========================================================

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ==========================================================
# EXTRACT TEXT
# ==========================================================

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# ==========================================================
# SPLIT TEXT INTO CHUNKS
# ==========================================================

def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i+chunk_size])

    return chunks

# ==========================================================
# STORE EMBEDDINGS
# ==========================================================

def create_vector_database(chunks):

    try:
        collection.delete(ids=collection.get()["ids"])
    except:
        pass

    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(

        ids=[str(i) for i in range(len(chunks))],

        documents=chunks,

        embeddings=embeddings

    )

# ==========================================================
# RETRIEVE CONTEXT
# ==========================================================

def retrieve_context(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=3

    )

    return results["documents"][0]

# ==========================================================
# LLM RESPONSE
# ==========================================================

def ask_llm(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not available in the context,
reply with:

"I couldn't find the answer in the uploaded PDF."

------------------------

Context:

{context}

------------------------

Question:

{question}

Answer:
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.3,

        messages=[

            {
                "role":"system",
                "content":"Answer only using the provided context."
            },

            {
                "role":"user",
                "content":prompt
            }

        ]

    )

    return response.choices[0].message.content

# ==========================================================
# BUILD VECTOR DATABASE
# ==========================================================

if uploaded_pdf:

    if "database_ready" not in st.session_state:

        with st.spinner("Reading PDF..."):

            text = extract_text(uploaded_pdf)

        with st.spinner("Splitting Text..."):

            chunks = split_text(text)

        with st.spinner("Creating Embeddings..."):

            create_vector_database(chunks)

        st.session_state.database_ready = True

        st.success("PDF Indexed Successfully!")

# ==========================================================
# QUESTION
# ==========================================================

question = st.text_input(
    "Ask your Question"
)

# ==========================================================
# GENERATE ANSWER
# ==========================================================

if st.button("Generate Answer"):

    if uploaded_pdf is None:

        st.warning("Please upload a PDF.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching Vector Database..."):

            context = retrieve_context(question)

        st.subheader("Retrieved Chunks")

        for i, chunk in enumerate(context, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(chunk)

        with st.spinner("Sending Context to Groq LLM..."):

            answer = ask_llm(
                question,
                context
            )

        st.subheader("LLM Answer")

        st.success(answer)