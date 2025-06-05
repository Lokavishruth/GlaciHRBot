import os
import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from pathlib import Path
# ----------- CONFIGURATION -----------


BASE_DIR = Path.cwd()  # Gets current working directory (where you run the app)
PDF_DIR = str(BASE_DIR / "data")
PERSIST_DIR = str(BASE_DIR / "chroma_db")

OLLAMA_MODEL_EMBED = "mxbai-embed-large:latest"
OLLAMA_MODEL_LLM = "llama3.2:3b"

# ----------- UTILITIES -----------
def load_pdf_text(pdf_path: Path) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def chunk_text(text: str, chunk_size=200, chunk_overlap=25):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def initialize_vector_db():
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.is_file() and f.suffix.lower() == ".pdf"]
    all_docs = []

    for pdf_file in pdf_files:
        text = load_pdf_text(pdf_file)
        chunks = chunk_text(text)
        docs = [Document(page_content=chunk, metadata={"source": str(pdf_file.name)}) for chunk in chunks]
        all_docs.extend(docs)

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_EMBED)
    if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
        vectordb = Chroma.from_documents(all_docs, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()
    else:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    return vectordb

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="HR Bot", layout="wide")
st.title("❄️ GlaciHR – Your Smart HR Assistant")
st.subheader("Ask me anything related to Leave, Appraisal, or Travel & Reimbursement policies")

# Session state initialization
if "db" not in st.session_state:
    st.session_state.db = initialize_vector_db()

if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(model=OLLAMA_MODEL_LLM)

if "encoder" not in st.session_state:
    st.session_state.encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New chat input
query = st.chat_input("Ask about HR policies...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    vectordb = st.session_state.db
    llm = st.session_state.llm
    encoder = st.session_state.encoder

    # Search top-5
    docs = vectordb.similarity_search(query, k=5)
    if not docs:
        bot_response = "No relevant documents found. Please rephrase your question."
    else:
        cross_inputs = [[query, doc.page_content] for doc in docs]
        scores = encoder.predict(cross_inputs)

        docs_scores = list(zip(docs, scores))
        docs_scores.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, score in docs_scores[:2]]
        top_docs_content = [doc.page_content for doc in top_docs]
        context_str = "\n\n".join(top_docs_content)

        # Prompt to LLM
        template = """Role: You are a smart HR Bot that answers users' queries as per company policy.

Answer the question based only on the following context:
{context}

Question: {question}

##### Important: If the question and context look irrelevant, ask the user to re-ask the question. #####
"""
        prompt = ChatPromptTemplate.from_template(template).invoke({
            "context": context_str,
            "question": query
        })

        response = llm.invoke(prompt)
        bot_response = response

    # Save assistant message
    st.session_state.messages.append({"role": "ai", "content": bot_response})

    with st.chat_message("ai"):
        st.markdown(bot_response)

        if docs:
            with st.expander("📄 Top-5 Retrieved Chunks with Scores"):
                for i, (doc, score) in enumerate(docs_scores):
                    st.markdown(f"**Chunk #{i+1} (Score: {score:.4f}) — Source: {doc.metadata.get('source')}**")
                    st.code(doc.page_content[:700], language="text")

            with st.expander("📂 Sources Used in Final Answer (Top 2):"):
                for i, doc in enumerate(top_docs):
                    st.markdown(f"- **Source {i+1}:** {doc.metadata.get('source')}")
