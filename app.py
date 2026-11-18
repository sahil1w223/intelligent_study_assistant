
import streamlit as st

from src.document_loader import file_loder
from src.ingestion import break_into_chunks
from src.vector_store import vector_store
from src.rag_chain import get_answer


st.set_page_config(
    page_title="Intelligent Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Study Notes & Exam Assistant")
st.write("Upload your lecture PDF and ask questions about your notes.")

uploaded_file = st.file_uploader(
    "Upload your PDF notes",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Reading PDF and building the search index..."):
            try:
                with open("uploaded_notes.pdf", "wb") as file:
                    file.write(uploaded_file.getvalue())

                documents = file_loder("uploaded_notes.pdf")
                chunks = break_into_chunks(documents)
                store = vector_store(chunks)

                st.session_state["store"] = store
                st.session_state["chunks"] = chunks
                st.session_state["filename"] = uploaded_file.name

                st.success(
                    f"PDF processed! Pages: {len(documents)} | "
                    f"Chunks: {len(chunks)}"
                )

            except Exception as e:
                st.error(f"Could not process PDF: {e}")

if "store" in st.session_state:
    st.subheader("Ask your notes")

    question = st.text_input(
        "Enter your question",
        placeholder="What does ReLU do?"
    )

    if st.button("Get Answer") and question.strip():
        with st.spinner("Finding information and generating your answer..."):
            try:
                result = get_answer(
                    st.session_state["store"],
                    st.session_state["chunks"],
                    question
                )

                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Sources")
                for doc in result["sources"]:
                    st.write(f"**PDF page:** {doc['Page_Number']}")
                    st.write(doc["Text"])

            except Exception as e:
                st.error(f"Could not generate answer: {e}")
