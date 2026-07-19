import streamlit as st
from dotenv import load_dotenv, dotenv_values
import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# import os
# from pathlib import Path


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        doc = pymupdf.open(stream=pdf.read(), filetype="pdf")

        for page in doc:
            text += page.get_text()
        doc.close()

    return text


def get_text_chunks(txt):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(txt)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def main():
    st.set_page_config(page_title='Chat with your Documents',
                       page_icon='📚', layout='centered')

    # Minimal custom styling
    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; max-width: 800px; }
        header, #MainMenu, footer { visibility: hidden; }
        .upload-box {
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #fafafa;
        }
        .stChatMessage { border-radius: 12px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; font-weight:600;'>📚 Chat with your Documents</h2>",
                unsafe_allow_html=True)

    # Track whether docs have been processed
    if 'docs_ready' not in st.session_state:
        st.session_state.docs_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Upload box only shows before processing
    if not st.session_state.docs_ready:
        with st.container():
            st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
            pdf = st.file_uploader(
                'Upload your PDF(s) to get started',
                accept_multiple_files=True,
                type=['pdf']
            )
            col1, col2 = st.columns([1, 4])
            with col1:
                go = st.button(
                    'Process', use_container_width=True, type='primary')
            st.markdown("</div>", unsafe_allow_html=True)

            if go and pdf:
                with st.spinner('Reading and indexing your documents...'):
                    raw_text = get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                st.session_state.docs_ready = True
                st.rerun()
            elif go and not pdf:
                st.warning('Please upload at least one PDF first.')
    else:
        # Small status line + option to add more docs
        top_col1, top_col2 = st.columns([5, 1])
        with top_col1:
            st.caption('✅ Documents indexed — ask away below.')
        with top_col2:
            if st.button('Reset', use_container_width=True):
                st.session_state.docs_ready = False
                st.session_state.chat_history = []
                st.rerun()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Chat input (disabled until docs are processed)
    question = st.chat_input(
        'Ask questions about your documents' if st.session_state.docs_ready else 'Upload a document first',
        disabled=not st.session_state.docs_ready
    )

    if question:
        st.session_state.chat_history.append(
            {'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.write(question)

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                # answer = get_answer(question, st.session_state.vectorstore)
                answer = "..."  # plug in your QA chain response here
                st.write(answer)

        st.session_state.chat_history.append(
            {'role': 'assistant', 'content': answer})


if __name__ == '__main__':
    main()
