import streamlit as st
from dotenv import load_dotenv
import pymupdf
from langchain_text_splitters import CharacterTextSplitter


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


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with your documents', page_icon='📚')

    st.header('Chat with your Documents (PDF)')
    st.text_input('Ask questions about your documents')
    with st.sidebar:
        st.subheader('Your Documents')
        pdf = st.file_uploader(
            'Upload your document and press GO', accept_multiple_files=True)
        if st.button('GO'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf)
                # st.write(raw_text)

                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)


if __name__ == '__main__':
    main()
