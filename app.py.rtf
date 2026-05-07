{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
from langchain_community.document_loaders import PyPDFDirectoryLoader\
from langchain_text_splitters import RecursiveCharacterTextSplitter\
from langchain_openai import OpenAIEmbeddings, ChatOpenAI\
from langchain_community.vectorstores import Chroma\
from langchain_core.prompts import ChatPromptTemplate\
from langchain_core.runnables import RunnablePassthrough\
from langchain_core.output_parsers import StrOutputParser\
import os\
import tempfile\
import shutil\
\
st.set_page_config(page_title="AI Lease Reviewer", page_icon="
\f1 \uc0\u55358 \u56830 
\f0 ")\
st.title("
\f1 \uc0\u55358 \u56830 
\f0  AI Lease Reviewer")\
st.write("Upload multiple lease agreements and ask questions about them.")\
\
# Get OpenAI key from Streamlit secrets\
if "OPENAI_API_KEY" not in st.secrets:\
    st.error("OpenAI API key not found. Add it in Streamlit secrets.")\
    st.stop()\
\
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]\
\
# Upload PDFs\
uploaded_files = st.file_uploader("Upload Lease PDFs", type="pdf", accept_multiple_files=True)\
\
if uploaded_files:\
    # Create temporary folder for PDFs\
    with tempfile.TemporaryDirectory() as temp_dir:\
        for uploaded_file in uploaded_files:\
            file_path = os.path.join(temp_dir, uploaded_file.name)\
            with open(file_path, "wb") as f:\
                f.write(uploaded_file.getbuffer())\
        \
        # Process the documents\
        with st.spinner("Processing leases..."):\
            loader = PyPDFDirectoryLoader(temp_dir)\
            docs = loader.load()\
            \
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
            chunks = splitter.split_documents(docs)\
            \
            embeddings = OpenAIEmbeddings()\
            vectorstore = Chroma.from_documents(chunks, embeddings)\
            \
            retriever = vectorstore.as_retriever(search_kwargs=\{"k": 4\})\
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\
            \
            prompt = ChatPromptTemplate.from_template(\
                "You are an expert real estate operations consultant. Answer ONLY using the lease documents. "\
                "Be precise and cite which lease if possible.\\n\\nContext: \{context\}\\n\\nQuestion: \{question\}"\
            )\
            \
            chain = (\
                \{"context": retriever | (lambda docs: "\\n---\\n".join(doc.page_content for doc in docs)),\
                 "question": RunnablePassthrough()\}\
                | prompt\
                | llm\
                | StrOutputParser()\
            )\
            \
            st.success(f"
\f1 \uc0\u9989 
\f0  Processed \{len(docs)\} lease documents!")\
            \
            # Chat interface\
            question = st.text_input("Ask a question about the leases (e.g., tenant responsibilities, early termination, rent rules):")\
            if st.button("Get Answer") and question:\
                with st.spinner("Thinking..."):\
                    answer = chain.invoke(question)\
                    st.write(answer)\
else:\
    st.info("Upload some sample lease PDFs to get started.")\
\
st.caption("Built as a portfolio project by Tyler \'97 AI Real Estate Ops Consultant")}