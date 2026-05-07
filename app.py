import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="AI Lease Reviewer", page_icon="🧾")
st.title("🧾 AI Lease Reviewer")
st.markdown("**Upload lease PDFs and ask questions**")

# OpenAI Key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

uploaded_files = st.file_uploader("Upload one or more lease PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            with open(os.path.join(temp_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        
        with st.spinner("Processing leases..."):
            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            prompt = ChatPromptTemplate.from_template(
                "You are an expert real estate operations consultant. Answer ONLY using the provided lease documents. "
                "Cite the lease name if possible.\n\nContext: {context}\n\nQuestion: {question}"
            )
            
            chain = (
                {"context": retriever | (lambda x: "\n---\n".join(doc.page_content for doc in x)),
                 "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.success(f"✅ Loaded {len(docs)} lease documents")
            
            question = st.text_input("Ask about the leases (tenant rules, termination, rent, maintenance, etc.):")
            if st.button("Get Answer") and question:
                with st.spinner("Thinking..."):
                    answer = chain.invoke(question)
                    st.write(answer)
else:
    st.info("👆 Upload some sample leases to begin")

st.caption("Portfolio Project by Tyler — AI Real Estate Ops Consultant")