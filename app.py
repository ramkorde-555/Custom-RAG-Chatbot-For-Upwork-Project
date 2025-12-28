import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate

# UI SETUP
st.set_page_config(page_title="Mistral Local RAG", layout="wide")
st.title("Local RAG ChatBot")
st.markdown("Grounding **Ministral-3B** in your proprietary documents.")

# INITIALIZE LOCAL MODELS
@st.cache_resource
def load_models():
    llm = ChatOllama(model="ministral-3:3b", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embeddings

llm, embeddings = load_models()


# File Loaders
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    # ".txt": TextLoader,
    # ".csv": CSVLoader,
}


# DOCUMENT INGESTION
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Build Knowledge Base") and uploaded_files:
        with st.spinner("Chunking & Embedding..."):
            all_docs = []
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[-1].lower()
    
                if ext in LOADER_MAPPING:
                    try:
                        temp_path = f"./temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # loader from loader map
                        loader_class = LOADER_MAPPING[ext]
                        loader = loader_class(temp_path)
                        
                        all_docs.extend(loader.load())
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
                else:
                    # Found unsupported file types
                    st.warning(f"Unsupported file type: {file.name}. Only PDF and DOCX are allowed.")

            # Split into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(all_docs)

            # Create/Update local Vector DB
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            # Store retriever in session state
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.success(f"Indexed {len(splits)} chunks successfully!")


# CHAT LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            # Define the Prompt
            qa_system_prompt = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer concise.
            
            Context: {context}"""

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                ("human", "{input}"),
            ])

            # Create the Chains
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)

            # Execute and Stream
            with st.spinner("Searching documents..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload and index your documents in the sidebar first!")