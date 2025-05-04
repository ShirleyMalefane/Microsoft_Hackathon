import streamlit as st
import os
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import pandas as pd
import pickle


st.set_page_config(page_title="Government Chatbot", layout="wide")
st.title("ChatPSI (Public Service Information)")


os.environ['LANGSMITH_API_KEY'] = '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'

# Load LLM model
llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="llama3.2:1b", base_url='http://127.0.0.1:11434')


data_path = "/Users/shirleymalefane/Desktop/Microsoft_Hackathon/microsoft_dataset.csv"
data = pd.read_csv(data_path)
data = data[0:70]
data['content'] = data['answer']
text_data = " ".join(data['content'].values)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text_data)


vector_store = Chroma.from_texts(chunks, embed_model)
with open('chunks.pkl', 'wb') as file:
    pickle.dump(chunks, file)


retriever = vector_store.as_retriever()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(1.5) 
            template = f"""Please answer strictly based on the provided context. Expand responses using retrieval augmentation. When responding do not say thing about context. {prompt}"""
            response = retrieval_chain.invoke({"input": template})
            answer = response['answer']

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.markdown(answer)
