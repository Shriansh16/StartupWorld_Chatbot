from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import streamlit as st
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
load_dotenv()
pkey=st.secrets["PINECONE_API_KEY"]
#pkey=""

import pickle

def download_embeddings():
    embedding_path = "local_embeddings"

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    return embedding
def find_match(input):
    pc = PineconeClient(api_key=pkey)
    model=download_embeddings()
    index_name='startupworld-chatbot'
    index=pc.Index(index_name)
    vectorstore = Pinecone(
    index, model.embed_query,"text"
                       )
    result=vectorstore.similarity_search(
    input,  # our search query
    k=5  # return 5 most relevant docs
      )
    return result
def query_refiner(conversation, query):
  #  if not conversation or not query:
   #     return query
    api_key1 = st.secrets["GROQ_API_KEY"]
    #api_key1 =""
    client = Groq(api_key=api_key1)
    response = client.chat.completions.create(
    model="gemma-7b-it",
     messages=[{"role": "system", "content": "If the user's query is unrelated to the conversation context, return it as is. Otherwise, refine the query in under 20 words."},
              {"role": "user", "content": f"Given the conversation log:\n{conversation}\n\nand the query:\n{query}\n\nDetermine if the query is relevant. If yes, refine it; if not, return it as is. Provide only the refined question, without any additional text."}
    ],
    temperature=0.5,
    max_tokens=256,
    top_p=1,
    stream=False,
    stop=None,
     )
    return response.choices[0].message.content
def get_conversation_string():
    conversation_string = ""
    start_index = max(len(st.session_state['responses']) - 2, 0)
    for i in range(start_index, len(st.session_state['responses']) - 1):        
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string

def load_pdf(pdf_path):
    loader=DirectoryLoader(pdf_path,glob='*.pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    return document