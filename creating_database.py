import os
import time
#from pdf_loader import *
from utils import *
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from pinecone import ServerlessSpec
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4


load_dotenv()
pc = PineconeClient(
    api_key=os.environ.get("PINECONE_API_KEY")
)
extracted_data=load_pdf("data/")
embeddings=download_embeddings()


spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)
index_name = 'startupworld-chatbot'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]
# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of ada 002
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

tokenizer = tiktoken.get_encoding('cl100k_base')
# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=20,
    length_function=tiktoken_len
)

from tqdm.auto import tqdm
from uuid import uuid4

batch_limit = 100
texts = []
metadatas = []

for i, record in enumerate(tqdm(extracted_data)):
    # first get metadata fields for this record
    metadata = {
        'source': str(record.metadata.get('source')),
        'page': record.metadata.get('page')

    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record.page_content)
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embeddings.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))
