from src.helper import load_pdf_file, text_split, OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')





extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbot-embeddings"


pc.create_index(
    name=index_name,
    dimension=1536, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
