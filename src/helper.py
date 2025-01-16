from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os



OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

# load the environment variables
from dotenv import load_dotenv
load_dotenv()


from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)