import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

def clear_index():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index('kyc-index')
        index.delete(delete_all=True)
        print("Cleared all vectors from index")
    except Exception as e:
        print(f"Error clearing index: {e}")

def process_uploaded_pdf(file_path):
    try:
        # Load and process the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store
        embeddings = OpenAIEmbeddings()
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index('kyc-index')
        
        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk.page_content)
            index.upsert(vectors=[{
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk.page_content}
            }])
        
        return True, f"Successfully processed {len(chunks)} chunks"
    except Exception as e:
        return False, str(e)