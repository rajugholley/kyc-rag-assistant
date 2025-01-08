import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

def load_and_process_document():
    try:
        # Load the PDF document
        loader = PyPDFLoader(r"C:\Users\Personal\Gen AI\Gen AI Hands-on practise\kyc_rag_assistant\data\Regulatory-Notice-17-40.pdf")
        documents = loader.load()
        print(f"Successfully loaded document with {len(documents)} pages")
        return documents
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

def split_documents(documents):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return None

def create_embeddings_and_store(chunks):
    try:
        embeddings = OpenAIEmbeddings()
        print("Creating embeddings for chunks...")
        
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index('kyc-index')
        
        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk.page_content)
            index.upsert(vectors=[{
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk.page_content}
            }])
        print(f"Successfully stored {len(chunks)} embeddings in Pinecone")
    except Exception as e:
        print(f"Error creating/storing embeddings: {e}")

if __name__ == "__main__":
    # Run the entire process
    documents = load_and_process_document()
    if documents:
        chunks = split_documents(documents)
        if chunks:
            create_embeddings_and_store(chunks)