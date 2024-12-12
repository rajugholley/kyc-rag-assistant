import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

try:
    # Initialize Pinecone with new syntax
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
    )
    
    # List all indexes
    indexes = pc.list_indexes()
    print("Successfully connected to Pinecone!")
    print("Active indexes:", indexes.names())
    
    # Try to connect to our specific index
    index = pc.Index('kyc-index')
    print("\nSuccessfully connected to 'kyc-index'!")
    print("Index description:", index.describe_index_stats())
    
except Exception as e:
    print("Error:", e)