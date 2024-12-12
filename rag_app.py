import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

def query_pinecone(query):
    try:
        # Initialize OpenAI
        embeddings = OpenAIEmbeddings()
        llm = OpenAI(temperature=0)
        
        # Create query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index('kyc-index')
        
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Prepare context
        context = ""
        for match in results['matches']:
            context += match['metadata']['text'] + "\n\n"
            
        # Generate answer
        prompt = f"""Based on the following context, answer the question. Only use information from the context to answer.
        
        Context: {context}
        
        Question: {query}"""
        
        response = llm.invoke(prompt)
        return response
    
    except Exception as e:
        print(f"Error querying Pinecone:", e)
        return str(e)