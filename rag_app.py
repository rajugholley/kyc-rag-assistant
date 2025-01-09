from langchain_openai import OpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone

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
            top_k=5,
            include_metadata=True
        )
        
        # Build context string
        context = ""
        for match in results['matches']:
            context += match['metadata']['text'] + "\n\n"
        
        # Combine prompts and get response
        system_prompt = """You are a knowledgeable assistant specialized in analyzing documents and providing accurate, well-structured responses. Your expertise lies
        in analyzing financial statements, regulatory documents, whitepapers from leading consulting firms etc and provding precise answers. The documents you analyze may contain different types of content including text, images and tables."""
        
        user_prompt = f"""Please analyze the following context and answer the question.
        Here are some additional instructions for you
        
        1. Your answers must be based strictly on the provided context
        2. If the answer requires combining information from different parts, explain how they relate
        3. If referencing a table, clearly indicate the table information
        4. If referencing an image, clearly indicate that image information
        5. Your answer must be clear and concise yet comprehensive
        6. Your answer must be well-organized with proper formatting when needed
        7. You must include relevant quotes, sections from the document when appropriate
        8. If user has asked you to summarize the document, then do summarize the main points without missing out the most important information
        
        If the answer isn't contained within the context, say "I cannot find information about this in the document.My abilities are limited to only analyzing what is contained in this document"
        If you find relevant information, structure your response clearly and cite specific parts of the document.
        
        Context: {context}
        
        Question: {query}
        """
        
        response = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        return response
        
    except Exception as e:
        print(f"Error querying Pinecone:", e)
        return str(e)