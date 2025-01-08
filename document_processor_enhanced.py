from pdf2image import convert_from_path
import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def process_and_analyze_pdf(pdf_path):
    try:
        processed_elements = []
        
        # Use both pdfplumber and PyMuPDF for better coverage
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text content
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Meaningful text
                    processed_elements.append(
                        Document(
                            page_content=text,
                            metadata={
                                'type': 'content',
                                'page': page_num,
                                'source': pdf_path
                            }
                        )
                    )
        
        return processed_elements

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
def store_in_pinecone(processed_elements):
    try:
        print("\nStoring elements in Pinecone...")
        embeddings = OpenAIEmbeddings()
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index('kyc-index')

        # Clear existing vectors
        index.delete(delete_all=True)

        # Store new vectors
        for i, element in enumerate(processed_elements):
            embedding = embeddings.embed_query(element.page_content)
            index.upsert(vectors=[{
                'id': f'elem_{i}',
                'values': embedding,
                'metadata': {
                    'text': element.page_content,
                    'type': element.metadata['type'],
                    'page': element.metadata['page']
                }
            }])

        print(f"Successfully stored {len(processed_elements)} elements in Pinecone")
        return True, "Document processed successfully"

    except Exception as e:
        print(f"Error storing in Pinecone: {e}")
        return False, str(e)