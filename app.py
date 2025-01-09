import streamlit as st
import os
import tempfile
from document_processor_enhanced import process_and_analyze_pdf, store_in_pinecone
from rag_app import query_pinecone

def main():
    st.title("Document Q&A Assistant")
    
    uploaded_file = st.file_uploader("Upload a PDF document (Max 6MB)", type=['pdf'])
    
    if uploaded_file is not None:
        if uploaded_file.size > 6 * 1024 * 1024:
            st.error("File size exceeds 6MB limit. Please upload a smaller file.")
        else:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    
                    # Process document
                    processed_elements = process_and_analyze_pdf(tmp_path)
                    if processed_elements:
                        success, message = store_in_pinecone(processed_elements)
                        if success:
                            st.success("Document processed successfully!")
                        else:
                            st.error(f"Error processing document: {message}")
            
            # Q&A Interface
            st.subheader("Ask questions about your document")
            user_question = st.text_input("Enter your question:")
            
            if user_question:
                with st.spinner('Generating response...'):
                    answer = query_pinecone(user_question)
                    st.write("Answer:", answer)

if __name__ == "__main__":
    main()