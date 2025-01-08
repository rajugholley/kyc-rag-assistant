import streamlit as st
import os
import tempfile
from document_processor_enhanced import process_and_analyze_pdf, store_in_pinecone
from rag_app import query_pinecone

def main():
    st.title("KYC & Document Q&A Assistant")
    
    # Create two tabs
    tab1, tab2 = st.tabs(["KYC Guidelines Q&A", "Custom Document Q&A"])
    
    # Tab 1: Original KYC Q&A
    with tab1:
        st.header("KYC Guidelines Assistant")
        user_question = st.text_input("Ask a question about KYC regulations:", key="kyc_question")
        
        if user_question:
            with st.spinner('Generating response...'):
                answer = query_pinecone(user_question)
                st.write("Answer:", answer)
    
    # Tab 2: Custom Document Upload & Q&A
    with tab2:
        st.header("Custom Document Analysis")
        uploaded_file = st.file_uploader("Upload a PDF document (Max 2MB)", type=['pdf'])
        
        if uploaded_file is not None:
            if uploaded_file.size > 2 * 1024 * 1024:
                st.error("File size exceeds 2MB limit. Please upload a smaller file.")
                return
                
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            with st.spinner('Processing document...'):
                # Clear existing vectors (in background)
                clear_index()
                # Process new document
                success, message = process_uploaded_pdf(tmp_path)
                
            # Clean up temp file
            os.unlink(tmp_path)
            
            if success:
                st.success("Document processed successfully!")
                
                # Q&A Interface
                st.subheader("Ask questions about your document")
                user_question = st.text_input("Enter your question:", key="custom_question")
                
                if user_question:
                    with st.spinner('Generating response...'):
                        answer = query_pinecone(user_question)
                        st.write("Answer:", answer)
            else:
                st.error(f"Error processing document: {message}")

if __name__ == "__main__":
    main()