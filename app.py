import streamlit as st
from rag_app import query_pinecone

def main():
    st.title("KYC Compliance Assistant")
    
    user_question = st.text_input("Ask a question about KYC regulations:", "")
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Generating response..."):
                answer = query_pinecone(user_question)
                st.write("Answer:", answer)
        else:
            st.warning("Please enter a question!")

if __name__ == "__main__":
    main()