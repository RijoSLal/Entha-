import streamlit as st
import time
from io import BytesIO
import pdfplumber
from graph_rag import GV_RAG

# initialize RAG class
gv_rag = GV_RAG()

# inject CSS to hide the original space limit and simple tweaks
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: system-ui !important;
    }
    small.st-emotion-cache-c8ta4l {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Entha?¿")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# track PDF processing state
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False 
    

# file uploader
uploaded_file = st.file_uploader("Limit 20MB per file • PDF", type=["pdf"])
if uploaded_file is not None:
    if "last_uploaded_filename" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_filename:
        # reset document state if it's a new file
        st.session_state.document_processed = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    # process if not already done
    if not st.session_state.document_processed:
        file_bytes = BytesIO(uploaded_file.read())
        with st.spinner("Analyzing document..."):
            with pdfplumber.open(file_bytes) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            gv_rag.document_preprocessing_and_insertion(all_text)
        st.session_state.document_processed = True
        st.session_state.messages = []

   
  
    with st.chat_message("assistant"):
        st.write("What do you want to know?")
    

# show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
prompt = st.chat_input("Ask anything")

# stream RAG response generator
def response_generator(query):
    response = gv_rag.LLM(query)
    for word in response.split():
        yield word + " "
        time.sleep(0.02)

# response logic
if prompt:
    if not st.session_state.document_processed:
        st.toast("Please upload document before asking questions")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})





