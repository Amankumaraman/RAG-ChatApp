# Import necessary libraries
import streamlit as st
import openai
from main import get_index_for_pdf
import os
from transformers import GPT2Tokenizer
import pandas as pd

# Set the title for the Streamlit app
st.title("RAG ChatBot")

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def count_tokens(text):
    return len(tokenizer.encode(text))

# Cached function to create a vectordb for the provided PDF files
@st.cache_data
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Define the template for the chatbot prompt
prompt_template = """
You are a helpful Assistant who retrieves all related information about a specific topic from a provided document.

Provide all relevant information based on the user query from the content of the PDF extract with metadata.

Keep your answers exactly as they appear in the document without modifications or personal interpretation.

Match user queries with the exact text from the document.

Focus on the metadata, particularly 'filename' and 'page' when answering.

The evidence is solely from the content of the PDF extract.

Respond with "Not applicable" if the text is irrelevant.

The PDF content is:
{pdf_extract}

"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    # search_results
    pdf_extract = "/n ".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

    # Count tokens and store data
    data = []
    for message in prompt:
        tokens = count_tokens(message["content"])
        # Calculate cost: $0.06 per 1,000 tokens as of 2021
        cost = tokens * 0.06 / 1000
        data.append({"role": message["role"], "content": message["content"], "tokens": tokens, "cost": cost})

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Create a download button for the CSV
    st.download_button(
        label="Download token usage data",
        data=df.to_csv(index=False),
        file_name="token_usage.csv",
        mime="text/csv",
    )
