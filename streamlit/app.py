# app.py
import streamlit as st
from main import initialize_components, get_schemas, generate_query
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)



# Streamlit app
st.title("MongoDB Query Generator")

# Schema selection


# Initialize components
@st.cache_resource
def get_components():
  return initialize_components()

query_gen, db_client = get_components()

# Enter schema
selected_schema = st.text_area(
    "Enter your Schema:",
    value=str(config['default_settings']['default_schema']),
    height=100
)

# Question input
question = st.text_area(
    "Enter your question:",
    value=config['default_settings']['default_question'],
    height=100
)

# RAG or Non-RAG
use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True)



# Generate button
if st.button("Generate MongoDB Query"):
    with st.spinner("Generating query..."):
        result = generate_query(query_gen, db_client, selected_schema, question, rag=use_rag)

    res, nearest, re_ranked = result

    # Display the generated query.
    if "<query>" in res.text and "</query>" in res.text:
        query = res.text.split("<query>")[1].split("</query>")[0]
        st.subheader("Query:")
        st.code(query, language="bash")
        # st.write(query)

    # LLMs Entire response.
    st.subheader("Generated MongoDB Query:")
    st.code(res.text, language="javascript")

    st.subheader("Retrieved Queries:")
    st.code(nearest, language="javascript")

    st.subheader("Re-ranked Queries:")
    st.code(re_ranked, language="javascript")

# Footer
st.sidebar.markdown("Chirayu Tripathi")
st.sidebar.markdown("GitHub: [Chirayu-Tripathi](https://github.com/Chirayu-Tripathi)")