
import os
import re
import pandas as pd
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from google.cloud import bigquery
from langchain.prompts import PromptTemplate
from google.api_core.retry import Retry
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import BigQueryLoader

# Load environment variables and configure GoogleAI
load_dotenv()
REGION = os.getenv('REGION')
PROJECT_ID = os.getenv('PROJECT_ID')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Define helper functions
def parse_bigquery_schema(documents):
    result = []
    for doc in documents:
        page_content = doc.page_content
        ddl_start = page_content.find("ddl: CREATE TABLE")
        ddl_end = page_content.find("\\n)\\nOPTIONS")
        ddl_content = page_content[ddl_start:ddl_end]
        table_name = re.search(r'`([^`]+)`', ddl_content).group(1)
        table_info = f"Table Name: `{table_name}`\n"
        columns_str = ddl_content.split("(")[1]
        columns = re.findall(r'(\w+ \w+),?', columns_str)
        column_info = "\n".join(["   - " + col.replace("\\n", "").strip() for col in columns])
        result.append(table_info + column_info)
    return "\n\n".join(result)


def schema_query(dataset):
    query = f"""
    SELECT table_name, ddl
    FROM `{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE'
    ORDER BY table_name;
    """
    return query


def parse_bigquery_schema_to_dict(documents):
    schema_dict = {}
    for doc in documents:
        page_content = doc.page_content
        ddl_start = page_content.find("ddl: CREATE TABLE")
        ddl_end = page_content.find("\\n)\\nOPTIONS")
        ddl_content = page_content[ddl_start:ddl_end]
        table_name = re.search(r'`([^`]+)`', ddl_content).group(1)
        columns_str = ddl_content.split("(")[1]
        columns = re.findall(r'(\w+ \w+),?', columns_str)
        schema_dict[table_name] = {col.split()[0]: col.split()[1] for col in columns}
    return schema_dict


# Streamlit setup
st.set_page_config(
    page_title="BigQueryAI",
    page_icon=":hotel:",
    layout="wide",
    initial_sidebar_state="auto"
)

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = {}

st.title("Google AI for BigQuery")
st.sidebar.title("Configuration")
dataset = st.sidebar.selectbox('Dataset', (f'{PROJECT_ID}.thelook','bigquery-public-data.github_repos'))

template = """SYSTEM: You are a bigquery specialist helping users by suggesting a GoogleSQL query that will help them answer their question againsts the provided context. Do not add ```googlesql``` around your answer, reply with the executable code only\n
=============
Question: \n\n{question}
=============
context: \n\n{schema}
"""

describe_template = """SYSTEM: you are a data scientist, describe this and interpret the query intent in relation to the bigquery schema answering the question: what is this information good for?, very briefly evaluate the IA generated sql in terms of optimization, stricly limit your response to information in this context\n
=============
bigquery schema: \n\n{bqschema}
=============
user query: \n\n{user_query}
=============
generated googlesql: \n\n{googlesql}
=============
data response: \n\n{response}
"""

# Initialize BigQuery client and data loader
client = bigquery.Client()
loader = BigQueryLoader(
    query=schema_query(dataset),
    metadata_columns="ddl",
    page_content_columns="ddl"
)

# Load data and parse schema
data = loader.load()
st.session_state['dataframe'] = parse_bigquery_schema(data)

# Select LLM model
model_name = st.sidebar.selectbox("LLM Model Name", ("gemini-pro","codechat-bison"))

# Set model parameters
max_output_tokens = st.sidebar.number_input("Max Output Tokens", min_value=1, value=2048)
temperature = st.sidebar.slider("Temperature (Randomness)", 0.0, 1.0, 0.0)
top_p = st.sidebar.slider("Top P (Determinism)", min_value=0, max_value=1,value=1)
top_k = st.sidebar.number_input("Top K (Vocabulary probability)", min_value=0, max_value=1,value=1)
verbose = st.sidebar.checkbox("Verbose", value=True)

# Initialize LLM instance
if model_name =="gemini-pro":
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        convert_system_message_to_human=True,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=verbose,
    )
else:
    llm = ChatVertexAI(
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=verbose,
    )

# Initialize output parser
output_parser = StrOutputParser()

# Define prompt templates
prompt = PromptTemplate.from_template(template)
chain = prompt | llm | output_parser

describe_prompt = PromptTemplate.from_template(describe_template)
describe_chain = describe_prompt | llm | output_parser

# Initialize UI elements
with st.container():
    user_query = st.text_input("Enter your query:")
    col1, col2 = st.columns(2)

    with col2:
        data_placeholder = col2.expander("Data", expanded=True).empty()
        schema_placeholder = col2.expander("BQ Schema", expanded=False).empty()
        schema_placeholder.write(parse_bigquery_schema_to_dict(data))

# Generate query, execute it, and display results
if col1.button("Generate Query"):
    bqschema = parse_bigquery_schema(data)
    googlesql = (
        chain.invoke(
            {
                "question": user_query,
                "schema": bqschema
            }
        )
    ).replace("```", "").replace("```", "")
    col1.code(googlesql, language="sql", line_numbers=True)
    try:
        custom_retry = Retry(initial=1.0, maximum=10.0, multiplier=2, deadline=300.0)
        bqdata = client.query(googlesql).result(timeout=300, retry=custom_retry).to_dataframe()
        data_placeholder.write(bqdata)
        if verbose :
            col1.write(describe_chain.invoke({"bqschema": bqschema,"user_query": user_query,"googlesql": googlesql,"response": bqdata }))
    except Exception as e:
        st.error(f"Error executing query: {e}")
