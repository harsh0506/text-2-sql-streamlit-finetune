import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import csv
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
model = model.to(device)
model.eval()

def generate_sql(input_prompt):
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_sql

def create_spark_session():
    return SparkSession.builder.appName("CSV to DataFrame").getOrCreate()

def create_table_query(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
    fields = [StructField(field, StringType(), True) for field in header]
    schema = StructType(fields)
    return schema

def run_sql_query(spark, csv_file, schema, table_name, query):
    try:
        df = spark.read.csv(csv_file, header=True, schema=schema)
        df.createOrReplaceTempView(table_name)
        result = spark.sql(query).collect()
        return result
    except Exception as e:
        return str(e)

def convert_structtype_to_create_table(schema, table_name):
    columns = [f"{field.name} {field.dataType.typeName()}" for field in schema.fields]
    create_table_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
    return create_table_query

# Streamlit app
st.title("SQL Query Generator")

# Option to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        csv_file = temp_file.name

    df = pd.read_csv(csv_file)
    # Display column names and data types
    st.write("### Columns and Data Types")
    st.write(df.dtypes)

    # Display statistical reports
    st.write("### Statistical Reports")
    st.write(df.describe(include='all'))

    # Basic Plots
    st.write("### Box Plots")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[column])
            st.pyplot(plt.gcf())
        
    # Use the CSV file for further processing
    table_name = st.text_input("Enter the table name:", value="users")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input("Ask your question here:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate SQL based on the question
        schema = create_table_query(csv_file)
        sql_create_table_query = convert_structtype_to_create_table(schema, table_name)
        input = "tables:\n" + sql_create_table_query + "\n" + "query for:" + question
        generated_sql = generate_sql(input)

        # Display generated SQL in chat message container
        response = f"Generated SQL: {generated_sql}"
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Run SQL query and display result in chat message container
        spark = create_spark_session()
        result = run_sql_query(spark, csv_file, schema, table_name, generated_sql)
        result_response = f"Query Result: {result}"
        with st.chat_message("assistant"):
            st.markdown(result_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result_response})
else:
    st.write("Please upload a CSV file.")
