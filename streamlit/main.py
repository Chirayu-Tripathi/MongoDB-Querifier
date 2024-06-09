import json
import os
import pandas as pd
import google.generativeai as genai
import ast
from sentence_transformers import SentenceTransformer
from pre_process import DataProcessor
from weavite_vector_db import WeaviateClient
from query_generation import QueryGeneration


# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)


# os.environ['GOOGLE_API_KEY'] = config['api_keys']['gemini_api_key']
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


def get_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def initialize_components():
    # Initialize DataProcessor
    processed_df = pd.read_csv('pre-processed.csv')
    processed_df['vector'] = processed_df['vector'].apply(lambda x: ast.literal_eval(x))

    # Initialize WeaviateClient and create class
    db_client = WeaviateClient()
    class_name = config['weaviate']['class_name']
    properties = config['weaviate']['properties']
    db_client.create_class(class_name, properties)
    db_client.add_data_object(class_name, processed_df)

    # Initialize QueryGeneration
    query_gen = QueryGeneration(get_model())

    return query_gen, db_client

def get_schemas():
    accounts = '''{"collections": [{"name": "accounts","indexes": [{"key": {"_id": 1}},{"key": {"account_id": 1}}],"uniqueIndexes": [],"document": {"properties": {"_id": {"bsonType": "objectId"},"account_id": {"bsonType": "int"},"limit": {"bsonType": "int"},"products": {"bsonType": "array","items": {"bsonType": "string"}}}}}],"version": 1}'''
    trips = '''{"collections": [{"name": "trips","indexes": [{"key": {"_id": 1}}],"uniqueIndexes": [],"document": {"properties": {"_id": {"bsonType": "objectId"},"tripduration": {"bsonType": "int"},"start_station_id": {"bsonType": "int"},"start_station_name": {"bsonType": "string"},"end_station_id": {"bsonType": "int"},"end_station_name": {"bsonType": "string"},"bikeid": {"bsonType": "int"},"usertype": {"bsonType": "string"},"birth_year": {"bsonType": "int"},"gender": {"bsonType": "int"},"start_station_location": {"bsonType": "object","properties": {"type": {"bsonType": "string","enum": ["Point"]},"coordinates": {"bsonType": "array","items": {"bsonType": "double"}}}},"end_station_location": {"bsonType": "object","properties": {"type": {"bsonType": "string","enum": ["Point"]},"coordinates": {"bsonType": "array","items": {"bsonType": "double"}}}},"start_time": {"bsonType": "date"},"stop_time": {"bsonType": "date"}}}}],"version": 1}'''
    posts_schema = '''{"collections": [{"name": "posts","indexes": [{"key": {"_id": 1}},{"key": {"permalink": 1}},{"key": {"author": 1}},{"key": {"title": 1}},{"key": {"tags": 1}},{"key": {"comments.date": 1}}],"uniqueIndexes": [],"document": {"properties": {"_id": {"bsonType": "string"},"body": {"bsonType": "string"},"permalink": {"bsonType": "string"},"author": {"bsonType": "string"},"title": {"bsonType": "string"},"tags": {"bsonType": "array","items": {"bsonType": "string"}},"comments": {"bsonType": "array","items": {"bsonType": "object","properties": {"body": {"bsonType": "string"},"email": {"bsonType": "string"},"author": {"bsonType": "string"},"date": {"bsonType": "date"}}}}}}}],"version": 1}'''
    inspections_schema = '''{"collections": [{"name": "inspections","indexes": [{"key": {"_id": 1}},{"key": {"id": 1}},{"key": {"certificate_number": 1}},{"key": {"date": 1}},{"key": {"result": 1}},{"key": {"sector": 1}},{"key": {"address.city": 1}},{"key": {"address.zip": 1}}],"uniqueIndexes": [],"document": {"properties": {"_id": {"bsonType": "string"},"id": {"bsonType": "string"},"certificate_number": {"bsonType": "int"},"business_name": {"bsonType": "string"},"date": {"bsonType": "string"},"result": {"bsonType": "string"},"sector": {"bsonType": "string"},"address": {"bsonType": "object","properties": {"city": {"bsonType": "string"},"zip": {"bsonType": "int"},"street": {"bsonType": "string"},"number": {"bsonType": "int"}}}}}}],"version": 1}'''

    return {'accounts': accounts, 'trips': trips, 'inspections': inspections_schema, 'posts': posts_schema}

def generate_query(query_gen, db_client, schema, question, rag=True):
    if rag:
      prompt = config['query_generation']['prompt_rag']
    else:
      prompt = config['query_generation']['prompt_nonrag']
    class_name = config['weaviate']['class_name']
    schemas = get_schemas()
    result = query_gen.generate_query(class_name, schema, question, db_client, prompt, rag=rag)
    return result

if __name__ == "__main__":
    main()