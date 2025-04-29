from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime
import pandas as pd
from google.cloud import bigquery
import subprocess
import os
from google.cloud import storage
import sys


# Update this path to match your local file structure
RAW_DATA_PATH = "retail_etl/data/raw/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Functions for each ETL step
def extract_data():
    DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "online_retail.csv")

    #df = pd.read_csv(os.path.join(RAW_DATA_PATH, "online_retail.csv"))
    df=pd.read_csv(DATA_PATH)
    output_path = os.path.join(BASE_DIR, "..", "..", "data", "processed", "retail.parquet")

    df.to_parquet(output_path)


def transform_data():
    DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "retail.parquet")

    df = pd.read_parquet(DATA_PATH)

    # 1. Remove rows with missing CustomerID
    df.dropna(subset=["Customer ID"], inplace=True)

# 2. Remove duplicates
    df.drop_duplicates(inplace=True)

# 3. Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print(df["InvoiceDate"].dtype)
    print(df["Price"].dtype)
    print(df["Price"].head(3))
    
# 4. remove canceled orders   
    df = df[~df["Invoice"].astype(str).str.startswith("C")]


# 5. Create TotalPrice = Quantity * UnitPrice
    df["TotalPrice"] = df["Quantity"] * df["Price"]

# Optional: Save the cleaned data
    output_path = os.path.join(BASE_DIR, "..", "..", "data", "processed", "cleaned_retail.parquet")

    df.to_parquet(output_path, index=False, coerce_timestamps="us")
    
   

def upload_to_gcs():
    local_file = os.path.join(BASE_DIR, "..", "..", "data", "processed", "cleaned_retail.parquet")

    bucket_name = "my-retail-bucket"
    blob_name = "processed/cleaned_retail.parquet"

    # Initialize a GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Upload the file
    blob.upload_from_filename(local_file)

    print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}")


def load_to_bigquery():
    client = bigquery.Client()
    bucket = "my-retail-bucket"
    blob_name = "processed/cleaned_retail.parquet"
    dataset_id = "retail"
    table_id = "sales"
    uri = f"gs://{bucket}/{blob_name}"
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        autodetect=True
    )

    load_job = client.load_table_from_uri(uri, full_table_id, job_config=job_config)
    load_job.result()
    print(f"Loaded {full_table_id} successfully.")
# Define the DAG



with DAG(
    dag_id="retail_etl",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["retail", "etl", "recommender"]
) as dag:

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )
    upload_gcs=PythonOperator(
        task_id ="upload_to_gcs",
        python_callable=upload_to_gcs,
    )

    load_bigquery = PythonOperator(
        task_id="load_to_bigquery",
        python_callable=load_to_bigquery,
    )



  
    extract >> transform >>upload_gcs>> load_bigquery

