# Retail Analytics Recommender Pipeline

A scalable end‑to‑end data engineering and recommendation system pipeline powered by Airflow, BigQuery, dbt, and Python. This project simulates a real‑world e‑commerce recommender workflow with rich analytics and hybrid recommendation models.


## 🎯 Project Objective

Build a robust ETL pipeline to ingest, clean, model and analyze sales data, culminating in personalized product recommendations. Implement collaborative and hybrid filtering approaches, and surface key KPIs via dbt models.


## 📚 About Dataset

**Online Retail II (UK-based, non‑store online giftware)**  
**Period**: December 1, 2009 – December 9, 2011  
Primarily all-occasion giftware; many customers are wholesalers.

**Attributes**:

| Column       | Type     | Description                                                                                          |
|--------------|----------|------------------------------------------------------------------------------------------------------|
| InvoiceNo    | Nominal  | 6‑digit invoice number; prefix 'C' indicates cancellation                                             |
| StockCode    | Nominal  | 5‑digit product code uniquely identifying each item                                                  |
| Description  | Nominal  | Product name                                                                                         |
| Quantity     | Numeric  | Quantity of each product per transaction                                                              |
| InvoiceDate  | Datetime | Date and time the transaction was generated                                                           |
| UnitPrice    | Numeric  | Unit price in £ sterling per unit                                                                    |
| CustomerID   | Nominal  | 5‑digit customer identifier                                                                           |
| Country      | Nominal  | Customer’s country of residence                                                                       |


## 🏗️ Architectural Components

1. **ETL (Apache Airflow)**

Implemented in `airflow/dags/retail_etl.py`, this DAG runs once per day (`@daily`) and contains four Python‑based tasks:

- **DAG ID**: `retail_etl`  
- **Tasks** (in order):

  1. **extract_data**  
     - Reads the raw CSV (`data/raw/online_retail.csv`)  
     - Writes a Parquet snapshot to `data/processed/retail.parquet`

  2. **transform_data**  
     - Loads `retail.parquet`  
     - Drops rows missing `Customer ID` and removes duplicates  
     - Parses `InvoiceDate` to `datetime`  
     - Filters out cancelled orders (Invoice numbers starting with “C”)  
     - Calculates a `TotalPrice` column (`Quantity × Price`)  
     - Writes cleaned data to `data/processed/cleaned_retail.parquet`

  3. **upload_to_gcs**  
     - Uploads `cleaned_retail.parquet` to your GCS bucket (`gs://<YOUR_BUCKET>/processed/cleaned_retail.parquet`)  
     - Requires `GOOGLE_APPLICATION_CREDENTIALS` to be set

  4. **load_to_bigquery**  
     - Loads the Parquet file from GCS into BigQuery  
       - **Dataset**: `retail`  
       - **Table**: `sales`  
     - Uses `autodetect=True` for schema inference


2. **Analytics Models (dbt)**

Defined in the dbt/ project using modular, version-controlled SQL models.

📐 Key Models (in models/marts/):

   - **RFM Segmentation** (`rfm.sql`): Recency, Frequency, Monetary score per user
Customer segments were derived using an RFM scoring model based on the following dimensions:

      - Recency: Days since last purchase  
      - Frequency: Total number of transactions  
      - Monetary: Total revenue per customer  

Each customer is scored on a scale from 1 to 5 (using `NTILE(5)`) for each dimension

Customer segmentation was performed using RFM (Recency, Frequency, Monetary) scoring, and mapped into seven behavioral segments:

| Segment Name        | Criteria (R, F, M)                    | Description                                  |
|---------------------|----------------------------------------|----------------------------------------------|
|  Champions         | R = 5 AND F ≥ 4 AND M ≥ 4              | Bought recently, frequently, and spent well. |
|  Loyal Customers   | R ≥ 4 AND F ≥ 3                        | Frequent and recent buyers.                  |
|  Potential Loyalists | R ≥ 3 AND F ≥ 2 AND M ≥ 2            | Engaged, could become loyal.                 |
|  New Customers      | R = 5 AND F = 1                       | Very recent first-time buyers.               |
|  At Risk           | R ≤ 2 AND F ≥ 2 AND M ≥ 2              | Once valuable but haven’t returned.          |
|  Hibernating        | R ≤ 2 AND F ≤ 2 AND M ≤ 2              | Inactive and low engagement.                 |
|  Others             | *(All other combinations)*            | Unclassified or mixed behavior patterns.     |

This segmentation was used in both analysis and dashboard filtering to highlight user lifecycle stages and tailor product recommendations.


   - **Product Performance** (`product_performance.sql`): total revenue & order frequency per `stock_code`
   - **Country Orders** (`country.sql`): number of orders & revenue by country
   - **Product Trends** (`trends.sql`): monthly time‑series of item sales
   - **Repeat Purchase Rate** (`RPR.sql`): % of returning buyers over time
Use dbt build to create these views/tables in BigQuery.

3. **Data Warehouse (BigQuery)**
   - **Raw Layer**:
     `sales` — populated directly from the ETL DAG using Parquet → GCS → BigQuery

   - **Staging / Marts (created by dbt)**:

`stg_sales` — cleaned and typed staging layer

rfm, product_performance, RPR, country, trend. — output of dbt models

BigQuery serves as the source of truth for analytics, reporting, and recommendation input features.


4. **Recommendation Engine (Python + BigQuery)**
   - **Collaborative Filtering** via `implicit` library:
     This module uses implicit feedback (e.g., purchase quantity) from user-item interactions and trains matrix factorization models like ALS, BPR, and LogisticMF to generate personalized product recommendations. Evaluation is performed using time-based and leave-one-out strategies to assess recommendation quality via precision, recall, hit rate, and MRR.
     - **ALS**: `AlternatingLeastSquares(factors=64, regularization=0.1, iterations=30)`
     - **BPR**: `BayesianPersonalizedRanking(factors=64, learning_rate=0.01, regularization=0.01, iterations=50)`
     - **LogMF**: `LogisticMatrixFactorization(factors=64, regularization=0.01, iterations=50)`
   - **Hybrid** (LightFM):
     This approach combines collaborative filtering with content-based features using the LightFM model. It learns latent factors from user-item interactions (such as the quantity purchased) and incorporates item descriptions (represented using TF-IDF) and user features (represented using an identity matrix) to improve recommendations. The model is evaluated using K-fold cross-validation, with performance metrics such as Precision@k, Recall@k, and F1 Score. The content-based features (item descriptions) are incorporated via the item_features matrix, while user features are represented using a simple identity matrix for each user.


     ```python
     from lightfm import LightFM
     model = LightFM(
       no_components=128,
       learning_rate=0.05,
       loss='warp',
       item_alpha=0.001,
       user_alpha=0.001
     )
     ```
   - **Evaluation**: Precision@5; best achieved **0.38** with LightFM hyperparameters above.


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Apache Airflow
- Google Cloud SDK & BigQuery access
- dbt Core

## 📂 Project Structure

- `airflow/`  
  DAG definitions & configs

- `data/raw/` & `data/processed/`  
  Source and intermediate files

- `dbt/`  
  Models, snapshots, and project config

- `recommender/`  
  Recommendation engine and the recommended list


  ## 📊 Dashboards

- **Sales & Revenue Overview**
  - KPI cards (total revenue, total order, average order value, repeat purchase rate)
  - Time‑series line chart of monthly revenue trends  
  - Pie chart of sales volume by Stock code
  - Geo map showing revenue by country  

- **RFM Segmentation**
  - Bar chart of user counts per RFM segment  
  - Heatmap of Recency, Frequency, Monetary score distributions  
  - Revenue by RFM segment  

- **Top 10 Customer Insights**  
  - Table of the top 10 customers(country, RFM segment, frequency, lifetime revenue)
  - Total revenue by country
  - Comparision between average order value and frequency

- **Recommendations**  
  - Personalized product lists from hybrid filtering per user

---

## 📊 Results & KPI Highlights

- **Best Model (Precision@5)**: `0.38` — achieved with a hybrid LightFM model  
- **Average Order Value (AOV)**: `$1,879.63`  
- **Repeat Purchase Rate**: `72%` — strong indication of customer loyalty
- **Peak Sales Month**: `November` — highest sales volume observed in both 2010 and 2011
- **Top Country**: `United Kingdom` — highest number of customers  
- **RFM Segments**: `7 distinct segments` enabling targeted recommendations  
  - Largest cohort: **At Risk** — requires re-engagement strategies  
  - Other key groups: **Loyal Customers**, **Potential Loyalists**  
- **Top 10 Customers**: Identified high-value buyers driving revenue concentration  
- **Pipeline Coverage**: Full 2009–2011 dataset cleaned and loaded into BigQuery, powering analytics and model inference  








