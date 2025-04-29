WITH rfm AS (
  SELECT 
    customer_id,
    DATE_DIFF(CURRENT_DATE(), DATE(MAX(InvoiceDate)), DAY) AS recency,  -- Convert MAX(InvoiceDate) to DATE
    COUNT(DISTINCT Invoice) AS frequency,
    SUM(TotalPrice) AS monetary
  FROM {{ ref('stg_sales') }}  -- Reference your staging table
  GROUP BY customer_id
)

SELECT * FROM rfm
