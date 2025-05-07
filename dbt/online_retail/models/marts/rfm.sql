WITH max_date AS (
  SELECT MAX(DATE(InvoiceDate)) AS latest_date
  FROM {{ ref('stg_sales') }}
),
rfm AS (
  SELECT 
    customer_id,
    DATE_DIFF((SELECT latest_date FROM max_date), DATE(MAX(InvoiceDate)), DAY) AS recency,
    COUNT(DISTINCT Invoice) AS frequency,
    SUM(TotalPrice) AS monetary
  FROM {{ ref('stg_sales') }}
  GROUP BY customer_id
)
,

scored_rfm AS (
  SELECT *,
    NTILE(5) OVER (ORDER BY recency DESC) AS recency_score,  -- Lower recency = higher score
    NTILE(5) OVER (ORDER BY frequency ASC) AS frequency_score,
    NTILE(5) OVER (ORDER BY monetary ASC) AS monetary_score
  FROM rfm
)

SELECT *,
       CONCAT(recency_score, frequency_score, monetary_score) AS rfm_segment
FROM scored_rfm
