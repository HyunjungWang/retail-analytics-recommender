SELECT
  FORMAT_DATE('%Y-%m', DATE(InvoiceDate)) AS year_month,
  SUM(TotalPrice) AS monthly_sales
FROM {{ ref('stg_sales') }}
GROUP BY year_month
ORDER BY year_month
