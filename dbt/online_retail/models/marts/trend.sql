SELECT
  EXTRACT(MONTH FROM `InvoiceDate`) AS month,
  SUM(`TotalPrice`) AS monthly_sales
  FROM {{ ref('stg_sales') }}  -- Reference your staging table
GROUP BY month
ORDER BY month
