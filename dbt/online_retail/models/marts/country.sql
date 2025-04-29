SELECT
  customer_id,
  Country,
  COUNT(DISTINCT `Invoice`) AS frequency,
  SUM(`TotalPrice`) AS total_spent
  FROM {{ ref('stg_sales') }}  -- Reference your staging table
GROUP BY customer_id, Country
