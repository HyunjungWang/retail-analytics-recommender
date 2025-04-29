SELECT
  customer_id,
  SUM(`TotalPrice`) AS lifetime_value
FROM {{ ref('stg_sales') }}  -- Reference your staging table
GROUP BY customer_id
