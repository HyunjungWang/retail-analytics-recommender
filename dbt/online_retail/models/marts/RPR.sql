WITH customer_orders AS (
  SELECT
    customer_id,
    COUNT(DISTINCT Invoice) AS order_count
  FROM {{ ref('stg_sales') }}
  GROUP BY customer_id
)

SELECT
  COUNTIF(order_count > 1) * 1.0 / COUNT(*) AS repeat_purchase_rate
FROM customer_orders
