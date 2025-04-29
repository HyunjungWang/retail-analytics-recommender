SELECT
  `StockCode`,
  SUM(`Quantity`) AS total_quantity_sold,
  SUM(`TotalPrice`) AS total_revenue
  FROM {{ ref('stg_sales') }}  -- Reference your staging table
GROUP BY `StockCode`
ORDER BY total_revenue DESC
