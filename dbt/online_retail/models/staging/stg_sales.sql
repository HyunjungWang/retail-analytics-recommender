
SELECT 
  CAST(`Customer ID` AS INT64) AS customer_id,
  Invoice,
  InvoiceDate,
  StockCode,
  Quantity,
  Price,
  Country,
  TotalPrice
FROM {{ source('retail', 'sales') }}
