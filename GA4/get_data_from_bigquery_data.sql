-- [GA4] BigQuery Export schema
-- https://support.google.com/analytics/answer/7029846?hl=en#zippy=%2Cpublisher-early-access-only

CREATE OR REPLACE TABLE `henry-automated-bi.ga4_to_snowflake.events_flat` 
PARTITION BY event_dt
AS
WITH base AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS event_dt,
    TIMESTAMP_MICROS(event_timestamp) AS event_timestamp,
    event_name,
    user_pseudo_id,
    geo.country,
    geo.region,
    device.category AS device_category,
    traffic_source.source,
    traffic_source.medium,
    traffic_source.name AS campaign,
    (SELECT value.string_value FROM UNNEST(event_params)
     WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params)
     WHERE key = 'page_referrer') AS page_referrer,
    (SELECT value.int_value FROM UNNEST(event_params)
     WHERE key = 'ga_session_id') AS ga_session_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'  -- 3 month
),
items AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS event_dt,
    event_name,
    user_pseudo_id,
    i.item_id,
    i.item_name,
    i.item_category,
    i.price,
    i.quantity
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
  UNNEST(items) AS i
  WHERE _TABLE_SUFFIX BETWEEN '20201201' AND '20210131'
)
SELECT
  b.event_dt,
  b.event_timestamp,
  b.event_name,
  b.user_pseudo_id,
  b.country,
  b.region,
  b.device_category,
  b.source,
  b.medium,
  b.campaign,
  b.page_location,
  b.page_referrer,
  b.ga_session_id,
  i.item_id,
  i.item_name,
  i.item_category,
  i.price,
  i.quantity
FROM base b
LEFT JOIN items i
  ON i.event_dt = b.event_dt
  AND i.event_name = b.event_name
  AND i.user_pseudo_id = b.user_pseudo_id;
