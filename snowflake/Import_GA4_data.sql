-- Use a high-privilege role for one-time setup
USE ROLE ACCOUNTADMIN;

-- Compute (auto-suspends after 60s idle)
CREATE OR REPLACE WAREHOUSE WH_XS
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

USE WAREHOUSE WH_XS;
  
-- Logical containers
CREATE OR REPLACE DATABASE GA4_DEMO;
CREATE OR REPLACE SCHEMA GA4_DEMO.RAW;
CREATE OR REPLACE SCHEMA GA4_DEMO.STAGE;
CREATE OR REPLACE SCHEMA GA4_DEMO.MART;

-- Point the session to them
USE WAREHOUSE WH_XS;
USE DATABASE GA4_DEMO;
USE SCHEMA GA4_DEMO.STAGE;

-- Create a Storage Integration to your GCS bucket
CREATE OR REPLACE STORAGE INTEGRATION GCS_INT
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = GCS
  ENABLED = TRUE
  STORAGE_ALLOWED_LOCATIONS = ('gcs://henry-ga4-snowflake/ga4/')
  COMMENT = 'GCS integration for GA4 Parquet files';

  
-- get the Snowflake’s GCP service account
DESC INTEGRATION GCS_INT;


USE DATABASE GA4_DEMO;
USE SCHEMA GA4_DEMO.STAGE;

-- Create file format + stage
CREATE OR REPLACE FILE FORMAT PARQ_FMT TYPE = PARQUET;

CREATE OR REPLACE STAGE GA4_GCS_STAGE
  URL = 'gcs://henry-ga4-snowflake/ga4/'
  STORAGE_INTEGRATION = GCS_INT
  FILE_FORMAT = PARQ_FMT;

-- Test visibility
LIST @GA4_GCS_STAGE/events_flat/date=2020-12-01/;

-- create table
USE DATABASE GA4_DEMO;
USE SCHEMA GA4_DEMO.RAW;


CREATE OR REPLACE TABLE GA4_DEMO.RAW.EVENTS_FLAT (
  EVENT_DT DATE,
  EVENT_TIMESTAMP TIMESTAMP_NTZ,  -- a real timestamp here
  EVENT_NAME STRING,
  USER_PSEUDO_ID STRING,
  COUNTRY STRING,
  REGION STRING,
  DEVICE_CATEGORY STRING,
  SOURCE STRING,
  MEDIUM STRING,
  CAMPAIGN STRING,
  PAGE_LOCATION STRING,
  PAGE_REFERRER STRING,
  GA_SESSION_ID NUMBER,
  ITEM_ID STRING,
  ITEM_NAME STRING,
  ITEM_CATEGORY STRING,
  PRICE FLOAT,
  QUANTITY NUMBER
);

------------------------------------
-- LOAD DATA
------------------------------------

-- 1) Temp staging table to hold Parquet rows as VARIANT
CREATE OR REPLACE TEMP TABLE STAGE_EVENTS_VARIANT (V VARIANT);


--2 )COPY each month into the temp table
-- November 2020 (30 days)
COPY INTO STAGE_EVENTS_VARIANT
FROM @STAGE.GA4_GCS_STAGE/events_flat/
PATTERN='.*date=2020-11-(0[1-9]|[12][0-9]|30)\/.*'
FILE_FORMAT = (FORMAT_NAME = STAGE.PARQ_FMT)
ON_ERROR = 'CONTINUE';

-- December 2020
COPY INTO STAGE_EVENTS_VARIANT
FROM @STAGE.GA4_GCS_STAGE/events_flat/
PATTERN='.*date=2020-12-(0[1-9]|[12][0-9]|3[01])\/.*'
FILE_FORMAT = (FORMAT_NAME = STAGE.PARQ_FMT)
ON_ERROR = 'CONTINUE';

-- January 2021
COPY INTO STAGE_EVENTS_VARIANT
FROM @STAGE.GA4_GCS_STAGE/events_flat/
PATTERN='.*date=2021-01-(0[1-9]|[12][0-9]|3[01])\/.*'
FILE_FORMAT = (FORMAT_NAME = STAGE.PARQ_FMT)
ON_ERROR = 'CONTINUE';

--Delete data when need to reload
/*
DELETE FROM GA4_DEMO.RAW.EVENTS_FLAT
WHERE EVENT_DT BETWEEN '2020-11-01' AND '2021-01-31';
*/

--3)  Cast microseconds to Snowflake TIMESTAMP and copy to raw.events_flat
INSERT INTO GA4_DEMO.RAW.EVENTS_FLAT
(
  EVENT_DT, EVENT_TIMESTAMP, EVENT_NAME, USER_PSEUDO_ID,
  COUNTRY, REGION, DEVICE_CATEGORY, SOURCE, MEDIUM, CAMPAIGN,
  PAGE_LOCATION, PAGE_REFERRER, GA_SESSION_ID,
  ITEM_ID, ITEM_NAME, ITEM_CATEGORY, PRICE, QUANTITY
)
SELECT
  V['event_dt']::DATE,
  TO_TIMESTAMP_NTZ(V['event_timestamp']::NUMBER / 1000000) AS EVENT_TIMESTAMP,
  V['event_name']::STRING,
  V['user_pseudo_id']::STRING,
  V['country']::STRING,
  V['region']::STRING,
  V['device_category']::STRING,
  V['source']::STRING,
  V['medium']::STRING,
  V['campaign']::STRING,
  V['page_location']::STRING,
  V['page_referrer']::STRING,
  V['ga_session_id']::NUMBER,
  V['item_id']::STRING,
  V['item_name']::STRING,
  V['item_category']::STRING,
  V['price']::FLOAT,
  V['quantity']::NUMBER
FROM STAGE_EVENTS_VARIANT;


--------------------------------------------------
-- create staging folder
CREATE SCHEMA IF NOT EXISTS GA4_DEMO.STAGING;
CREATE OR REPLACE SCHEMA GA4_DEMO.DEV;

--
--
-- after that is transformation by dbt!!!
--
--
--------------------------------------------------





