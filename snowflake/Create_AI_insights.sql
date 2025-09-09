CREATE TABLE IF NOT EXISTS GA4_DEMO.DEV_MART.AI_INSIGHTS (
  insight_date       DATE,
  headline           STRING,
  bullets            VARIANT,      -- JSON array
  actions            VARIANT,      -- JSON array
  top_movers_used    VARIANT,      -- JSON array (the input you sent to the LLM)
  kpi_used           VARIANT,      -- JSON object (the input you sent)
  model_name         STRING,
  prompt_tokens      NUMBER,
  created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
  raw_prompt         STRING,
  raw_response       VARIANT
);

ALTER TABLE GA4_DEMO.DEV_MART.AI_INSIGHTS ADD COLUMN IF NOT EXISTS  DISPLAY_TEXT STRING;
ALTER TABLE GA4_DEMO.DEV_MART.AI_INSIGHTS  ADD COLUMN IF NOT EXISTS WINDOW_START DATE;
ALTER TABLE GA4_DEMO.DEV_MART.AI_INSIGHTS  ADD COLUMN IF NOT EXISTS WINDOW_END DATE;
ALTER TABLE GA4_DEMO.DEV_MART.AI_INSIGHTS ADD COLUMN IF NOT EXISTS INSIGHT_TYPE STRING;

Select * from GA4_DEMO.DEV_MART.AI_INSIGHTS where INSIGHT_DATE = '2021-01-31';