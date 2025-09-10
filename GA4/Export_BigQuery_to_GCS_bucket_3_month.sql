DECLARE project_id STRING DEFAULT 'henry-automated-bi';
DECLARE bucket     STRING DEFAULT 'henry-ga4-snowflake';

DECLARE d DATE     DEFAULT DATE '2020-11-01';
DECLARE d_end DATE DEFAULT DATE '2021-01-31';

WHILE d <= d_end DO
  EXECUTE IMMEDIATE FORMAT("""
    EXPORT DATA OPTIONS (
      uri = 'gs://%s/ga4/events_flat/date=%s/part-*.parquet',
      format = 'PARQUET',
      overwrite = true
    ) AS
    SELECT *
    FROM `%s.ga4_to_snowflake.events_flat`
    WHERE event_dt = DATE '%s'
  """,
    bucket,
    FORMAT_DATE('%Y-%m-%d', d),   -- cast DATE -> STRING
    project_id,
    FORMAT_DATE('%Y-%m-%d', d)    -- cast DATE -> STRING
  );

  SET d = DATE_ADD(d, INTERVAL 1 DAY);
END WHILE;