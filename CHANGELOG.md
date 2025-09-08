Changelog

v0.1 – Initial Data Loading
	•	Exported GA4 sample dataset from BigQuery into Parquet in GCS.
	•	Set up Snowflake STORAGE INTEGRATION and STAGE to pull Parquet files into RAW.EVENTS_FLAT.
	•	Learned the hard way that EXPORT DATA can’t touch GA4 meta tables, so had to flatten and export per date.
	•	Hit timestamp issues (event_timestamp looked like garbage) → fixed with TO_TIMESTAMP_NTZ(event_timestamp/1e6).

⸻

v0.2 – First Staging Layer
	•	Built stg_ga4_events.sql to standardise GA4 columns (event_ts, source, medium, price, qty, etc).
	•	Added dedup with row_number(), but at first my key was too coarse (didn’t include price/qty).
	•	Discovered MART totals didn’t line up with raw.
	•	Realised GA4 raw exports do include duplicates, which inflate revenue.

⸻

v0.3 – Dedup Refinement
	•	Refined row_key to use (event_dt, user_pseudo_id, ga_session_id, item_id, item_name, price, quantity).
	•	Reconciled against a strict dedup baseline (RAW_REV_DEDUP).
	•	Confirmed duplicate inflation was the big reason for revenue gaps.
	•	Learned that aligning MART with strict dedup gives me stable, trustworthy KPIs.

⸻

v0.4 – Clean Purchase Lines
	•	Found ~25 “purchase” rows with null price/qty.
	•	Decided to keep them visible in stg_ga4_events but filter them out in a new view: stg_ga4_purchase_lines.
	•	Adjusted facts to consume from stg_ga4_purchase_lines only.
	•	Fixed my failing singular test (no more trailing semicolon mistakes 😅).
	•	Tests now pass cleanly.

⸻

v0.5 – Fact Tables and KPIs
	•	Built fct_purchases_daily → daily revenue, orders (distinct order_key), AOV.
	•	Built fct_attribution_daily → revenue/orders by source, medium, campaign.
	•	Reconciliation test added: MART vs strict dedup.
	•	Now MART aligns with RAW_DEDUP (tiny diffs only).
	•	Locked dedup logic with dbt tests (unique row_key, recomputation test).
	•	Core KPIs (Revenue, Orders, AOV, Revenue by Channel) are now in place.

⸻

Current Status
	•	End-to-end pipeline runs from GA4 → GCS → Snowflake → dbt → clean marts.
	•	Dedup logic solid.
	•	Tests ensure no regressions.
	•	Fact models produce stable metrics ready for dashboards.

⸻

Next Steps
	•	Document dedup contract clearly in README.
	•	Add metrics YAML for semantic KPIs.
	•	Build trend signals (WoW deltas, anomalies, z-scores).
	•	Create AI insights table for dashboards.
	•	Hook up Power BI / Tableau with KPI cards + Insights panel.

⸻

Henry Chiu, Aug 2025
