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

v0.6 – Date Spine & Trend Signals
• Created dense dim_date and a zero-filled purchase view to include empty days.
• Added WoW deltas, 7-day moving averages, z-score anomaly flags.
• Fixed partial date coverage (Nov no-sales span) so trends compute correctly.

⸻

v0.7 – Channel Trends & Top Movers
• Built v_channel_daily, v_channel_trends, and a “Top Movers” query with an improved impact score (size + rate).
• Power BI star schema: DIM_DATE + DIM_CHANNEL to Facts and Views (composite key on Source/Medium/Campaign).
• DAX patterns for: anchor last date, rolling windows (Day/Week/Month/Total), dynamic labels, YoY/QoQ/MoM/WoW.

⸻

v0.8 – AI Insights (LLM) – Storage & Serving
• Created GA4_DEMO.DEV_MART.AI_INSIGHTS table (headline, bullets, actions, JSON inputs, model metadata).
• Wrote ai_insight_writer.py (daily) and ai_insight_backfill.py (one-time historical) using Gemini 2.5 Flash.
• Enforced JSON schema in prompts; compact display_text renderer (headline + 2 bullets + “Next steps”).
• Rate-limit + retry logic; safe defaults when revenue/orders are zero (no token spend).

⸻

v0.9 – Security, Roles & Connectivity
• Provisioned AI_APP_ROLE (runtime) and BI_VIEWER_ROLE (read-only); created AI_WRITER and POWERBI_SVC users.
• Granted USAGE on DB/Schema/Warehouse; SELECT/INSERT/UPDATE/DELETE on AI_INSIGHTS.
• Power BI connection via SSO + ODBC DSN; MFA-friendly setup validated.

⸻

v1.0 – Executive Dashboard (Power BI)
• KPI cards + dynamic titles; date anchor slicer + compare-mode slicer (Day/Week/Month/Total).
• Line trends with anchor marker; Top Movers table ranked by impact (robust RANKX over SUMMARIZE with TREATAS).
• AI Insight card bound to AI_INSIGHTS.display_text and filtered by date/type for instant narrative.
• Visual polish: readable labels, compact legends, clean titles, short source labels (“Private” for redacted).

⸻

v1.1 – Backfill & Data Quality Enhancements
• Backfilled 2020-11-01 → 2021-01-31 with four insights per day (Total/Month/Week/Day).
• Skipped LLM calls for 2020-11 (no purchases) using default insights.
• Patched individual holes (e.g., missing 2020-12-01 Total insight).
• Eliminated QUALIFY misuse in Python queries; swapped to supported filters when needed.

⸻

v1.2 – DevEx, Docs & Sharing
• Added .env.example, Readme, repo structure, and runbook notes.
• GIF demo + LinkedIn copy; clarified free-tier LLM limits and model choices.
• Compact text utility (ai_display_compact.py) merged into writer/backfill for consistent output.

⸻

Henry Chiu, Sep 2025