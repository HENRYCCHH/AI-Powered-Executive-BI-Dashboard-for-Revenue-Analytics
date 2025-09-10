# AI‑Powered Revenue Analytics Dashboard (Snowflake · dbt · Gemini · Power BI)

<img src="assets/demo.gif" alt="Dashboard demo" width="800">

A portfolio project that turns raw GA4 events into an executive dashboard with **daily AI insights**.
Data lands in **Snowflake**, is transformed with **dbt**, summarized by **Google Gemini** (free tier), and visualized in **Power BI**.

---

## Highlights
- **Automated pipeline**: ELT → dbt models → AI writer → Power BI refresh.
- **Four insight modes** per date: **Total**, **Month** (rolling 30d), **Week** (rolling 7d), **Day**.
- **Compact “Next steps”** recommendations sized to fit a PBI text card.
- **Channel movers**: ranks sources/mediums/campaigns with an impact score (size × rate of change).
- **Least‑privilege** Snowflake roles: writer can only `MERGE` the AI table; viewer is read‑only.

---

## Tech stack
- **Data Source**: Google Analytics 4
- **Warehouse**: GCS BigQuery & S3 Bucket, Snowflake
- **Modeling**: dbt Cloud
- **LLMs**: Google Gemini (Flash)
- **Viz**: Power BI Desktop / Service
- **Python**: `snowflake-connector-python`, `google-genai`, `python-dotenv`

---

## Repo layout
```
/assets/                 # images & animated demo GIF
/GA4/                    # Source SQL used to pull sample GA4 data into Snowflake 
/snowflake/              # DDL/DML for schemas, tables, views, and roles (incl. AI_INSIGHTS + grants)
/dbt/                    # dbt project (models, seeds, macros)
/powerbi/                # .pbix report
ai_insight_backfill.py   # one-time historical load (writes 4 insights per day)
ai_insight_writer.py     # daily writer (keeps latest 4 insights fresh)
.env.example             # sample environment file
README.md
```

---

## Getting started

### 1) Python env
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: pip install google-genai snowflake-connector-python python-dotenv
```

### 2) Configure environment
Copy `.env.example` → `.env`, then set your values:
```dotenv
# Snowflake
SF_ACCOUNT=your_account_locator
SF_USER=AI_WRITER
SF_PASSWORD=********
SF_ROLE=AI_APP_ROLE
SF_WAREHOUSE=WH_XS
SF_DATABASE=GA4_DEMO
SF_SCHEMA=DEV_MART
SF_DIM_DB=GA4_DEMO
SF_DIM_SCHEMA=DEV
SF_DIM_TABLE=DIM_DATE
SF_AI_TABLE=AI_INSIGHTS

# Gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash   # or gemini-2.5-flash-lite

# Optional backfill window
# BACKFILL_START=2020-11-01
# BACKFILL_END=2021-01-31
```

### 3) Historical backfill (one‑time)
```bash
python ai_insight_backfill.py
```
Writes **4 rows per day** into `GA4_DEMO.DEV_MART.AI_INSIGHTS` with a compact `display_text`.

### 4) Daily writer (keeps latest fresh)
```bash
python ai_insight_writer.py
```
Generates insights for the **latest date in DIM_DATE** across Total/Month/Week/Day and `MERGE`s them into the AI table.

---

## Power BI
1. **Get Data → Snowflake**, import `AI_INSIGHTS` (plus your fact views).
2. Add a **date slicer** and a **compare‑mode slicer** (`Total/Month/Week/Day`).
3. Bind your text card to the `DISPLAY_TEXT` column (or the measure that selects the latest row per date/type).
4. Optional titles:
   - *AI Insight — {Selected Mode} · ending {Selected Date}*
   - *Daily Revenue Trend — {Start} → {End}*

> Note: Publish to the Service (Pro/Fabric or Trial) to enable **scheduled refresh**.

---

## Notes
- **Free Gemini quota** varies by model; Flash‑Lite allows more RPM/RPD if you need faster backfills.
- The **Month** period is **rolling 30 days** by default (change to calendar MTD if preferred).
- `display_text` is sized for a small PBI text card (≤ ~6 lines). Adjust in `render_display_text()`.

---

## License
MIT — use freely; attribution appreciated.
