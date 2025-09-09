import os, json, datetime as dt
from dotenv import load_dotenv
import snowflake.connector
from google import genai
from google.genai import types
from datetime import date, timedelta
import datetime as dt


load_dotenv()
# -------------------------------------------------------------------
# ENV / Locations
# -------------------------------------------------------------------
AI_DB        = os.environ["SF_DATABASE"]           # e.g. GA4_DEMO
AI_SCHEMA    = os.environ["SF_SCHEMA"]             # e.g. DEV_MART
AI_TABLE     = os.getenv("SF_AI_TABLE", "AI_INSIGHTS")
AI_TBL_FQN   = f"{AI_DB}.{AI_SCHEMA}.{AI_TABLE}"

# Facts live with the mart (adjust if different)
FACT_DB      = AI_DB
FACT_SCHEMA  = AI_SCHEMA
FCT_PURCH    = f"{FACT_DB}.{FACT_SCHEMA}.FCT_PURCHASES_DAILY"
FCT_ATTRIB   = f"{FACT_DB}.{FACT_SCHEMA}.FCT_ATTRIBUTION_DAILY"

# DIM_DATE may be in a different schema; override via env if needed
DIM_DB       = os.getenv("SF_DIM_DB", AI_DB)
DIM_SCHEMA   = os.getenv("SF_DIM_SCHEMA", AI_SCHEMA)
DIM_TABLE    = os.getenv("SF_DIM_TABLE", "DIM_DATE")
DIM_FQN      = f"{DIM_DB}.{DIM_SCHEMA}.{DIM_TABLE}"


# --- 0) Clients ---------------------------------------------------------------
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def connect_snowflake():
    """
    Here use password auth (simple). If you want key-pair, see commented section below.
    """
    return snowflake.connector.connect(
        user=os.environ["SF_USER"],
        password=os.environ.get("SF_PASSWORD"),
        account=os.environ["SF_ACCOUNT"],
        role=os.environ["SF_ROLE"],
        warehouse=os.environ["SF_WAREHOUSE"],
        database=AI_DB,
        schema=AI_SCHEMA,
    )

# For key-pair auth instead of password:
# import pathlib, base64, cryptography.hazmat.primitives.serialization as ser
# private_key_pem = pathlib.Path("sf_pk8.pem").read_bytes()
# private_key = ser.load_pem_private_key(private_key_pem, password=b"your_passphrase")
# pk_bytes = private_key.private_bytes(ser.Encoding.DER, ser.PrivateFormat.PKCS8, ser.NoEncryption())
# return snowflake.connector.connect(..., private_key=pk_bytes, ...)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_columns(cur):
    cur.execute(f"ALTER TABLE {AI_TBL_FQN} ADD COLUMN IF NOT EXISTS INSIGHT_TYPE STRING")
    cur.execute(f"ALTER TABLE {AI_TBL_FQN} ADD COLUMN IF NOT EXISTS DISPLAY_TEXT STRING")

def period_window(anchor: date, period: str, min_date: date) -> tuple[date, date]:
    """
    Inclusive [start, end] window:
      - Day   : [anchor, anchor]
      - Week  : rolling last 7 days = [anchor-6, anchor]
      - Month : rolling last 30 days = [anchor-29, anchor]
      - Total : [min_date, anchor]
    Change 'Month' to calendar MTD by replacing with: start = anchor.replace(day=1)
    """
    p = period.lower()
    if p == "day":
        return anchor, anchor
    if p == "week":
        return anchor - timedelta(days=6), anchor
    if p == "month":
        return anchor - timedelta(days=29), anchor
    if p == "total":
        return min_date, anchor
    raise ValueError(f"Unknown period '{period}'")

def previous_window(start: date, end: date) -> tuple[date, date]:
    """Window immediately before, same length."""
    days = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start, prev_end

# --- Compact display text (Power BI-friendly) -------------------------------
def render_display_text(insight: dict,
                        max_lines: int = 6,
                        max_len: int = 120) -> str:
    """
    Headline (<=90) + 2 bullets + 'Next steps' + 2 actions.
    Always shows a Next steps section (falls back to defaults if empty).
    """
    default_actions = [
        "Review channel changes and data freshness.",
        "Validate tagging and paid spend pacing."
    ]

    head = (insight.get("headline") or "").strip()[:90]

    # Clean + trim
    bullets = [str(b).strip() for b in (insight.get("bullets") or []) if str(b).strip()]
    actions = [str(a).strip() for a in (insight.get("actions") or []) if str(a).strip()]

    # Keep exactly two bullets / two actions for a tight box
    bullets = [b[:max_len] for b in bullets][:2]
    actions = [a[:max_len] for a in actions][:2]

    # Ensure we always have actions for "Next steps"
    if not actions:
        actions = default_actions[:2]

    lines = []
    if head:
        lines.append(head)
    for b in bullets:
        lines.append(f"• {b}")
    lines.append("")              # spacer
    lines.append("Next steps")
    for a in actions:
        lines.append(f"– {a}")

    return "\n".join(lines[:max_lines])


# -------------------------------------------------------------------
# Data fetchers (windowed)
# -------------------------------------------------------------------
def kpi_for_window(cur, start: date, end: date) -> dict:
    # Current window
    cur.execute(f"""
        SELECT
          COALESCE(SUM(revenue),0) AS rev,
          COALESCE(SUM(orders),0)  AS ord
        FROM {FCT_PURCH}
        WHERE date BETWEEN %s AND %s
    """, (start, end))
    rev, ords = cur.fetchone()
    aov = (rev / ords) if ords else 0

    # Previous window
    prev_start, prev_end = previous_window(start, end)
    cur.execute(f"""
        SELECT
          COALESCE(SUM(revenue),0) AS rev,
          COALESCE(SUM(orders),0)  AS ord
        FROM {FCT_PURCH}
        WHERE date BETWEEN %s AND %s
    """, (prev_start, prev_end))
    prev_rev, prev_ord = cur.fetchone()
    prev_aov = (prev_rev / prev_ord) if prev_ord else 0

    return {
        "window_start": start.isoformat(),
        "window_end":   end.isoformat(),
        "revenue":      float(rev or 0),
        "orders":       int(ords or 0),
        "aov":          float(aov or 0),
        "prev_revenue": float(prev_rev or 0),
        "prev_orders":  int(prev_ord or 0),
        "prev_aov":     float(prev_aov or 0),
        "delta_abs":    float((rev or 0) - (prev_rev or 0)),
        "delta_pct":    ( (rev - prev_rev) / prev_rev ) if prev_rev else None
    }

def movers_for_window(cur, start: date, end: date):
    # Build movers by comparing current and previous windows per channel
    prev_start, prev_end = previous_window(start, end)
    cur.execute(f"""
        WITH cur AS (
          SELECT source, medium, campaign, SUM(revenue) AS rev_cur
          FROM {FCT_ATTRIB}
          WHERE date BETWEEN %s AND %s
          GROUP BY 1,2,3
        ),
        prev AS (
          SELECT source, medium, campaign, SUM(revenue) AS rev_prev
          FROM {FCT_ATTRIB}
          WHERE date BETWEEN %s AND %s
          GROUP BY 1,2,3
        ),
        j AS (
          SELECT
            COALESCE(c.source,p.source)   AS source,
            COALESCE(c.medium,p.medium)   AS medium,
            COALESCE(c.campaign,p.campaign) AS campaign,
            COALESCE(c.rev_cur,0)  AS rev_cur,
            COALESCE(p.rev_prev,0) AS rev_prev
          FROM cur c
          FULL OUTER JOIN prev p
          USING (source, medium, campaign)
        )
        SELECT
          source, medium, campaign,
          rev_cur,
          rev_prev,
          (rev_cur - rev_prev) AS delta_abs,
          CASE WHEN rev_prev=0 THEN NULL ELSE (rev_cur - rev_prev)/rev_prev END AS delta_pct,
          -- improved impact score (size * log scaling + rate)
          (ABS(rev_cur - rev_prev) * 0.6)
          + (COALESCE(ABS((rev_cur - rev_prev)/NULLIF(rev_prev,0)),0) * 0.4) AS impact_score
        FROM j
        QUALIFY ROW_NUMBER() OVER (ORDER BY impact_score DESC NULLS LAST) <= 8
    """, (start, end, prev_start, prev_end))

    rows = cur.fetchall()
    movers = []
    for s, m, c, cur_rev, prev_rev, d_abs, d_pct, score in rows:
        movers.append({
            "source": s or "(none)",
            "medium": m or "(none)",
            "campaign": c or "(none)",
            "revenue_cur": float(cur_rev or 0),
            "revenue_prev": float(prev_rev or 0),
            "delta_abs": float(d_abs or 0) if d_abs is not None else None,
            "delta_pct": float(d_pct) if d_pct is not None else None,
            "impact_score": float(score or 0),
        })
    return movers



# -------------------------------------------------------------------
# LLM call
# -------------------------------------------------------------------
# --- Gemini call that returns strict, short JSON -----------------------------
def call_gemini(kpi: dict, movers: list[dict]) -> dict:
    from google.genai import types

    system_rules = (
        "You are a data analyst. Be concise and actionable.\n"
        "Return STRICT JSON per the schema. No extra text."
    )
    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "bullets":  {"type": "array", "items": {"type": "string"}},
            "actions":  {"type": "array", "items": {"type": "string"}}
        },
        "required": ["headline", "bullets", "actions"]
    }

    payload = {"kpi": kpi, "top_movers": movers}
    prompt = f"""{system_rules}

Constraints:
- Headline ≤ 90 chars.
- Exactly 2 bullets (≤110 chars each).
- Exactly 2 actions (≤110 chars each).
- Business tone. Avoid jargon.

Schema:
{json.dumps(schema, indent=2)}

Data:
{json.dumps(payload, separators=(',', ':'))}

Return only valid JSON.
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # Parse and harden
    txt = resp.text or ""
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = {"headline": "AI insight", "bullets": [txt.strip()[:110]], "actions": []}

    # Normalize counts and lengths; renderer will still enforce defaults if empty
    data["headline"] = str(data.get("headline", ""))[:90]
    data["bullets"]  = [str(b)[:110] for b in (data.get("bullets") or [])][:2]
    data["actions"]  = [str(a)[:110] for a in (data.get("actions") or [])][:2]
    return data


def init_session(cur):
    cur.execute(f"USE ROLE {os.environ['SF_ROLE']}")
    cur.execute(f"USE WAREHOUSE {os.environ['SF_WAREHOUSE']}")
    cur.execute(f"USE DATABASE {AI_DB}")
    cur.execute(f"USE SCHEMA {AI_SCHEMA}")



# -------------------------------------------------------------------
# Upsert (MERGE) one row per (insight_date, insight_type)
# -------------------------------------------------------------------
# --- Upsert (MERGE) by (insight_date, insight_type) -------------------------
def upsert_ai_insight(
    cur,
    insight_date,
    insight_type: str,           # e.g., "Day", "Week", "Month", or "Total"
    insight: dict,
    kpi: dict,
    movers: list,
    model_name: str,
    prompt_tokens: int | None = None,
    raw_prompt: str | None = None,
    raw_response: dict | list | str | None = None,
):
    bullets_json = json.dumps(insight.get("bullets", []), ensure_ascii=False)
    actions_json = json.dumps(insight.get("actions", []), ensure_ascii=False)
    movers_json  = json.dumps(movers or [], ensure_ascii=False)
    kpi_json     = json.dumps(kpi or {}, ensure_ascii=False)

    raw_resp_json = json.dumps(raw_response, ensure_ascii=False) if isinstance(raw_response, (dict, list)) \
                    else (raw_response if isinstance(raw_response, str) and raw_response.strip() else "{}")

    display_text = render_display_text(insight)

    sql = f"""
MERGE INTO {AI_TBL_FQN} t
USING (
  SELECT
    %s::DATE   AS insight_date,
    %s::STRING AS insight_type,
    %s::STRING AS headline,
    PARSE_JSON(%s) AS bullets,
    PARSE_JSON(%s) AS actions,
    PARSE_JSON(%s) AS top_movers_used,
    PARSE_JSON(%s) AS kpi_used,
    %s::STRING AS model_name,
    %s::NUMBER AS prompt_tokens,
    %s::STRING AS raw_prompt,
    PARSE_JSON(%s) AS raw_response,
    %s::STRING AS display_text
) s
ON  t.insight_date = s.insight_date
AND NVL(t.insight_type,'') = NVL(s.insight_type,'')
WHEN MATCHED THEN UPDATE SET
  headline        = s.headline,
  bullets         = s.bullets,
  actions         = s.actions,
  top_movers_used = s.top_movers_used,
  kpi_used        = s.kpi_used,
  model_name      = s.model_name,
  prompt_tokens   = s.prompt_tokens,
  raw_prompt      = s.raw_prompt,
  raw_response    = s.raw_response,
  display_text    = s.display_text,
  created_at      = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
  insight_date, insight_type, headline, bullets, actions,
  top_movers_used, kpi_used, model_name, prompt_tokens,
  raw_prompt, raw_response, display_text
) VALUES (
  s.insight_date, s.insight_type, s.headline, s.bullets, s.actions,
  s.top_movers_used, s.kpi_used, s.model_name, s.prompt_tokens,
  s.raw_prompt, s.raw_response, s.display_text
)
"""
    cur.execute(
        sql,
        (
            insight_date,
            insight_type,
            insight.get("headline", ""),
            bullets_json,
            actions_json,
            movers_json,
            kpi_json,
            model_name,
            0 if prompt_tokens is None else int(prompt_tokens),
            raw_prompt or "",
            raw_resp_json,
            display_text,
        ),
    )



def add_months(d: date, months: int) -> date:
    """Like Excel EDATE: shift month, clamp the day to the last valid day."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # clamp day to end-of-month
    # find first of next month, then step back one day
    first_next = (date(y, m, 1).replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day   = (first_next - timedelta(days=1)).day
    return date(y, m, min(d.day, last_day))








# -------------------------------------------------------------------
# Main driver: generate FOUR insights per anchor date
# -------------------------------------------------------------------
def main():
    cn = connect_snowflake()
    cur = cn.cursor()
    try:
        ensure_columns(cur)

        # Anchor = latest date in DIM_DATE (keeps UI + AI in sync even if no sales that day)
        cur.execute(f"SELECT MIN(date), MAX(date) FROM {DIM_FQN}")
        min_d, max_d = cur.fetchone()
        if max_d is None:
            raise RuntimeError("DIM_DATE is empty")

        for insight_type in ("Total","Month","Week","Day"):
            start, end = period_window(max_d, insight_type, min_d)

            # Build inputs
            kpi = kpi_for_window(cur, start, end)
            movers = movers_for_window(cur, start, end)

            # LLM
            insight = call_gemini(kpi, movers)

            # Upsert
            upsert_ai_insight(
                cur,
                insight_date=end,             # store under anchor day
                insight_type=insight_type,
                insight=insight,
                kpi=kpi,
                movers=movers,
                model_name="gemini-2.5-flash",
                prompt_tokens=None,
                raw_prompt=None,
                raw_response=insight,         # optional: store final JSON
            )

        cn.commit()
        print("[OK] Wrote insights for Total, Month, Week, Day.")

    finally:
        try: cur.close()
        except: pass
        cn.close()


if __name__ == "__main__":
    main()