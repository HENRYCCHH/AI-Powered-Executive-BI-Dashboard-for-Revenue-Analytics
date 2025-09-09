# ai_insight_backfill.py
import os, json, calendar, time
from datetime import date, timedelta, datetime, timezone
from typing import Tuple, Dict, List, Optional

from dotenv import load_dotenv
import snowflake.connector

# --- Optional LLM (falls back to defaults if key not set) ---------------------
GEMINI_AVAILABLE = False
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -------------------- ENV & clients ------------------------------------------
load_dotenv()

SF_ACCOUNT   = os.environ["SF_ACCOUNT"]
SF_USER      = os.environ["SF_USER"]
SF_PASSWORD  = os.environ.get("SF_PASSWORD")
SF_ROLE      = os.environ["SF_ROLE"]
SF_WAREHOUSE = os.environ["SF_WAREHOUSE"]
SF_DATABASE  = os.environ["SF_DATABASE"]
SF_SCHEMA    = os.environ["SF_SCHEMA"]

AI_TABLE      = os.getenv("SF_AI_TABLE", "AI_INSIGHTS")            # may be unqualified
# Prefer fully-qualified table if provided; otherwise use short name
DIM_TABLE     = os.getenv("SF_DIM_TABLE_FQN") or os.getenv("SF_DIM_TABLE", "DIM_DATE")
FCT_PURCHASES = os.getenv("FCT_PURCHASES", "fct_purchases_daily")  # can be fully qualified
FCT_ATTRIB    = os.getenv("FCT_ATTRIBUTION", "fct_attribution_daily")

MIN_REV_LLM      = float(os.getenv("MIN_REVENUE_FOR_LLM", "0"))
MIN_ORD_LLM      = int(os.getenv("MIN_ORDERS_FOR_LLM", "0"))
STOP_LLM_BEFORE  = os.getenv("STOP_LLM_BEFORE")  # e.g. "2020-12-01" (optional)

GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "4"))     # polite defaults for free key
MAX_RPD = int(os.getenv("GEMINI_MAX_RPD", "24"))

# --- display-text tunables (same behavior as ai_display_compact.py) ----------
DISPLAY_MAX_LINES = int(os.getenv("DISPLAY_MAX_LINES", "9"))   # how many lines the card can show
DISPLAY_MAX_LEN   = int(os.getenv("DISPLAY_MAX_LEN",   "240")) # max chars per line

# Fallback actions if the LLM returns none (semi-colon separated override supported)
FALLBACK_ACTIONS = os.getenv(
    "DISPLAY_FALLBACK_ACTIONS",
    "Validate GA4 → warehouse pipeline.;Review paid campaigns & pacing.;Assess promo/seasonality effects."
).split(";")
FALLBACK_ACTIONS = [a.strip() for a in FALLBACK_ACTIONS if a.strip()]

_min_interval = 60.0 / max(1, MAX_RPM)
_last_call_ts = 0.0
_calls_made_today = 0
_calls_day_anchor = None  # YYYY-MM-DD for the count window

def _reset_daily_window_if_needed():
    global _calls_made_today, _calls_day_anchor
    today = datetime.now(timezone.utc).date()
    if _calls_day_anchor != today:
        _calls_day_anchor = today
        _calls_made_today = 0

def _sleep_to_respect_rpm():
    global _last_call_ts
    now = time.monotonic()
    wait = _min_interval - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()

# If AI table name is unqualified, prefix with current DB/Schema for DDL/MERGE
def fully_qualify(name: str) -> str:
    if "." in name:
        return name
    return f"{SF_DATABASE}.{SF_SCHEMA}.{name}"

AI_INSIGHTS_TBL = fully_qualify(AI_TABLE)

# LLM client (optional)
gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------- Snowflake connection -----------------------------------
def connect_snowflake():
    return snowflake.connector.connect(
        user=SF_USER,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        role=SF_ROLE,
        warehouse=SF_WAREHOUSE,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

def ensure_table(cur):
    """Create AI_INSIGHTS if needed + ensure columns exist."""
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {AI_INSIGHTS_TBL} (
      insight_date       DATE,
      insight_type       STRING,
      headline           STRING,
      bullets            VARIANT,
      actions            VARIANT,
      top_movers_used    VARIANT,
      kpi_used           VARIANT,
      model_name         STRING,
      prompt_tokens      NUMBER,
      created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
      raw_prompt         STRING,
      raw_response       VARIANT,
      display_text       STRING
    )
    """)
    # idempotent adds (ok if they already exist)
    cur.execute(f"ALTER TABLE {AI_INSIGHTS_TBL} ADD COLUMN IF NOT EXISTS INSIGHT_TYPE STRING")
    cur.execute(f"ALTER TABLE {AI_INSIGHTS_TBL} ADD COLUMN IF NOT EXISTS DISPLAY_TEXT STRING")

# -------------------- date window helpers ------------------------------------
def add_months(d: date, months: int) -> date:
    """EDATE-like: shift by N months, clamp day."""
    y = d.year + (d.month - 1 + months)//12
    m = (d.month - 1 + months)%12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))

def period_window(anchor: date, period: str, min_date: date) -> Tuple[date, date]:
    p = period.lower()
    if p == "day":
        return anchor, anchor
    if p == "week":
        return anchor - timedelta(days=6), anchor           # last 7 days rolling
    if p == "month":
        return add_months(anchor, -1) + timedelta(days=1), anchor  # rolling month
    if p == "total":
        return min_date, anchor
    raise ValueError(f"Unknown period {period}")

def prev_window(start: date, end: date) -> Tuple[date, date]:
    days = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start, prev_end

def _clean_lines(items) -> List[str]:
    """Normalize list-of-strings: coerce to str, strip, drop empties."""
    if not items:
        return []
    out = []
    for x in items:
        s = str(x).strip()
        if s:
            out.append(s)
    return out

def sanitize_insight(insight: Dict) -> Dict:
    """
    Ensure headline/bullets/actions exist, trim to a compact set, and
    guarantee at least one action so 'Next steps' never renders empty.
    """
    head    = str(insight.get("headline") or "").strip()
    bullets = _clean_lines(insight.get("bullets"))[:4]   # keep it tight
    actions = _clean_lines(insight.get("actions"))

    if not actions:
        actions = FALLBACK_ACTIONS[:]                    # always at least one
    actions = actions[:3]

    return {"headline": head, "bullets": bullets, "actions": actions}



# -------------------- compact display text -----------------------------------
def render_display_text(insight: Dict) -> str:
    """
    Compact headline + bullets + 'Next steps' (always present with ≥1 action).
    Uses DISPLAY_MAX_LINES / DISPLAY_MAX_LEN limits.
    """
    # Make sure content is normalized
    i = sanitize_insight(insight)

    lines: List[str] = []
    if i["headline"]:
        lines.append(i["headline"])

    for b in i["bullets"]:
        lines.append(f"• {b}")

    # Always show Next steps with at least one action
    lines.append("")
    lines.append("Next steps")
    for a in i["actions"]:
        lines.append(f"– {a}")

    # Crop for UI hygiene
    lines = [l[:DISPLAY_MAX_LEN] for l in lines][:DISPLAY_MAX_LINES]
    return "\n".join(lines)

# -------------------- KPI & Movers over a window -----------------------------
def fetch_kpi_for_window(cur, start_d: date, end_d: date) -> Dict:
    cur.execute(f"""
        SELECT COALESCE(SUM(revenue),0) AS revenue,
               COALESCE(SUM(orders),0)  AS orders
        FROM {FCT_PURCHASES}
        WHERE date BETWEEN %s AND %s
    """, (start_d, end_d))
    rev, ords = cur.fetchone()

    p0, p1 = prev_window(start_d, end_d)
    cur.execute(f"""
        SELECT COALESCE(SUM(revenue),0) AS revenue,
               COALESCE(SUM(orders),0)  AS orders
        FROM {FCT_PURCHASES}
        WHERE date BETWEEN %s AND %s
    """, (p0, p1))
    prev_rev, prev_orders = cur.fetchone()

    aov = (rev / ords) if ords else 0.0
    delta = rev - prev_rev
    pct = (delta / prev_rev) if prev_rev else None

    return {
        "start_date":    start_d.isoformat(),
        "end_date":      end_d.isoformat(),
        "revenue":       float(rev),
        "orders":        int(ords),
        "aov":           float(aov),
        "prev_revenue":  float(prev_rev),
        "prev_orders":   int(prev_orders),
        "rev_delta":     float(delta),
        "rev_pct":       float(pct) if pct is not None else None,
    }

def fetch_top_movers_for_window(cur, start_d: date, end_d: date,
                                min_rev_cur: float = 50.0, limit: int = 8) -> List[Dict]:
    p0, p1 = prev_window(start_d, end_d)
    sql = f"""
WITH s AS (
  SELECT
    source, medium, campaign,
    SUM(IFF(date BETWEEN %s AND %s, revenue, 0)) AS rev_cur,
    SUM(IFF(date BETWEEN %s AND %s, revenue, 0)) AS rev_prev
  FROM {FCT_ATTRIB}
  GROUP BY 1,2,3
),
m AS (
  SELECT
    source, medium, campaign,
    rev_cur, rev_prev,
    (rev_cur - rev_prev) AS delta,
    IFF(rev_prev=0,NULL,(rev_cur-rev_prev)/NULLIF(rev_prev,0)) AS pct,
    ABS(rev_cur - rev_prev) * 0.7
      + ABS( (rev_cur - rev_prev) / NULLIF(rev_prev,0) ) * 0.3 AS impact_score
  FROM s
)
SELECT
  source, medium, campaign, rev_cur, rev_prev, delta, pct, impact_score
FROM m
WHERE rev_cur >= %s
ORDER BY impact_score DESC NULLS LAST
LIMIT %s
"""
    cur.execute(sql, (start_d, end_d, p0, p1, min_rev_cur, limit))
    movers: List[Dict] = []
    for (src, med, camp, rc, rp, dlt, pct, score) in cur.fetchall():
        movers.append({
            "source": src or "(none)",
            "medium": med or "(none)",
            "campaign": camp or "(none)",
            "revenue_cur": float(rc or 0),
            "revenue_prev": float(rp or 0),
            "delta_abs": float(dlt or 0) if dlt is not None else None,
            "delta_pct": float(pct) if pct is not None else None,
            "impact_score": float(score or 0),
        })
    return movers

# -------------------- Default (no-LLM) insight -------------------------------
def default_insight_text(kpi: Dict) -> Dict:
    rev = kpi.get("revenue", 0.0)
    ords = kpi.get("orders", 0)
    start = kpi.get("start_date")
    end   = kpi.get("end_date")
    head = "No purchase activity in this period." if (rev <= 0 and ords <= 0) else "Low activity in this period."
    bullets = [
        f"Window: {start} → {end}",
        f"Revenue: ${rev:,.0f}, Orders: {ords:,}",
        "Check channel spend, tagging, and data freshness."
    ]
    actions = [
        "Validate GA4 → warehouse pipeline.",
        "Review paid campaigns & pacing.",
        "Assess promo/seasonality effects."
    ]
    return {"headline": head, "bullets": bullets, "actions": actions}

# -------------------- Gemini call (JSON-out) ---------------------------------
def gemini_generate_with_limits(llm_client, prompt, schema,
                                model="gemini-2.5-flash",
                                temperature=0.2, max_retries=5):
    """
    Calls Gemini with throttling + retries.
    Returns parsed JSON (dict) if schema is enforced; otherwise response text.
    """
    global _calls_made_today
    _reset_daily_window_if_needed()

    if _calls_made_today >= MAX_RPD:
        raise RuntimeError(f"Daily Gemini cap reached (MAX_RPD={MAX_RPD}).")

    # Respect RPM before attempting
    _sleep_to_respect_rpm()

    # Exponential backoff on 429/5xx
    backoff = 1.5
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm_client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": temperature,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            _calls_made_today += 1

            txt = resp.text or ""
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                return {"headline": "AI insight", "bullets": [txt.strip()], "actions": []}

        except Exception as e:
            msg = str(e).lower()
            retryable = ("429" in msg) or ("quota" in msg) or ("rate" in msg) or ("timeout" in msg) or ("503" in msg)
            if attempt == max_retries or not retryable:
                raise
            time.sleep(delay)
            delay *= backoff

def call_gemini(kpi: Dict, movers: List[Dict]) -> Dict:
    if gemini_client is None:
        return default_insight_text(kpi)

    system_rules = (
        "You are a data analyst. Write concise, business-friendly insights.\n"
        "Return STRICT JSON only. No extra prose."
    )
    schema = {
      "type": "object",
      "properties": {
        "headline": {"type":"string"},
        "bullets":  {"type":"array","items":{"type":"string"}},
        "actions":  {"type":"array","items":{"type":"string"}}
      },
      "required":["headline","bullets","actions"]
    }
    payload = {"kpi": kpi, "top_movers": movers}
    prompt = f"""{system_rules}

Schema:
{json.dumps(schema, indent=2)}

Data:
{json.dumps(payload, separators=(',',':'))}
Return only valid JSON per the schema; no prose.
"""
    insight = gemini_generate_with_limits(
        llm_client=gemini_client,
        prompt=prompt,
        schema=schema,
        model=GEMINI_MODEL,
        temperature=0.2
    )
    return insight

# -------------------- Upsert row (date + type) -------------------------------
def upsert_ai_insight(cur,
                      insight_date: date,
                      insight_type: str,
                      insight: Dict,
                      kpi: Dict,
                      movers: List[Dict],
                      model_name: str,
                      prompt_tokens: Optional[int] = None,
                      raw_prompt: Optional[str] = None,
                      raw_response: Optional[Dict | List | str] = None):
    # --- NEW: normalize first so we always have clean bullets/actions
    clean = sanitize_insight(insight)

    bullets_json = json.dumps(clean.get("bullets", []), ensure_ascii=False)
    actions_json = json.dumps(clean.get("actions", []), ensure_ascii=False)
    movers_json  = json.dumps(movers or [], ensure_ascii=False)
    kpi_json     = json.dumps(kpi or {}, ensure_ascii=False)

    display_text = render_display_text(clean)

    if isinstance(raw_response, (dict, list)):
        raw_resp_json = json.dumps(raw_response, ensure_ascii=False)
    elif isinstance(raw_response, str) and raw_response.strip():
        raw_resp_json = raw_response
    else:
        raw_resp_json = "{}"

    sql = f"""
MERGE INTO {AI_INSIGHTS_TBL} t
USING (
  SELECT
    %s::DATE    AS insight_date,
    %s::STRING  AS insight_type,
    %s::STRING  AS headline,
    PARSE_JSON(%s) AS bullets,
    PARSE_JSON(%s) AS actions,
    PARSE_JSON(%s) AS top_movers_used,
    PARSE_JSON(%s) AS kpi_used,
    %s::STRING  AS model_name,
    %s::NUMBER  AS prompt_tokens,
    %s::STRING  AS raw_prompt,
    PARSE_JSON(%s) AS raw_response,
    %s::STRING  AS display_text
) s
ON t.insight_date = s.insight_date AND t.insight_type = s.insight_type
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
);
"""
    cur.execute(
        sql,
        (
            insight_date,
            insight_type,
            clean.get("headline", ""),  # <— use clean headline
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

# -------------------- Backfill driver ----------------------------------------
def backfill_all_days(cur,
                      min_anchor: Optional[date] = None,
                      max_anchor: Optional[date] = None):
    # date span from DIM_DATE (can be fully qualified)
    cur.execute(f"SELECT MIN(date), MAX(date) FROM {DIM_TABLE}")
    dim_min, dim_max = cur.fetchone()

    if min_anchor is None: min_anchor = dim_min
    if max_anchor is None: max_anchor = dim_max

    # optional LLM cutoff (use defaults before this date)
    stop_llm_date = date.fromisoformat(STOP_LLM_BEFORE) if STOP_LLM_BEFORE else None

    periods = ["total", "month", "week", "day"]

    d = min_anchor
    total_rows = 0
    while d <= max_anchor:
        for p in periods:
            start_d, end_d = period_window(d, p, dim_min)
            kpi = fetch_kpi_for_window(cur, start_d, end_d)
            movers = fetch_top_movers_for_window(cur, start_d, end_d, min_rev_cur=50.0, limit=8)

            use_llm = gemini_client is not None \
                      and (kpi["revenue"] >= MIN_REV_LLM) \
                      and (kpi["orders"]  >= MIN_ORD_LLM)
            if stop_llm_date and end_d < stop_llm_date:
                use_llm = False

            if use_llm:
                enriched = dict(kpi); enriched["period"] = p
                insight = call_gemini(enriched, movers)
                model_used = GEMINI_MODEL
            else:
                insight = default_insight_text(kpi)
                model_used = "default"

            upsert_ai_insight(cur, d, p, insight, kpi, movers, model_used)
            total_rows += 1

        cur.connection.commit()
        print(f"[OK] {d} -> wrote 4 insights")
        d += timedelta(days=1)

    print(f"[DONE] backfill complete. rows written/updated: {total_rows}")

# -------------------- Main ----------------------------------------------------
def main():
    cn = connect_snowflake()
    try:
        cur = cn.cursor()
        cur.execute(f"USE DATABASE {SF_DATABASE}")
        cur.execute(f"USE SCHEMA {SF_SCHEMA}")
        ensure_table(cur)

        # Limit window via env (optional)
        bf_start = os.getenv("BACKFILL_START")
        bf_end   = os.getenv("BACKFILL_END")
        min_anchor = date.fromisoformat(bf_start) if bf_start else None
        max_anchor = date.fromisoformat(bf_end) if bf_end else None

        backfill_all_days(cur, min_anchor, max_anchor)

        # Quick peek
        cur.execute(f"""
          SELECT insight_date, insight_type, LEFT(headline,80) AS headline
          FROM {AI_INSIGHTS_TBL}
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY insight_date, insight_type ORDER BY created_at DESC
          ) = 1
          ORDER BY insight_date DESC, insight_type
          LIMIT 16
        """)
        for r in cur.fetchall():
            print("[ROW]", r)

    finally:
        try: cur.close()
        except: pass
        cn.close()

if __name__ == "__main__":
    main()