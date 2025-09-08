import os, json, datetime as dt
from dotenv import load_dotenv
import snowflake.connector
from google import genai
from google.genai import types
from datetime import date, timedelta
import datetime as dt


load_dotenv()
# after load_dotenv()
AI_DB       = os.environ["SF_DATABASE"]          # GA4_DEMO
AI_SCHEMA   = os.environ["SF_SCHEMA"]            # DEV_MART
AI_TABLE    = os.getenv("SF_AI_TABLE", "AI_INSIGHTS")

DIM_DB      = AI_DB                               # same DB
DIM_SCHEMA  = os.getenv("SF_DIM_SCHEMA", AI_SCHEMA)  # DEV
DIM_DATE_TB = os.getenv("SF_DIM_TABLE", "DIM_DATE")

# FQNs
AI_INSIGHTS_TBL = f"{AI_DB}.{AI_SCHEMA}.{AI_TABLE}"
DIM_DATE_FQN    = f"{DIM_DB}.{DIM_SCHEMA}.{DIM_DATE_TB}"
FCT_PURCHASES   = f"{AI_DB}.{AI_SCHEMA}.FCT_PURCHASES_DAILY"
TREND_VIEW      = f"{AI_DB}.{AI_SCHEMA}.V_PURCHASE_TRENDS"
MOVERS_VIEW     = f"{AI_DB}.{AI_SCHEMA}.V_AI_INSIGHT"

# --- 0) Clients ---------------------------------------------------------------
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def connect_snowflake():
    """
    Use password auth (simple). If you want key-pair, see commented section below.
    """
    return snowflake.connector.connect(
        user=os.environ["SF_USER"],
        password=os.environ.get("SF_PASSWORD"),
        account=os.environ["SF_ACCOUNT"],
        role=os.environ["SF_ROLE"],
        warehouse=os.environ["SF_WAREHOUSE"],
        database=os.environ["SF_DATABASE"],
        schema=os.environ["SF_SCHEMA"],
    )

# For key-pair auth instead of password:
# import pathlib, base64, cryptography.hazmat.primitives.serialization as ser
# private_key_pem = pathlib.Path("sf_pk8.pem").read_bytes()
# private_key = ser.load_pem_private_key(private_key_pem, password=b"your_passphrase")
# pk_bytes = private_key.private_bytes(ser.Encoding.DER, ser.PrivateFormat.PKCS8, ser.NoEncryption())
# return snowflake.connector.connect(..., private_key=pk_bytes, ...)

# --- 1) Pull data we want Gemini to see --------------------------------------
def fetch_inputs(cur, strict_sync: bool = True):
    """
    Pick the anchor date from the dense calendar so AI matches BI.
    Pull KPIs from the dense trend view (zeros on no-activity days).
    Movers are taken for the same date; if none and strict_sync=True, return [].
    """

    # 1) Anchor on the MAX date in DIM_DATE (dense calendar)
    cur.execute(f"SELECT MAX(date) FROM {DIM_DATE_FQN}")
    anchor_date = cur.fetchone()[0]
    if anchor_date is None:
        raise RuntimeError("DIM_DATE is empty")

    # 2) KPIs from your dense trend view (left-joined to DIM_DATE).
    #    Use the name you actually built. If your project uses
    #    `v_purchase_trend_calender` (typo) or `v_purchase_trends_calendar`,
    #    change the FROM accordingly.
    cur.execute(f"""
        SELECT
          t.date, t.revenue, t.orders, t.aov,
          t.rev_wow_delta, t.rev_wow_pct, t.rev_anomaly_flag
        FROM {TREND_VIEW} t
        WHERE t.date = %s
    """, (anchor_date,))
    row = cur.fetchone()

    if row:
        kpi = {
            "date":          row[0].isoformat(),
            "revenue":       float(row[1] or 0),
            "orders":        int(row[2] or 0),
            "aov":           float(row[3] or 0),
            "rev_wow_delta": float(row[4]) if row[4] is not None else None,
            "rev_wow_pct":   float(row[5]) if row[5] is not None else None,
            "anomaly_flag":  int(row[6] or 0),
        }
    else:
        # Dense view should return a row; but just in case, emit zero KPIs
        kpi = {
            "date":          anchor_date.isoformat(),
            "revenue":       0.0,
            "orders":        0,
            "aov":           0.0,
            "rev_wow_delta": None,
            "rev_wow_pct":   None,
            "anomaly_flag":  0,
        }

    # 3) Top movers for the same anchor date
    cur.execute(f"""
        SELECT
          source, medium, campaign,
          rev_cur, rev_w7, rev_wow_delta, rev_wow_pct, impact_score
        FROM {MOVERS_VIEW}
        WHERE date = %s
        ORDER BY impact_score DESC NULLS LAST
        LIMIT 8
    """, (anchor_date,))
    rows = cur.fetchall()

    # Optional fallback: if you want movers even when the anchor day is empty,
    # set strict_sync=False and we'll use the most recent prior date with movers.
    if not rows and not strict_sync:
        cur.execute("""
            SELECT MAX(date) FROM {MOVERS_VIEW} WHERE date <= %s
        """, (anchor_date,))
        alt = cur.fetchone()[0]
        if alt:
            cur.execute(f"""
                SELECT
                  source, medium, campaign,
                  rev_cur, rev_w7, rev_wow_delta, rev_wow_pct, impact_score
                FROM {MOVERS_VIEW}
                WHERE date = %s
                ORDER BY impact_score DESC NULLS LAST
                LIMIT 8
            """, (alt,))
            rows = cur.fetchall()

    movers = []
    for (source, medium, campaign, cur_rev, w7, dlt, pct, score) in rows or []:
        movers.append({
            "source":         source or "(none)",
            "medium":         medium or "(none)",
            "campaign":       campaign or "(none)",
            "revenue_today":  float(cur_rev or 0),
            "revenue_w7":     float(w7) if w7 is not None else None,
            "delta_abs":      float(dlt) if dlt is not None else None,
            "delta_pct":      float(pct) if pct is not None else None,
            "impact_score":   float(score or 0),
        })

    return anchor_date, kpi, movers

# expects GEMINI_API_KEY in your environment (.env loaded earlier)

def _shorten(s: str, max_len: int) -> str:
    s = (s or "").strip()
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

def _format_for_display(headline: str, bullets: list[str], actions: list[str]) -> str:
    # Single field you can bind to a text box (optional)
    lines = []
    if headline:
        lines.append(headline)
    if bullets:
        lines.append("")  # blank line
        lines.extend([f"• {b}" for b in bullets])
    if actions:
        lines.append("")
        lines.append("Next steps")
        lines.extend([f"– {a}" for a in actions])
    return "\n".join(lines)

def call_gemini(kpi: dict, movers: list[dict]) -> dict:
    """
    Returns a compact dict:
      {
        "headline": str (<=60 chars),
        "bullets":  [up to 3 items, each <=90 chars],
        "actions":  [up to 3 items, each <=80 chars],
        "display":  single string for a text box (optional)
      }
    """
    system_rules = (
        "You are a data analyst. Write clear, business-friendly insights.\n"
        "Return STRICT JSON that matches the schema below. Do not include extra text.\n"
        "Constraints:\n"
        "- headline ≤ 80 characters\n"
        "- bullets: 2–3 items, each ≤ 110 characters\n"
        "- actions: 1–2 items, each ≤ 90 characters\n"
        "- No fluff, no repeated numbers, keep it crisp."
    )

    # We’ll still use a schema so the SDK enforces JSON back.
    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "bullets":  {"type": "array", "items": {"type": "string"}},
            "actions":  {"type": "array", "items": {"type": "string"}}
        },
        "required": ["headline", "bullets", "actions"]
    }

    # Cap the context you send (no raw PII)
    payload = {
        "kpi": kpi,
        # send at most 5 movers to keep prompt compact
        "top_movers": movers[:5]
    }

    # Ultra-explicit constraints so it fits your Power BI card
    style_guardrails = (
        "Constraints:\n"
        "• Headline: max 60 characters.\n"
        "• Return exactly 2–3 bullets (max 90 chars each).\n"
        "• Return 1–3 actionable next steps (max 80 chars each).\n"
        "• No markdown, no extra commentary, JSON only."
    )

    prompt = f"""
{system_rules}

{style_guardrails}

Schema:
{json.dumps(schema, indent=2)}

Data:
{json.dumps(payload, separators=(',', ':'))}

Return only valid JSON per the schema; no prose.
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,  # keep it crisp & consistent
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # Parse & sanitize
    txt = resp.text or ""
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = {"headline": "AI Insight", "bullets": [txt.strip()], "actions": []}

    # Normalize shapes
    head = _shorten(str(data.get("headline", "")), 60)
    bullets = [ _shorten(str(b), 90) for b in (data.get("bullets") or []) ]
    actions = [ _shorten(str(a), 80) for a in (data.get("actions") or []) ]

    # Enforce counts
    if not bullets:
        # fallback bullet if model was too terse
        bullets = ["Performance drivers summarized for the selected period."]
    bullets = bullets[:3]
    actions = actions[:3]

    # Build a single display field (optional – doesn’t change your table schema)
    display = _format_for_display(head, bullets, actions)

    return {"headline": head, "bullets": bullets, "actions": actions, "display": display}


def init_session(cur):
    cur.execute(f"USE ROLE {os.environ['SF_ROLE']}")
    cur.execute(f"USE WAREHOUSE {os.environ['SF_WAREHOUSE']}")
    cur.execute(f"USE DATABASE {AI_DB}")
    cur.execute(f"USE SCHEMA {AI_SCHEMA}")



def upsert_ai_insight(
    cur,
    insight_date,
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
    raw_resp_json = (
        json.dumps(raw_response, ensure_ascii=False)
        if isinstance(raw_response, (dict, list))
        else (raw_response if isinstance(raw_response, str) and raw_response.strip() else "{}")
    )
    display_text = render_display_text(insight, max_lines=7, max_line_len=120)

    sql = """
MERGE INTO GA4_DEMO.DEV_MART.AI_INSIGHTS t
USING (
  SELECT
    %s::DATE    AS insight_date,
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
ON t.insight_date = s.insight_date
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
  insight_date, headline, bullets, actions, top_movers_used, kpi_used,
  model_name, prompt_tokens, raw_prompt, raw_response, display_text
) VALUES (
  s.insight_date, s.headline, s.bullets, s.actions, s.top_movers_used, s.kpi_used,
  s.model_name, s.prompt_tokens, s.raw_prompt, s.raw_response, s.display_text
);
"""
    cur.execute(
        sql,
        (
            insight_date,
            insight.get("headline", ""),
            bullets_json,
            actions_json,
            movers_json,
            kpi_json,
            model_name,
            0 if prompt_tokens is None else int(prompt_tokens),
            raw_prompt or "",
            raw_resp_json,
            display_text,                   # <— NEW BIND
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


def _trim_to_words(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    cut = s.rfind(" ", 0, max_len - 1)
    if cut == -1:
        return s[: max_len - 1] + "…"
    return s[:cut].rstrip() + "…"



def _cap_list(items, max_items: int, label: str):
    """Keep at most `max_items`; if more, add a ‘(+N more …)’ line."""
    items = [i for i in items if str(i).strip()]
    if len(items) <= max_items:
        return items
    extra = len(items) - max_items
    return items[:max_items] + [f"(+{extra} more {label})"]



def period_window(anchor: date, period: str, min_date: date) -> tuple[date, date]:
    """
    Inclusive window for insights ending at 'anchor'.
    - 'day'   : that one day
    - 'week'  : last 7 days (anchor-6 .. anchor)
    - 'month' : last month *rolling* (EDATE(anchor,-1)+1 .. anchor)
    - 'total' : all-time to date (min_date .. anchor)
    """
    p = period.lower()
    if p == "day":
        start = anchor
    elif p == "week":
        start = anchor - timedelta(days=6)
    elif p == "month":
        # Rolling month: e.g. anchor=2025-09-02 -> start=2025-08-03
        start = add_months(anchor, -1) + timedelta(days=1)
    elif p == "total":
        start = min_date
    else:
        raise ValueError(f"Unknown period {period}")

    # clamp to available data
    start = max(start, min_date)
    return start, anchor



def render_display_text(insight: dict,
                        max_lines: int = 7,
                        max_line_len: int = 120) -> str:
    """
    Compact, readable box:
      - 1 headline
      - up to 3 bullets
      - 'Next steps' + up to 2 actions
      - word-safe truncation + ellipses
      - adds '(+N more …)' lines if we clipped items
    """
    head = _trim_to_words(insight.get("headline", ""), 80)

    bullets = [str(b).strip() for b in insight.get("bullets", []) if str(b).strip()]
    actions = [str(a).strip() for a in insight.get("actions", []) if str(a).strip()]

    bullets = _cap_list(bullets, 3, "insight(s)")
    actions = _cap_list(actions, 2, "action(s)")

    lines = []
    if head:
        lines.append(head)

    for b in bullets:
        lines.append("• " + _trim_to_words(b, max_line_len - 2))

    if actions:
        lines.append("")  # blank line
        lines.append("Next steps")
        for a in actions:
            lines.append("– " + _trim_to_words(a, max_line_len - 2))

    # final safety: crop per-line and total lines
    lines = [ _trim_to_words(l, max_line_len) for l in lines ][:max_lines]
    return "\n".join(lines)

def main():
    # 1) Connect to Snowflake
    cn = connect_snowflake()
    try:
        cur = cn.cursor()
        # 2) Make sure we’re writing to the right DB/Schema (redundant if set on connect)
        cur.execute(f"USE DATABASE {os.environ['SF_DATABASE']}")
        cur.execute(f"USE SCHEMA {os.environ['SF_SCHEMA']}")

        # 3) Pull inputs
        latest_date, kpi, movers = fetch_inputs(cur)
        print(f"[INFO] Latest date = {latest_date}, movers={len(movers)}")

        # 4) Call Gemini
        insight = call_gemini(kpi, movers)
        print(f"[INFO] Headline: {insight.get('headline','')}")
        # Optional: print bullets/actions for debugging
        # print(json.dumps(insight, indent=2))

        # 5) Upsert into Snowflake
        upsert_ai_insight(
            cur,
            latest_date,
            insight,
            kpi,
            movers,
            model_name="gemini-2.5-flash"   # or whatever you used
        )
        cn.commit()
        print("[OK] Insight upserted.")
        # Snowflake connector autocommits by default; if you disabled it, uncomment:
        # cn.commit()

        # 6) Quick verification
        cur.execute("""
            SELECT INSIGHT_DATE, HEADLINE, LEFT(DISPLAY_TEXT, 120)
            FROM GA4_DEMO.DEV_MART.AI_INSIGHTS
            ORDER BY INSIGHT_DATE DESC
            LIMIT 3
        """)
        print("[INFO] Recent rows:", cur.fetchall())

    finally:
        try:
            cur.close()
        except Exception:
            pass
        cn.close()


if __name__ == "__main__":
    main()