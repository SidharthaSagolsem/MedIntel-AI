# app.py
# MedIntel AI – Clinical Document Intelligence System
# Main Streamlit application entry point.
#
# Run with:  streamlit run app.py

import re
import streamlit as st
import time

# ── Page Configuration (MUST be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="MedIntel AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    /* gradient text title */
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #38bdf8;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* entity pills */
    .entity-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #7dd3fc;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        margin: 3px;
        font-weight: 500;
    }
    .entity-pill-med {
        background: rgba(52, 211, 153, 0.15);
        border-color: rgba(52, 211, 153, 0.4);
        color: #6ee7b7;
    }

    /* timeline */
    .timeline-item {
        padding: 0.75rem 0 0.75rem 1.5rem;
        border-left: 2px solid #334155;
        position: relative;
        margin-left: 0.5rem;
    }
    .timeline-item::before {
        content: '';
        width: 10px; height: 10px;
        background: #38bdf8;
        border-radius: 50%;
        position: absolute;
        left: -6px; top: 50%;
        transform: translateY(-50%);
    }
    .timeline-date  { font-size: 0.78rem; color: #94a3b8; font-weight: 500; }
    .timeline-event { color: #e2e8f0; font-size: 0.88rem; margin-top: 2px; }

    /* lab table */
    .lab-row {
        display: flex; justify-content: space-between;
        padding: 0.5rem 0; border-bottom: 1px solid #1e293b; font-size: 0.88rem;
    }
    .lab-name  { color: #94a3b8; }
    .lab-value { color: #fcd34d; font-weight: 600; }

    /* chat bubbles */
    .chat-user {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #e2e8f0;
    }
    .chat-ai {
        background: rgba(129, 140, 248, 0.1);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #e2e8f0;
    }
    .chat-label {
        font-size: 0.72rem; color: #64748b; margin-bottom: 4px;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* metric card */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155; border-radius: 12px;
        padding: 1rem; text-align: center;
    }
    .metric-number {
        font-size: 2rem; font-weight: 700; color: #38bdf8;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

    .source-box {
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #475569; border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem; font-size: 0.8rem; color: #94a3b8;
        margin: 0.3rem 0; font-style: italic;
    }

    .conf-high   { color: #34d399; font-weight: 600; }
    .conf-medium { color: #fbbf24; font-weight: 600; }
    .conf-low    { color: #f87171; font-weight: 600; }
    .conf-none   { color: #64748b; font-weight: 600; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "processed":    False,
        "raw_text":     "",
        "entities":     {},
        "timeline":     [],
        "risks":        [],
        "chat_history": [],
        "doc_name":     "",
        "page_count":   0,
        "anonymize":    True, # Default privacy to ON
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Document Processing Pipeline ──────────────────────────────────────────────
def process_document(uploaded_file):
    """
    Runs the full analysis pipeline:
    PDF ingestion → entity extraction → timeline → risk detection → RAG index
    """
    from ingestion  import extract_text_from_pdf, get_page_count
    from extraction import extract_medical_entities
    from timeline   import build_timeline
    from risk       import detect_risks
    from utils      import anonymize_text
    import rag

    file_bytes = uploaded_file.read()

    bar = st.progress(0, text="📖 Extracting text…")
    raw_text   = extract_text_from_pdf(file_bytes)
    page_count = get_page_count(file_bytes)

    if not raw_text or len(raw_text.strip()) < 30:
        bar.empty()
        st.error(
            "⚠️ Could not extract text from this PDF.\n\n"
            "If it's a scanned document, Tesseract OCR must be installed:\n"
            "- **Linux**: `sudo apt install tesseract-ocr`\n"
            "- **macOS**: `brew install tesseract`\n"
            "- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return

    # 🚨 PRIVACY MIDDLEWARE 🚨
    if st.session_state.anonymize:
        bar.progress(10, text="🔒 Anonymizing patient data…")
        raw_text = anonymize_text(raw_text)

    bar.progress(20, text="🧬 Extracting medical entities…")
    entities = extract_medical_entities(raw_text)

    bar.progress(50, text="📅 Building timeline…")
    timeline = build_timeline(raw_text, entities)

    bar.progress(70, text="🚨 Detecting risks…")
    risks = detect_risks(entities, raw_text)

    bar.progress(85, text="🔗 Indexing for Q&A…")
    rag.reset_index()
    rag.index_document(raw_text)

    bar.progress(100, text="✅ Complete!")
    time.sleep(0.5)
    bar.empty()

    st.session_state.update({
        "processed":    True,
        "raw_text":     raw_text,
        "entities":     entities,
        "timeline":     timeline,
        "risks":        risks,
        "doc_name":     uploaded_file.name,
        "page_count":   page_count,
        "chat_history": [],
    })

    st.success("✅ Analysis complete!")
    time.sleep(0.4)
    st.rerun()


# ── UI Sections ───────────────────────────────────────────────────────────────

def show_landing():
    """Welcome screen before any document is uploaded – upload widget front and centre."""
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0 1rem;">
            <div style="font-size:5rem;">🏥</div>
            <h2 style="font-family:'Space Grotesk',sans-serif;color:#e2e8f0;margin:1rem 0;">
                MedIntel AI – Clinical Document Intelligence
            </h2>
            <p style="color:#64748b;font-size:1rem;line-height:1.8;">
                Analyse clinical PDFs to extract diseases, medications, lab values,
                build an intelligent patient timeline, and chat with your document.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Prominent Upload Card ──────────────────────────────────────────────────
    st.markdown("""
    <div style="
        border: 2px dashed #38bdf8;
        border-radius: 20px;
        background: rgba(56,189,248,0.06);
        padding: 2rem 2rem 1rem;
        margin: 0.5rem 0 1.5rem;
        text-align: center;
    ">
        <div style="font-size:2.8rem;margin-bottom:0.5rem;">📂</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;
                    color:#38bdf8;font-weight:600;margin-bottom:0.3rem;">
            Drop your medical PDF here
        </div>
        <div style="color:#64748b;font-size:0.88rem;margin-bottom:1rem;">
            Discharge summaries · Clinical notes · Lab reports
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a medical PDF",
        type=["pdf"],
        key="main_uploader",
        help="Discharge summaries, clinical notes, lab reports – any medical PDF.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"📄 **{uploaded_file.name}** – ready to analyse")
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("🔬  Analyse Document", use_container_width=True, type="primary", key="main_analyse_btn"):
                process_document(uploaded_file)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    feats = [
        ("🧬", "Entity Extraction", "Diseases, medications & lab values", "#38bdf8"),
        ("📅", "Timeline",          "Chronological patient history",       "#818cf8"),
        ("🚨", "Risk Detection",    "Intelligent clinical alerts",          "#f87171"),
        ("💬", "Q&A Chat",          "Ask questions about the document",     "#34d399"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc, colour) in zip(cols, feats):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="color:{colour};font-weight:600;margin:0.5rem 0;">{title}</div>
                <div style="color:#64748b;font-size:0.82rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_patient_summary(entities: dict):
    """📊 Conditions, medications, lab values."""
    st.markdown("### 📊 Patient Summary")

    # Conditions
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🦠 Conditions / Diagnoses</div>', unsafe_allow_html=True)
    diseases = entities.get("diseases", [])
    if diseases:
        pills = "".join(f'<span class="entity-pill">{d}</span>' for d in diseases[:20])
        st.markdown(f'<div style="line-height:2.2;">{pills}</div>', unsafe_allow_html=True)
    else:
        st.caption("No conditions extracted.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Medications
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💊 Medications</div>', unsafe_allow_html=True)
    meds = entities.get("medications", [])
    if meds:
        pills = "".join(f'<span class="entity-pill entity-pill-med">{m}</span>' for m in meds[:20])
        st.markdown(f'<div style="line-height:2.2;">{pills}</div>', unsafe_allow_html=True)
    else:
        st.caption("No medications extracted.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Lab values
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Key Lab Values</div>', unsafe_allow_html=True)
    lab_values = entities.get("lab_values", [])
    if lab_values:
        for lv in lab_values:
            st.markdown(f"""
            <div class="lab-row">
                <span class="lab-name">{lv['name']}</span>
                <span class="lab-value">{lv['value']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No lab values extracted.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_timeline(timeline: list):
    """📅 Chronological patient timeline."""
    st.markdown("### 📅 Patient Timeline")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    if not timeline:
        st.caption("No timeline events could be constructed.")
    else:
        cat_colour = {
            "diagnosis":        "#38bdf8",
            "medication_start": "#34d399",
            "medication_stop":  "#f87171",
            "hospitalization":  "#f59e0b",
            "procedure":        "#a78bfa",
            "lab_event":        "#fcd34d",
            "follow_up":        "#94a3b8",
        }
        for item in timeline[:25]:
            date     = item.get("date",     "Date Unknown")
            event    = item.get("event",    "")
            category = item.get("category", "")
            colour   = cat_colour.get(category, "#64748b")

            st.markdown(f"""
            <div class="timeline-item" style="border-left-color:{colour};">
                <div class="timeline-date">📆 {date}</div>
                <div class="timeline-event">{event}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_risk_alerts(risks: list):
    """🚨 Risk severity alerts with explainability."""
    st.markdown("### 🚨 Risk Alerts")
    if not risks:
        st.success("✅ No significant risks detected.")
        return

    for idx, risk in enumerate(risks):
        level       = risk.get("level",       "info")
        title       = risk.get("title",       "")
        message     = risk.get("message",     "")
        explanation = risk.get("explanation", "")
        icon        = risk.get("icon",        "ℹ️")
        body        = f"**{icon} {title}** \n{message}"

        if level == "high":
            st.error(body)
        elif level == "medium":
            st.warning(body)
        elif level == "low":
            st.success(body)
        else:
            st.info(body)

        if explanation:
            with st.expander("🔍 Explain this risk", expanded=False):
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(56,189,248,0.07);
                        border-left: 3px solid #38bdf8;
                        border-radius: 0 8px 8px 0;
                        padding: 0.75rem 1rem;
                        font-size: 0.88rem;
                        color: #cbd5e1;
                        line-height: 1.7;
                    ">
                        💡 <strong style="color:#38bdf8;">Why was this flagged?</strong><br><br>
                        {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)


# ── Clinical Trends ──────────────────────────────────────────────────────────

_TREND_PATTERNS = {
    "HbA1c":           r"(?:hba1c|hemoglobin\s*a1c)\s*[:\-]?\s*([\d.]+)",
    "Blood Glucose":   r"(?:blood\s*glucose|fasting\s*glucose|rbs|fbs|ppbs)\s*[:\-]?\s*([\d.]+)",
    "Blood Pressure (Systolic)": r"(?:bp|blood\s*pressure)\s*[:\-]?\s*([\d]+)\s*/\s*[\d]+",
    "Creatinine":      r"(?:creatinine)\s*[:\-]?\s*([\d.]+)",
    "Hemoglobin":      r"(?:hemoglobin|hb|haemoglobin)\s*[:\-]?\s*([\d.]+)",
    "Cholesterol":     r"(?:cholesterol|ldl|hdl|triglycerides?)\s*[:\-]?\s*([\d.]+)",
    "Heart Rate":      r"(?:pulse|heart\s*rate|hr)\s*[:\-]?\s*([\d]+)",
    "O2 Saturation":   r"(?:o2\s*sat|spo2|oxygen\s*saturation)\s*[:\-]?\s*([\d.]+)",
    "Weight":          r"(?:weight)\s*[:\-]?\s*([\d.]+)",
    "BMI":             r"(?:bmi)\s*[:\-]?\s*([\d.]+)",
}

_DATE_BLOCK_RE = re.compile(
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
    r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}'
    r'|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
    r'|\b\d{4}-\d{2}-\d{2}\b',
    re.IGNORECASE,
)


def _parse_date_safe(date_str: str):
    try:
        import dateparser
        return dateparser.parse(date_str, settings={"RETURN_AS_TIMEZONE_AWARE": False})
    except Exception:
        return None


def extract_lab_trends(raw_text: str) -> dict:
    import re as _re
    trends: dict = {k: [] for k in _TREND_PATTERNS}
    text_lower = raw_text.lower()
    window_size = 800
    step = 400
    seen: dict = {k: set() for k in _TREND_PATTERNS}

    for start in range(0, len(text_lower), step):
        chunk = text_lower[start: start + window_size]
        date_match = _DATE_BLOCK_RE.search(chunk)
        if not date_match:
            continue
        date_obj = _parse_date_safe(date_match.group(0))
        if date_obj is None:
            continue

        for label, pattern in _TREND_PATTERNS.items():
            m = _re.search(pattern, chunk)
            if m:
                try:
                    val = float(m.group(1).strip())
                except ValueError:
                    continue
                key = (date_obj, val)
                if key not in seen[label]:
                    seen[label].add(key)
                    trends[label].append((date_obj, val))

    for label in trends:
        trends[label].sort(key=lambda x: x[0])

    return {k: v for k, v in trends.items() if v}


def render_clinical_trends(raw_text: str):
    import pandas as pd
    st.markdown("### 📈 Clinical Trends")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Lab Value Trends Over Time</div>', unsafe_allow_html=True)

    trends = extract_lab_trends(raw_text)

    if not trends:
        st.caption("⚠️ No dated lab readings found for trend analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    plotted = 0
    skipped_labels = []

    for label, series in trends.items():
        if len(series) < 2:
            skipped_labels.append((label, series[0][1] if series else None))
            continue

        df = pd.DataFrame(series, columns=["Date", label])
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = df.sort_values("Date").set_index("Date")

        st.markdown(
            f'<div style="margin:1rem 0 0.3rem;">'
            f'<span style="color:#38bdf8;font-weight:600;font-size:0.95rem;">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.line_chart(df, use_container_width=True, height=200)
        plotted += 1

    if skipped_labels:
        st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
        for label, val in skipped_labels:
            val_str = f" — latest reading: **{val}**" if val is not None else ""
            st.caption(f"🔹 *{label}*{val_str} — Not enough data points for trend analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    if plotted == 0 and skipped_labels:
        st.info(
            "📋 Lab values were found but all have only a single recorded reading. "
            "Upload a document with multiple dated lab results to see trends."
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_chat():
    import rag
    st.markdown("### 💬 Ask the Document")

    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label">You</div>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            answer     = content.get("answer",     "")
            confidence = content.get("confidence", "medium")
            sources    = content.get("sources",    [])
            conf_label = {"high": "High", "medium": "Medium", "low": "Low", "none": "N/A"}.get(confidence, "Medium")

            st.markdown(f"""
            <div class="chat-ai">
                <div class="chat-label">
                    MedIntel AI &nbsp;·&nbsp;
                    <span class="conf-{confidence}">{conf_label} confidence</span>
                </div>
                {answer}
            </div>
            """, unsafe_allow_html=True)

            if sources:
                with st.expander("📄 Source Passages", expanded=False):
                    for src in sources:
                        st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.caption("💡 Suggested questions:")
        suggestions = [
            "What conditions does the patient have?",
            "What medications are prescribed?",
            "What are the latest lab results?",
            "Is there any hospitalisation history?",
        ]
        c1, c2 = st.columns(2)
        for i, q in enumerate(suggestions):
            col = c1 if i % 2 == 0 else c2
            with col:
                if st.button(q, key=f"sug_{i}", use_container_width=True):
                    _submit_question(q)

    question = st.chat_input("Ask anything about this document…")
    if question:
        _submit_question(question)


def _submit_question(question: str):
    import rag
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("🤔 Searching document…"):
        result = rag.answer_question(question, st.session_state.raw_text)
    st.session_state.chat_history.append({"role": "assistant", "content": result})
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedIntel AI")
    st.caption("Clinical Document Intelligence")
    st.divider()
    
    # 🚨 PRIVACY TOGGLE 🚨
    st.markdown("### 🔒 Privacy Controls")
    st.toggle(
        "Anonymize Patient Data (GDPR)", 
        key="anonymize",
        help="Masks PII like names, IDs, and contact info before the AI processes the document."
    )
    st.divider()

    if st.session_state.processed:
        st.markdown("### 📊 Document Stats")
        st.caption(f"📄 {st.session_state.doc_name}")
        ents = st.session_state.entities
        st.metric("Conditions",      len(ents.get("diseases",    [])))
        st.metric("Medications",     len(ents.get("medications", [])))
        st.metric("Lab Values",      len(ents.get("lab_values",  [])))
        st.metric("Timeline Events", len(st.session_state.timeline))
        st.metric("Risk Alerts",     len(st.session_state.risks))
        st.metric("Pages Analysed",  st.session_state.page_count)

        st.divider()
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session()
            st.rerun()
    else:
        st.info("📂 Upload a PDF on the main page to get started.", icon="ℹ️")

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** MedIntel AI is for educational/research use only. "
        "Not a substitute for professional medical advice."
    )


# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 MedIntel AI</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#64748b;margin-top:-0.5rem;font-size:0.95rem;">'
    'Clinical Document Intelligence – Extract · Analyse · Understand'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()


# ── Route ─────────────────────────────────────────────────────────────────────
if not st.session_state.processed:
    show_landing()
else:
    # ── Metric row ────────────────────────────────────────────────────────────
    entities = st.session_state.entities
    timeline = st.session_state.timeline
    risks    = st.session_state.risks

    cols = st.columns(5)
    for col, (val, label, icon) in zip(cols, [
        (len(entities.get("diseases",    [])), "Conditions",      "🦠"),
        (len(entities.get("medications", [])), "Medications",     "💊"),
        (len(entities.get("lab_values",  [])), "Lab Values",      "🔬"),
        (len(timeline),                        "Timeline Events", "📅"),
        (st.session_state.page_count,          "Pages Analysed",  "📄"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.4rem;">{icon}</div>
                <div class="metric-number">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main layout ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        render_patient_summary(entities)
        render_timeline(timeline)

    with right_col:
        render_risk_alerts(risks)
        st.markdown("<br>", unsafe_allow_html=True)
        render_chat()

    # ── Clinical Trends (full width below) ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    render_clinical_trends(st.session_state.raw_text)