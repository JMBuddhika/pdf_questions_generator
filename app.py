"""
PDF Question Generator — Clean Refactor
======================================
Generate MCQ, Active Recall, or Closure questions from PDFs using the Groq API.

Setup:
  1) Get an API key: https://console.groq.com/keys
  2) Create .env: GROQ_API_KEY=your_key_here
  3) pip install groq python-dotenv streamlit pypdf reportlab
  4) streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import re
from enum import Enum
from typing import Dict, List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from reportlab.lib import enums
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

BASE_SYSTEM_PROMPT = (
    "You are an expert assessment designer. Create high-quality questions "
    "strictly grounded in the provided source text. Do not invent facts. "
    'Respond ONLY with valid JSON. Always follow the "TASK_TYPE" provided.'
)

# Rough guard to avoid absurdly large prompts; truncate with a note when exceeded.
MAX_INPUT_CHARS = 80_000

REFLECTIVE_STARTERS = re.compile(
    r"^\s*(how|why|in what way|discuss|explain|reflect|evaluate|compare|"
    r"contrast|synthesize|to what extent|what.*implication)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "with",
    "by", "at", "as", "is", "are", "was", "were", "this", "that", "these",
    "those", "it", "its", "from", "into", "over", "under", "about",
    "between", "among", "be", "been", "being", "which", "who", "whom",
    "whose", "their", "there", "then", "than", "such"
}


class QuestionType(str, Enum):
    MCQ = "MCQ"
    ACTIVE_RECALL = "Active Recall"
    CLOSURE = "Closure"


# -----------------------------------------------------------------------------
# PDF helpers
# -----------------------------------------------------------------------------

def extract_text_from_pdf(file) -> Tuple[List[str], str]:
    """Return page-wise text and a combined full-text string."""
    reader = PdfReader(file)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return pages, "\n".join(pages)


# -----------------------------------------------------------------------------
# Groq client helpers
# -----------------------------------------------------------------------------

def require_api_key() -> str:
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not found. Add it in Streamlit Secrets (cloud) or .env (local).")
        st.stop()
    return key


def call_groq_json(messages: List[Dict[str, str]], api_key: str, temperature: float) -> dict:
    """Call Groq Chat Completions, requesting a JSON object and parsing it robustly."""
    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_json_response(content)
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}") from e


def parse_json_response(text: str) -> dict:
    """Parse JSON with recovery (strip trailing commas, grab inner JSON block)."""
    text = text.strip()

    def _json_try(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) Direct
    direct = _json_try(text)
    if direct is not None:
        return direct

    # 2) Inner-most braces
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])  # trailing commas
        inner = _json_try(candidate)
        if inner is not None:
            return inner

    raise ValueError("Failed to parse JSON from model output.")


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

def build_mcq_prompt(text: str, count: int, difficulty: str) -> List[Dict[str, str]]:
    system = f'{BASE_SYSTEM_PROMPT} TASK_TYPE="{QuestionType.MCQ.value}".'
    user = f"""
TASK_TYPE: {QuestionType.MCQ.value}

Create exactly {count} multiple-choice questions from the text below.

Requirements:
- Test key concepts from the text
- Provide exactly 4 options per question
- Write only the option text (no A/B/C/D labels)
- Set correct_answer to: "A", "B", "C", or "D"
- Include a brief explanation (1-2 sentences)
- Difficulty level: {difficulty}

JSON Response Format:
{{
  "questions": [
    {{
      "question": "What is the main concept?",
      "options": ["First option", "Second option", "Third option", "Fourth option"],
      "correct_answer": "A",
      "explanation": "Brief explanation here"
    }}
  ]
}}

Source Text:
{_maybe_truncate(text)}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_active_prompt(text: str, count: int, difficulty: str) -> List[Dict[str, str]]:
    system = f'{BASE_SYSTEM_PROMPT} TASK_TYPE="{QuestionType.ACTIVE_RECALL.value}".'
    user = f"""
TASK_TYPE: {QuestionType.ACTIVE_RECALL.value}

Create exactly {count} open-ended reflective questions from the text below.
Each question MUST include a concise model answer.

Requirements:
- Encourage summarizing, connecting ideas, or reflecting on implications
- Start with: 'How', 'Why', 'Explain', 'Discuss', 'Evaluate', or 'To what extent'
- NO blanks (____) or multiple choice options
- Provide an "answer" field with 2-3 sentences OR 2-4 bullet points
- Difficulty level: {difficulty}

JSON Response Format:
{{
  "questions": [
    {{
      "question": "How do the main concepts in this text relate to each other?",
      "answer": "They relate through... (2-3 sentences or 2-4 bullet points)"
    }}
  ]
}}

Source Text:
{_maybe_truncate(text)}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_closure_prompt(text: str, count: int, difficulty: str) -> List[Dict[str, str]]:
    system = f'{BASE_SYSTEM_PROMPT} TASK_TYPE="{QuestionType.CLOSURE.value}".'
    user = f"""
TASK_TYPE: {QuestionType.CLOSURE.value}

Create exactly {count} fill-in-the-blank (cloze) questions from the text below.

Requirements:
- Include exactly ONE ____ blank per question for a key term/concept
- Use declarative sentences (avoid 'how/why' questions)
- Provide a short, precise "answer" (the missing term/phrase)
- Difficulty level: {difficulty}

JSON Response Format:
{{
  "questions": [
    {{
      "question": "The process of ____ is essential for learning.",
      "answer": "repetition"
    }}
  ]
}}

Source Text:
{_maybe_truncate(text)}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


PROMPT_BUILDERS = {
    QuestionType.MCQ.value: build_mcq_prompt,
    QuestionType.ACTIVE_RECALL.value: build_active_prompt,
    QuestionType.CLOSURE.value: build_closure_prompt,
}


def _maybe_truncate(text: str) -> str:
    if len(text) <= MAX_INPUT_CHARS:
        return text
    st.info(
        f"Long selection detected. Using the first {MAX_INPUT_CHARS:,} characters to stay within model limits."
    )
    return text[:MAX_INPUT_CHARS]


# -----------------------------------------------------------------------------
# Post-generation coercion & validation
# -----------------------------------------------------------------------------

def looks_like_reflection(q: str) -> bool:
    return bool(REFLECTIVE_STARTERS.search(q or "")) and "____" not in (q or "")


def looks_like_cloze(q: str) -> bool:
    return (q or "").count("____") == 1


def pick_keyword_for_blank(sentence: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{3,}", sentence or "")
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9]{4,}", sentence or "")
    return max(tokens, key=len) if tokens else ""


def convert_to_cloze(item: Dict) -> Dict:
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()

    if looks_like_cloze(q) and a:
        return {"question": q, "answer": a}

    if a and a.lower() in q.lower():
        new_q = re.compile(re.escape(a), re.IGNORECASE).sub("____", q, count=1)
        return {"question": new_q.strip().rstrip("?."),
                "answer": a}

    keyword = pick_keyword_for_blank(q)
    if not keyword:
        stem = q.rstrip(" ?.!").strip()
        if not stem:
            return {}
        keyword = pick_keyword_for_blank(stem) or "concept"
        new_q = stem.replace(keyword, "____", 1)
        if "____" not in new_q:
            new_q = f"____ is related to {stem}"
        return {"question": new_q, "answer": keyword}

    new_q = re.sub(re.escape(keyword), "____", q, count=1, flags=re.IGNORECASE)
    return {"question": new_q.rstrip(" ?.!"), "answer": keyword}


def convert_to_reflection(item: Dict) -> Dict:
    q = (item.get("question") or "").strip().replace("____", "this concept")
    q = re.sub(r"\s+", " ", q).strip()

    if not looks_like_reflection(q):
        stem = q.rstrip("?.!").strip()
        if not stem:
            return {}
        q = f"Explain the significance of {stem}." if len(stem.split()) <= 8 \
            else f"How does {stem} relate to the broader themes discussed?"

    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    a = (item.get("answer") or "").strip()
    return {"question": q, "answer": a} if a else {"question": q}


def coerce_to_type(items: List[Dict], qtype: QuestionType) -> List[Dict]:
    out: List[Dict] = []
    for it in items:
        try:
            if qtype is QuestionType.CLOSURE:
                conv = convert_to_cloze(it)
                if conv and looks_like_cloze(conv.get("question", "")) and conv.get("answer"):
                    out.append(conv)
            elif qtype is QuestionType.ACTIVE_RECALL:
                conv = convert_to_reflection(it)
                if conv and "____" not in conv.get("question", "") and conv.get("answer"):
                    out.append(conv)
            else:
                out.append(it)  # MCQ already structurally validated separately
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# Generation orchestration
# -----------------------------------------------------------------------------

def generate_questions(
    text: str,
    count: int,
    qtype: QuestionType,
    difficulty: str,
    temperature: float,
    api_key: str,
) -> List[Dict]:
    """End-to-end generation + validation for the selected type."""
    builder = PROMPT_BUILDERS[qtype.value]
    messages = builder(text, count, difficulty)

    data = call_groq_json(messages, api_key, temperature)
    candidates = data.get("questions", [])
    if not isinstance(candidates, list):
        return []

    # Base validation
    validated: List[Dict] = []
    for item in candidates:
        if qtype is QuestionType.MCQ:
            if not _valid_mcq(item):
                continue
            item["options"] = _clean_options(item.get("options", []))
            item["correct_answer"] = str(item.get("correct_answer", "")).strip().upper()[:1]
            validated.append(item)
        else:
            if item.get("question"):
                validated.append(item)

    # Type-specific coercion
    if qtype in (QuestionType.ACTIVE_RECALL, QuestionType.CLOSURE):
        validated = coerce_to_type(validated, qtype)

    # Ensure Active Recall has answers
    if qtype is QuestionType.ACTIVE_RECALL:
        validated = [q for q in validated if q.get("answer")]

    return validated[:count]


def _valid_mcq(item: Dict) -> bool:
    return all([item.get("question"), item.get("options"), item.get("correct_answer")]) \
           and len(item.get("options", [])) == 4


def _clean_options(options: List[str]) -> List[str]:
    """Strip any letter prefixes (A., B), keep only first four."""
    pat = re.compile(r"^\s*[A-Da-d][\.\)]\s*")
    cleaned = [pat.sub("", str(opt)).strip() for opt in (options or []) if str(opt).strip()]
    return cleaned[:4]


# -----------------------------------------------------------------------------
# PDF creation
# -----------------------------------------------------------------------------

def build_pdf(
    title: str,
    qtype: QuestionType,
    items: List[Dict],
    include_answers: bool
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    styles["Title"].alignment = enums.TA_CENTER
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    if qtype is QuestionType.MCQ:
        _pdf_mcq(story, items, styles, include_answers)
    elif qtype is QuestionType.CLOSURE:
        _pdf_cloze(story, items, styles, include_answers)
    else:
        _pdf_active(story, items, styles, include_answers)

    doc.build(story)
    return buf.getvalue()


def _pdf_mcq(story, items, styles, include_answers):
    for idx, q in enumerate(items, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {q.get('question', '')}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        bullets = [ListItem(Paragraph(opt, styles["BodyText"]), leftIndent=12)
                   for opt in _clean_options(q.get("options", []))]
        story.append(ListFlowable(bullets, bulletType="A", start=1, leftIndent=12))

        if include_answers:
            story.append(Spacer(1, 6))
            ans = str(q.get("correct_answer", "")).strip().upper()[:1]
            story.append(Paragraph(f"<b>Answer:</b> {ans}", styles["BodyText"]))
            expl = q.get("explanation", "")
            if expl:
                story.append(Paragraph(f"<i>{expl}</i>", styles["BodyText"]))

        story.append(Spacer(1, 12))


def _pdf_cloze(story, items, styles, include_answers):
    for idx, q in enumerate(items, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {q.get('question', '')}", styles["BodyText"]))
        if include_answers:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Answer:</b> {q.get('answer', '')}", styles["BodyText"]))
        story.append(Spacer(1, 12))


def _pdf_active(story, items, styles, include_answers):
    for idx, q in enumerate(items, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {q.get('question', '')}", styles["BodyText"]))
        if include_answers:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Answer:</b> {q.get('answer', '')}", styles["BodyText"]))
        story.append(Spacer(1, 12))


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_extract(file_bytes: bytes):
    return extract_text_from_pdf(io.BytesIO(file_bytes))


def sidebar_config() -> float:
    with st.sidebar:
        st.header("⚙️ Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="Lower = more focused, Higher = more creative",
        )
        return temperature


def page_selector(pages: List[str]) -> Optional[str]:
    with st.expander("🔍 Select specific pages (optional)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            start = st.number_input("Start page", 1, len(pages), 1)
        with col_b:
            end = st.number_input("End page", 1, len(pages), len(pages))

        if start > end:
            st.error("Start page must be ≤ end page.")
            return None

        chosen = "\n".join(pages[start - 1:end])
        if start != 1 or end != len(pages):
            st.info(f"Using pages {start} to {end}.")
        return chosen


def question_controls():
    st.subheader("Question Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        qtype = st.selectbox(
            "Question Type",
            [QuestionType.MCQ.value, QuestionType.ACTIVE_RECALL.value, QuestionType.CLOSURE.value],
            index=0,
        )
    with col2:
        count = st.number_input("Number of questions", min_value=1, max_value=50, value=10)
    with col3:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

    include_answers = st.checkbox("Include answers in PDF", value=True)
    return QuestionType(qtype), int(count), difficulty, include_answers


def preview_mcq(items: List[Dict], include_answers: bool):
    labels = ["A", "B", "C", "D"]
    for idx, q in enumerate(items, 1):
        with st.expander(f"Question {idx}"):
            st.markdown(f"**{q.get('question', '')}**")
            st.write("")
            for label, opt in zip(labels, _clean_options(q.get("options", []))):
                st.write(f"{label}. {opt}")
            if include_answers:
                st.write("")
                ans = str(q.get("correct_answer", "")).strip().upper()[:1]
                st.success(f"**Answer:** {ans}")
                if q.get("explanation"):
                    st.info(f"**Explanation:** {q.get('explanation')}")


def preview_short(items: List[Dict], include_answers: bool):
    for idx, q in enumerate(items, 1):
        with st.expander(f"Question {idx}"):
            st.markdown(f"**{q.get('question', '')}**")
            if include_answers:
                st.write("")
                st.success(f"**Answer:** {q.get('answer', '')}")


def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Question Generator", page_icon="📝", layout="wide")
    st.title("📝 PDF Question Generator")
    st.caption("Generate study questions from PDFs with Groq")

    try:
        api_key = require_api_key()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    temperature = sidebar_config()

    uploaded = st.file_uploader("📄 Upload PDF", type=["pdf"])
    if not uploaded:
        st.info("👆 Upload a PDF to get started.")
        return

    with st.spinner("Extracting text from PDF..."):
        pages, full_text = cached_extract(uploaded.getvalue())

    st.success(f"✅ Extracted {len(pages)} pages ({len(full_text):,} characters).")

    selected_text = page_selector(pages)
    if selected_text is None:
        return
    if not selected_text.strip():
        st.warning("⚠️ No text found on selected pages.")
        return

    qtype, count, difficulty, include_answers = question_controls()

    if st.button("🚀 Generate Questions", type="primary"):
        with st.spinner(f"Generating {count} {qtype.value} questions..."):
            try:
                items = generate_questions(
                    text=selected_text,
                    count=count,
                    qtype=qtype,
                    difficulty=difficulty,
                    temperature=temperature,
                    api_key=api_key,
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Check your API key, connection, or try fewer pages.")
                return

        if not items:
            st.error("❌ No questions generated. Try different settings or a different page range.")
            return

        st.success(f"✨ Generated {len(items)} questions!")
        st.subheader("📋 Preview")

        if qtype is QuestionType.MCQ:
            preview_mcq(items, include_answers)
        else:
            preview_short(items, include_answers)

        st.subheader("📥 Download")
        title = f"{qtype.value} Questions"
        pdf_bytes = build_pdf(title, qtype, items, include_answers)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{qtype.value.replace(' ', '_').lower()}_questions.pdf",
            mime="application/pdf",
            type="primary",
        )


if __name__ == "__main__":
    main()
