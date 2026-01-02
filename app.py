import streamlit as st
import os
import re
import io
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq

# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not GROQ_API_KEY or not OPENROUTER_API_KEY:
    st.error("❌ Missing API keys. Check your .env file.")
    st.stop()

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Public Policy Insight & Impact Analyzer (PPIIA)",
    layout="wide"
)

st.title("🏛️ Public Policy Insight & Impact Analyzer (PPIIA)")

# =========================
# SESSION STATE
# =========================
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "model_used" not in st.session_state:
    st.session_state.model_used = None

# =========================
# BILL VALIDATION
# =========================
BILL_KEYWORDS = [
    "bill", "act", "parliament", "lok sabha",
    "rajya sabha", "statement of objects",
    "introduced", "passed", "minister"
]

def is_valid_bill(text: str) -> bool:
    text = text.lower()
    hits = sum(1 for k in BILL_KEYWORDS if k in text)
    return len(text) > 500 and hits >= 4

# =========================
# TEXT EXTRACTION
# =========================
def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_pdf_from_bytes(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,text/html"
    }

    r = requests.get(url, headers=headers, timeout=20)
    content_type = r.headers.get("Content-Type", "").lower()

    # ---- PDF URL ----
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            return extract_pdf_from_bytes(r.content)
        except Exception:
            raise ValueError(
                "PDF detected but could not be parsed. Please download and upload the PDF."
            )

    # ---- HTML page (Bill page) ----
    if "text/html" in content_type:
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator="\n")

    raise ValueError("Unsupported URL format")

# =========================
# LLM CALLS
# =========================
def call_groq(prompt):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=3500
    )
    return llm.invoke(prompt).content

def call_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3500
    }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    return res.json()["choices"][0]["message"]["content"]

# =========================
# AUTO SWITCH
# =========================
def ask_llm(prompt):
    try:
        return call_groq(prompt), "Groq"
    except Exception:
        st.warning("⚠️ Groq failed → switching to OpenRouter")
        return call_openrouter(prompt), "OpenRouter"

# =========================
# USER INPUT (PDF + URL ONLY)
# =========================
input_type = st.radio(
    "Select Input Type",
    ["PDF Upload", "URL"]
)

bill_text = ""

if input_type == "PDF Upload":
    file = st.file_uploader("Upload Government Bill PDF", type=["pdf"])
    if file:
        bill_text = extract_pdf(file)

elif input_type == "URL":
    url = st.text_input("Enter Government Bill URL (PDF or Bill page)")
    if url:
        try:
            bill_text = extract_from_url(url)
        except Exception as e:
            st.error(f"❌ {e}")
            st.info("👉 Recommendation: Download the PDF and upload it directly.")
            st.stop()

# =========================
# ANALYSIS GENERATION
# =========================
if bill_text:
    if not is_valid_bill(bill_text):
        st.error("❌ This does not appear to be a valid government bill.")
        st.stop()

    st.success("✅ Valid government bill detected")

    with st.expander("📜 Preview Extracted Text"):
        st.text_area("Bill Text", bill_text[:3000], height=250)

    if st.button("🔍 GENERATE ANALYSIS"):
        with st.spinner("🤖 Analyzing bill..."):

            PROMPT = f"""
You are a public policy analyst.

Analyze the following government bill and generate outputs using EXACT section headers.

SECTOR:
- One main sector only

SUMMARY:
- 10 to 20 easy bullet points
- Explain for normal citizens

IMPACT:
Citizens:
- Bullet points
Businesses:
- Bullet points
Government:
- Bullet points

POSITIVES:
- Bullet points

RISKS:
- Bullet points

BENEFICIARIES:
- Bullet points

RULES:
- Use simple English
- Do not hallucinate
- If info missing, say so

BILL TEXT:
{bill_text[:12000]}
"""
            analysis, model = ask_llm(PROMPT)
            st.session_state.analysis = analysis
            st.session_state.model_used = model

# =========================
# DISPLAY + CHAT
# =========================
if st.session_state.analysis:

    st.success(f"✅ Analysis generated using {st.session_state.model_used}")

    tabs = st.tabs(["📊 Sector", "📝 Summary", "📈 Impact"])

    with tabs[0]:
        st.write(
            re.search(r"SECTOR:(.*?)(SUMMARY:)", st.session_state.analysis, re.S)
            .group(1)
        )

    with tabs[1]:
        st.write(
            re.search(r"SUMMARY:(.*?)(IMPACT:)", st.session_state.analysis, re.S)
            .group(1)
        )

    with tabs[2]:
        st.write(
            re.search(r"IMPACT:(.*)", st.session_state.analysis, re.S)
            .group(1)
        )

    st.markdown("---")
    st.subheader("💬 Ask AI about this Bill")

    user_q = st.text_input("Your question", key="user_question")

    if user_q:
        chat_prompt = f"""
Answer ONLY from the analysis below.
If answer not present, say so.

ANALYSIS:
{st.session_state.analysis}

QUESTION:
{user_q}
"""
        with st.spinner("🤖 Thinking..."):
            reply, model = ask_llm(chat_prompt)

        st.success(f"Answered using {model}")
        st.write(reply)
