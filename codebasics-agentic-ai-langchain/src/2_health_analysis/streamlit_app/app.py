from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

load_dotenv()

SAMPLE_REPORT_PATH = Path(__file__).resolve().parent.parent / "blood_work.txt"

EXTRACTION_PROMPT = """
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_work_report}
"""

DIET_PROMPT = """
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 3-4 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extracted_values}
"""


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemma-4-31b-it")


def read_uploaded_report(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8")


def extract_values(llm, blood_work_report: str) -> str:
    response = llm.invoke(EXTRACTION_PROMPT.format(blood_work_report=blood_work_report))
    return response.text


def generate_diet_plan(llm, extracted_values: str) -> str:
    response = llm.invoke(DIET_PROMPT.format(extracted_values=extracted_values))
    return response.text


st.set_page_config(page_title="Blood Report Analyser", page_icon="🩸", layout="wide")
st.title("🩸 Blood Report Analyser")
st.caption(
    "Upload a blood work report to get extracted test values, a simple health summary "
    "and an Indian diet plan."
)

with st.sidebar:
    st.header("Report input")
    source = st.radio("Choose a source", ["Upload file", "Paste text", "Use sample report"])

    report_text = ""
    if source == "Upload file":
        uploaded_file = st.file_uploader("Blood report", type=["txt", "pdf"])
        if uploaded_file:
            report_text = read_uploaded_report(uploaded_file)
    elif source == "Paste text":
        report_text = st.text_area("Paste the report here", height=300)
    else:
        report_text = SAMPLE_REPORT_PATH.read_text()

    analyse = st.button("Analyse report", type="primary", disabled=not report_text.strip())

if report_text.strip():
    with st.expander("View report", expanded=not analyse):
        st.text(report_text)
else:
    st.info("Provide a blood report from the sidebar to get started.")

if analyse:
    llm = get_llm()
    try:
        with st.status("Analysing report...", expanded=True) as status:
            st.write("Stage 1: Extracting test values...")
            extracted_values = extract_values(llm, report_text)
            st.write("Stage 2: Generating health summary and diet plan...")
            diet_plan = generate_diet_plan(llm, extracted_values)
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

    st.session_state["extracted_values"] = extracted_values
    st.session_state["diet_plan"] = diet_plan

if "diet_plan" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Extracted Values")
        st.markdown(st.session_state["extracted_values"])
    with col2:
        st.subheader("🥗 Health Summary & Diet Plan")
        st.markdown(st.session_state["diet_plan"])

    st.download_button(
        "Download analysis",
        data=(
            "=== EXTRACTED VALUES ===\n"
            f"{st.session_state['extracted_values']}\n\n"
            "=== HEALTH SUMMARY & DIET PLAN ===\n"
            f"{st.session_state['diet_plan']}\n"
        ),
        file_name="blood_report_analysis.txt",
    )

st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical advice.")
