import os
import json
import shutil
import streamlit as st
from texts import texts
from rag_compare import (
    load_and_chunk,
    build_or_load_vectorstore,
    compare,
    health_check,
    create_embedding_client,
    create_chat_client
)
from fpdf import FPDF

# 1. Lade Konfiguration
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

st.set_page_config(
    page_title=config["app_name"],
    page_icon=config["logo_path"],
    layout="wide"
)

try:
    health_check()
    st.sidebar.success("✅ Azure OpenAI-Verbindung erfolgreich!")
except Exception:
    st.sidebar.error("❌ Azure OpenAI Verbindung fehlgeschlagen")
    st.stop()

embedding_client = create_embedding_client()
llm_client = create_chat_client()

if "language" not in st.session_state:
    st.session_state.language = config.get("default_language", "de")

st.sidebar.image(config["logo_path"], use_container_width=True)
st.sidebar.title(config["app_name"])

language = st.sidebar.selectbox(
    "Sprache / Language", options=["de", "en"], index=0 if st.session_state.language == "de" else 1
)
st.session_state.language = language
current_texts = texts[language]

st.title(current_texts["welcome_title"])
st.write(current_texts["welcome_message"])

available_docs = {
    "DSGVO": "data/dsgvo.pdf",
    "GDPR": "data/gdpr.pdf",
    "NIS2": "data/nis2.pdf",
    "NIS2 (en)": "data/nis2_en.pdf",
    "Data Act": "data/data_act.pdf",
    "Data Act (en)": "data/data_act_en.pdf"
}


col1, col2 = st.columns(2)
with col1:
    doc1_name = st.selectbox("Vergleiche oder nutze folgende Regularik:", list(available_docs.keys()), index=0)
with col2:
    doc2_name = st.selectbox("mit Gesetz/Regularik (oder leer lassen für einzelnes Gesetz/Regularik):", ["---"] + list(available_docs.keys()), index=0)

user_role = st.radio("Für wen soll die Antwort formuliert werden?", ["Jurist", "Anwender", "Techniker/Entwickler"], horizontal=True)

query = st.text_input(current_texts["query_input"])
rebuild = st.sidebar.button("🔄 Embeddings neu erstellen")
status = st.sidebar.empty()

# PDF Export
def generate_pdf(query, vergleich, filename="vergleich.pdf"):
    pdf = FPDF()
    pdf.add_page()
    logo_path = "data/health365_logo.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=60)
    font_path = "fonts/DejaVuSans.ttf"
    if os.path.exists(font_path):
        try:
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.add_font("DejaVu", "B", font_path, uni=True)
            font_family = "DejaVu"
        except RuntimeError:
            font_family = "Arial"
    else:
        font_family = "Arial"

    try:
        pdf.set_font(font_family, '', 14)
    except RuntimeError:
        pdf.set_font("Arial", '', 14)

    pdf.ln(30)
    pdf.set_font(font_family, 'B', 16)
    title = "Comparison Analysis" if language == "en" else "Vergleichsanalyse"
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)

    pdf.set_font(font_family, 'B', 12)
    pdf.cell(0, 10, "Question:" if language == "en" else "Fragestellung:", ln=True)
    pdf.set_font(font_family, '', 12)
    pdf.multi_cell(0, 10, query)

    pdf.ln(10)
    pdf.set_font(font_family, 'B', 12)
    pdf.cell(0, 10, "Result:" if language == "en" else "Ergebnis:", ln=True)
    pdf.set_font(font_family, '', 12)
    for line in vergleich.split("\n"):
        pdf.multi_cell(0, 10, line)

    return pdf.output(dest='S').encode('latin-1', errors='ignore')


# Verarbeitung
if query:
    try:
        with st.spinner(current_texts["processing"]):
            chunks1 = load_and_chunk(available_docs[doc1_name])
            db1 = build_or_load_vectorstore(chunks1, doc1_name, embedding_client)

            if rebuild:
                shutil.rmtree(f"vector_store/faiss_{doc1_name.lower()}", ignore_errors=True)
                status.info("🔄 Embeddings werden neu erstellt ...")

            # Rolle abhängig von Sprache
            role_instruction = {
                "Jurist": {
                    "de": "Fasse die Inhalte rechtlich strukturiert zusammen. Gib konkrete Fundstellen (Artikel, Absatz, Recital) aber immer den Atikel an und verwende juristische Sprache.",
                    "en": "Summarize the content in a legally structured format. Provide exact references (Article, Paragraph, Recital) and use legal terminology."
                },
                "Anwender": {
                    "de": "Erkläre die Inhalte möglichst einfach, nutzerfreundlich und anwendungsbezogen so das ich diese in eine Richtlinie oder Dokument übersetzen kann.",
                    "en": "Explain the content as simply, user-friendly, and application-oriented as possible so that I can translate it into a guideline or document."
                },
                "Techniker/Entwickler": {
                    "de": "Fasse die Inhalte so zusammen, dass ein technischer Verantwortlicher sie als umsetzbare Maßnahmen interpretieren kann. Berücksichtige mögliche Standards, Verfahren, Protokolle und Tools (z. B. ISO-Normen, BSI, OWASP, Logging, Monitoring, Schnittstellen etc.).",
                    "en": "Summarize the content so that a technical stakeholder can interpret it as actionable steps. Consider relevant standards, procedures, protocols and tools (e.g. ISO, BSI, OWASP, logging, monitoring, APIs)."
                }
            }[user_role][language]

            if doc2_name != "---":
                chunks2 = load_and_chunk(available_docs[doc2_name])
                db2 = build_or_load_vectorstore(chunks2, doc2_name, embedding_client)
                vergleich = compare(query, db1, db2, llm_client, doc1_name, doc2_name)
            else:
                top_chunks = db1.similarity_search(query, k=3)
                context = "\n\n".join([chunk.page_content for chunk in top_chunks])
                if language == "en":
                    prompt_intro = f"You are an AI assistant. Use the following legal excerpt from {doc1_name} to answer the question. {role_instruction}"
                    prompt_text = f"{prompt_intro}\n\nExcerpt:\n{context}\n\nQuestion: {query}"
                else:
                    prompt_intro = f"Du bist ein KI-Assistent. Nutze folgenden Gesetzestext aus {doc1_name}, um die Frage zu beantworten. {role_instruction}"
                    prompt_text = f"{prompt_intro}\n\nTextauszug:\n{context}\n\nFrage: {query}"
                vergleich = llm_client.invoke(prompt_text).content

            st.markdown(f"## {doc1_name} – relevante Textstellen")
            for i, chunk in enumerate(db1.similarity_search(query, k=3), 1):
                title = chunk.metadata.get("title_guess", chunk.page_content[:60])
                page = chunk.metadata.get("page", "N/A")
                with st.expander(f"{doc1_name} Chunk #{i} (Seite {page}) – {title}"):
                    st.write(chunk.page_content)

            if doc2_name != "---":
                st.markdown(f"## {doc2_name} – relevante Textstellen")
                for i, chunk in enumerate(db2.similarity_search(query, k=3), 1):
                    title = chunk.metadata.get("title_guess", chunk.page_content[:60])
                    page = chunk.metadata.get("page", "N/A")
                    with st.expander(f"{doc2_name} Chunk #{i} (Seite {page}) – {title}"):
                        st.write(chunk.page_content)

            st.markdown("## Ergebnis")
            st.write(vergleich)

            st.download_button("📄 Ergebnis als PDF", generate_pdf(query, vergleich), file_name="vergleich.pdf", mime="application/pdf")
            st.download_button("📝 Ergebnis als Markdown", vergleich.encode('utf-8'), file_name="vergleich.md", mime="text/markdown")

    except Exception:
        st.error("❌ Es ist ein Fehler aufgetreten.")
