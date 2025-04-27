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

# 2. Setze Streamlit Page Config
st.set_page_config(
    page_title=config["app_name"],
    page_icon=config["logo_path"],
    layout="wide"
)

# 3. Health Check für Azure OpenAI
try:
    health_check()
    st.sidebar.success("✅ Azure OpenAI-Verbindung erfolgreich!")
except Exception as e:
    st.sidebar.error(f"❌ Azure OpenAI Verbindung fehlgeschlagen: {e}")
    st.stop()

# 4. Erzeuge Embedding- und Chat-Clients
embedding_client = create_embedding_client()
llm_client = create_chat_client()

# Sprache laden oder Standard setzen
if "language" not in st.session_state:
    st.session_state.language = config.get("default_language", "de")

# Sidebar
st.sidebar.image(config["logo_path"], use_container_width=True)
st.sidebar.title(config["app_name"])

language = st.sidebar.selectbox(
    "Sprache / Language", options=["de", "en"], index=0 if st.session_state.language == "de" else 1
)
st.session_state.language = language

# Textfunktion aus aktuellem Sprachzustand
current_texts = texts[language]

# Hauptinhalt
st.title(current_texts["welcome_title"])
st.write(current_texts["welcome_message"])

query = st.text_input(current_texts["query_input"])

# Button zum Neu-Erstellen der Embeddings
rebuild_embeddings = st.sidebar.button("🔄 Embeddings neu erstellen")

# Statusanzeige in Sidebar
status_text = st.sidebar.empty()

# Hilfsfunktionen Export
def generate_pdf(query, vergleich, filename="vergleich.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo einfügen
    logo_path = "data/Health365AC_Logo_Freigestellt.png"  # Pfad zu deinem Logo (ggf. anpassen)
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=60)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.ln(40)  # Abstand unter dem Logo
    pdf.cell(0, 10, "Vergleichsanalyse DSGVO vs. NIS2", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Fragestellung:", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, query)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Vergleichsergebnis:", ln=True)

    pdf.set_font("Arial", '', 12)
    for line in vergleich.split("\n"):
        pdf.multi_cell(0, 10, line)

    return pdf.output(dest='S').encode('latin-1')

def generate_markdown(text):
    return text.encode('utf-8')

# Verarbeitung
if query or rebuild_embeddings:
    with st.spinner(current_texts["processing"]):
        dsgvo_chunks = load_and_chunk("data/dsgvo.pdf")
        nis2_chunks = load_and_chunk("data/nis2.pdf")

        if rebuild_embeddings:
            # Speicher löschen
            if os.path.exists("vector_store/faiss_dsgvo"):
                shutil.rmtree("vector_store/faiss_dsgvo")
            if os.path.exists("vector_store/faiss_nis2"):
                shutil.rmtree("vector_store/faiss_nis2")
            st.success("✅ Embeddings wurden neu erstellt!")
            status_text.success("🔄 Embeddings neu erstellt!")

        try:
            db_dsgvo = build_or_load_vectorstore(dsgvo_chunks, "vector_store/faiss_dsgvo", embedding_client)
            db_nis2 = build_or_load_vectorstore(nis2_chunks, "vector_store/faiss_nis2", embedding_client)
            status_text.success("✅ FAISS erfolgreich geladen!")
        except Exception as e:
            status_text.error(f"❌ Fehler beim Laden der Vektordatenbank: {e}")
            st.stop()

        if query:
            top_chunks_dsgvo = db_dsgvo.similarity_search(query, k=3)
            top_chunks_nis2 = db_nis2.similarity_search(query, k=3)

            st.markdown(f"## {current_texts['preview_dsgvo']}")
            for i, chunk in enumerate(top_chunks_dsgvo, 1):
                page_number = chunk.metadata.get('page', None)
                if isinstance(page_number, int):
                    page_display = page_number + 1
                else:
                    page_display = "N/A"
                with st.expander(f"DSGVO Chunk #{i} (Seite {page_display})"):
                    st.write(chunk.page_content)

            st.markdown(f"## {current_texts['preview_nis2']}")
            for i, chunk in enumerate(top_chunks_nis2, 1):
                page_number = chunk.metadata.get('page', None)
                if isinstance(page_number, int):
                    page_display = page_number + 1
                else:
                    page_display = "N/A"
                with st.expander(f"NIS2 Chunk #{i} (Seite {page_display})"):
                    st.write(chunk.page_content)

            vergleich = compare(query, db_dsgvo, db_nis2, llm_client)

            st.markdown(f"## {current_texts['comparison_result']}")
            st.write(vergleich)

            st.markdown("---")
            st.subheader("🔽 Exportieren:")

            col1, col2 = st.columns(2)

            with col1:
                pdf_file = pdf_file = generate_pdf(query, vergleich)
                st.download_button(
                    label=current_texts["export_button"] + " (PDF)",
                    data=pdf_file,
                    file_name="vergleich.pdf",
                    mime="application/pdf"
                )

            with col2:
                md_file = generate_markdown(vergleich)
                st.download_button(
                    label=current_texts["export_button"] + " (Markdown)",
                    data=md_file,
                    file_name="vergleich.md",
                    mime="text/markdown"
                )
