import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI

# Lade Umgebungsvariablen
load_dotenv()

# Wrapper-Klasse für Embeddings
class AzureEmbeddingWrapper:
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=os.getenv("OPENAI_DEPLOYMENT_EMBEDDING_NAME"),
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        return self.embed_query(text)

# Azure Health Check Funktion
def health_check():
    client = AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_CHAT_NAME"),
        model="gpt-4-32k",
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0.2,
    )
    _ = client.invoke("PING")

# OpenAI Azure Embedding-Client erstellen
def create_embedding_client():
    return AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

# Azure Chat-Client erstellen
def create_chat_client():
    return AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_CHAT_NAME"),
        model="gpt-4-32k",
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0.2,
    )

# Lade und chunke ein PDF-Dokument
def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

# Vektorstore laden oder neu erstellen
def build_or_load_vectorstore(chunks, path, embedding_client):
    index_file = os.path.join(path, "index.faiss")
    embedding = AzureEmbeddingWrapper(embedding_client)
    if os.path.exists(index_file):
        # Lade FAISS mit explizitem Trust
        return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_texts(
            [doc.page_content for doc in chunks],
            embedding
        )
        vs.save_local(path)
        return vs

# Zwei Chunks vergleichen mit festen Textnamen DSGVO und NIS2
def compare_chunks(chunk1, chunk2, llm):
    prompt = PromptTemplate.from_template("""
Vergleiche die folgenden zwei Gesetzestexte:

--- DSGVO:
{a}

--- NIS2:
{b}

Erstelle eine juristisch präzise Analyse:
- Was ist ähnlich?
- Wo unterscheiden sich die Texte?
- Ergänzen sich bestimmte Punkte?
""")
    antwort = llm.invoke(prompt.format(a=chunk1, b=chunk2))
    return antwort.content

# Vergleich basierend auf Suchbegriff
def compare(query_text, db1, db2, llm):
    top_chunk_1 = db1.similarity_search(query_text, k=1)[0]
    top_chunk_2 = db2.similarity_search(query_text, k=1)[0]
    return compare_chunks(top_chunk_1.page_content, top_chunk_2.page_content, llm)
