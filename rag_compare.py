import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI

load_dotenv()

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

def create_embedding_client():
    return AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

def create_chat_client():
    return AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_CHAT_NAME"),
        model="gpt-4-32k",
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0.2,
    )

def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Extra-Metadaten hinzufügen
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_page"] = chunk.metadata.get("page", "N/A")
        chunk.metadata["title_guess"] = chunk.page_content.split("\n")[0][:80]
    return chunks

def build_or_load_vectorstore(chunks, name, embedding_client):
    path = os.path.join("vector_store", f"faiss_{name.lower()}")
    index_file = os.path.join(path, "index.faiss")
    embedding = AzureEmbeddingWrapper(embedding_client)

    if os.path.exists(index_file):
        return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(chunks, embedding)
        vs.save_local(path)
        return vs

def compare_chunks(chunk1, chunk2, llm, name1="Text A", name2="Text B"):
    prompt = PromptTemplate.from_template(f"""
Vergleiche die folgenden zwei Gesetzestexte:

--- {name1}:
{{a}}

--- {name2}:
{{b}}

Gib eine juristisch präzise Analyse:
- Was ist ähnlich?
- Wo unterscheiden sich die Texte?
- Ergänzen sich bestimmte Punkte?
""")
    antwort = llm.invoke(prompt.format(a=chunk1, b=chunk2))
    return antwort.content

def compare(query_text, db1, db2, llm, name1, name2):
    top_chunk_1 = db1.similarity_search(query_text, k=1)[0]
    top_chunk_2 = db2.similarity_search(query_text, k=1)[0]
    return compare_chunks(top_chunk_1.page_content, top_chunk_2.page_content, llm, name1, name2)
