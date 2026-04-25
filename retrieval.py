from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
import uuid
from langfuse import observe
import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
import pickle


vectorstore = None
all_child_chunks = []

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

def save_child_chunks(all_child_chunks, path="child_chunks.pkl"):
    with open(path, "wb") as f:
        pickle.dump(all_child_chunks, f)

def load_child_chunks(path="child_chunks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@observe()
def save_to_disk(parent_chunk, path="parent_chunk.json"):
    with open(path, "w") as f:
        json.dump(parent_chunk, f)

@observe()
def load_from_disk(path="parent_chunk.json"):
    with open(path, "r") as f:
        return json.load(f)

@observe()
def document_loading():
    print("Loading PDF...")
    loader = DirectoryLoader("docs/", glob="*.pdf", loader_cls=PyMuPDFLoader)
    document = loader.load()
    print(f"Loaded {len(document)} pages")
    return document

@observe()
def parenttext_splitting(document):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parent_chunks = parent_splitter.split_documents(document)
    return parent_chunks

@observe()
def metadata(child_chunks, parent_id):
    prompt = f"""You are a metadata extractor. Extract metadata from the chunk.

Chunk: {child_chunks.page_content}
Parent ID: {parent_id}

Return ONLY a flat JSON dictionary where:
- All keys are strings
- All values are strings, numbers, or booleans ONLY
- NO nested dictionaries
- NO lists
- NO complex objects

Required fields:
- parent_id: {parent_id}
- content_type: one of "side_effects", "dosage", "drug_interaction", "warnings", "clinical_trials", "general"
- drug: "aripiprazole" if mentioned, else "unknown"
- population: "adult", "pediatric", "elderly", or "general"

Example output:
{{"parent_id": "{parent_id}", "content_type": "side_effects", "drug": "aripiprazole", "population": "adult"}}

Return only the JSON dictionary, nothing else."""

    response = llm.invoke(prompt)
    try:
        metadata_dict = json.loads(response.content)
        clean_metadata = {}
        for key, value in metadata_dict.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean_metadata[key] = value
        return clean_metadata
    except:
        return {
            "parent_id": parent_id,
            "content_type": "general",
            "drug": "unknown",
            "population": "general"
        }

@observe()
def child_chunk_creation(parent_chunks):
    parent_dict = {}
    all_child_chunks = []
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for i, chunk in enumerate(parent_chunks):
        parent_id = str(uuid.uuid4())
        parent_dict[parent_id] = chunk.page_content
        child_chunk = child_splitter.split_documents([chunk])
        print(f"Parent chunk {i} split into {len(child_chunk)} child chunks.")
        all_child_chunks.extend(child_chunk)
        for j in child_chunk:
            metadata_info = metadata(j, parent_id)
            j.metadata.update(metadata_info)
    save_to_disk(parent_dict, path="parent_chunk.json")
    save_child_chunks(all_child_chunks)
    return all_child_chunks

@observe()
def vectorstore_creation(all_child_chunks):
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db"
    )
    vectorstore.add_documents(all_child_chunks)
    return vectorstore

@observe()
def hybrid_search_rerank(query):  # only query — uses globals
    global vectorstore, all_child_chunks
    vector = vectorstore.as_retriever(search_kwargs={"k": 10})
    lexical = BM25Retriever.from_documents(all_child_chunks)
    lexical.k = 10
    ensemble = EnsembleRetriever(retrievers=[vector, lexical], weights=[0.6, 0.4])
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
    comp_retrieve = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble)
    results = comp_retrieve.invoke(query)
    return results

@observe()
def attach_parent_context(results):
    parent_text = []
    for i in results:
        parent = i.metadata.get("parent_id")
        if parent:
            parent_content = load_from_disk().get(parent, i.page_content)
            parent_text.append(parent_content)
        else:
            parent_text.append(i.page_content)
    return "\n\n".join([t for t in parent_text if t])

@observe()
def initialize_retrieval():
    global vectorstore, all_child_chunks

    if (os.path.exists("./chroma_db") and
        os.path.exists("parent_chunk.json") and
        os.path.exists("child_chunks.pkl")):

        print("Loading from disk...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        result = vectorstore.get(include=["documents", "metadatas"])
        all_child_chunks = [
            Document(page_content=i, metadata=j)
            for i, j in zip(result["documents"], result["metadatas"])
        ]

    elif os.path.exists("child_chunks.pkl") and os.path.exists("parent_chunk.json"):
        print("Loading existing child chunks...")
        all_child_chunks = load_child_chunks(path="child_chunks.pkl")
        vectorstore = vectorstore_creation(all_child_chunks)

    else:
        print("Building from scratch...")
        documents = document_loading()
        parent_chunks = parenttext_splitting(documents)
        all_child_chunks = child_chunk_creation(parent_chunks)
        vectorstore = vectorstore_creation(all_child_chunks)

    print("Retrieval initialized successfully")

if __name__ == "__main__":
    initialize_retrieval()
    query = "What are the side effects of Abilify?"
    results = hybrid_search_rerank(query)
    context = attach_parent_context(results)
    print(context)