import os
import io
import re
import json
import yaml
import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from functools import lru_cache

import gradio as gr
import pinecone
import PyPDF2
import spacy
from docx import Document
from striprtf.striprtf import rtf_to_text
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from abc import ABC, abstractmethod

from langchain.vectorstores import Pinecone, FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import numpy as np

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config(BaseModel):
    general: Dict[str, Any] = Field(...)
    pinecone: Dict[str, str] = Field(...)
    embedding_models: List[str] = Field(...)
    vector_store: Dict[str, Any] = Field(...)
    processing: Dict[str, int] = Field(...)
    search: Dict[str, Any] = Field(...)
    logging: Dict[str, str] = Field(...)
    spacy: Dict[str, str] = Field(...)
    html_processing: Dict[str, List[str]] = Field(...)
    gradio: Dict[str, Any] = Field(...)
    redis: Dict[str, Any] = Field(...)
    languages: List[str] = Field(...)

def load_config() -> Config:
    try:
        with open('config.yaml', 'r') as config_file:
            config_data = yaml.safe_load(config_file)
        return Config(**config_data)
    except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
        logger.error(f"Error loading configuration: {e}")
        raise

CONFIG = load_config()

# Initialize Redis client
redis_client = redis.Redis(host=CONFIG.redis["host"], port=CONFIG.redis["port"], db=CONFIG.redis["db"])

@lru_cache(maxsize=1)
def load_spacy_model(lang: str):
    try:
        return spacy.load(f"{lang}_core_web_sm")
    except OSError:
        logger.error(f"Failed to load spaCy model for {lang}. Make sure to download it using: python -m spacy download {lang}_core_web_sm")
        raise

nlp_models = {lang: load_spacy_model(lang) for lang in CONFIG.languages}

class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    @lru_cache(maxsize=10000)
    def embed_text(self, text: str) -> List[float]:
        return super().embed_text(text)

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
        pass

    @abstractmethod
    def update(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        pass

class PineconeVectorStore(VectorStore):
    def __init__(self, index_name: str, embedding_model: Any):
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        if not pinecone_api_key or not pinecone_environment:
            raise ValueError("Pinecone API key or environment not found in .env file")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index = pinecone.Index(index_name)
        self.vector_store = Pinecone(self.index, embedding_model.embed_text, "text")

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        self.vector_store.add_texts(texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def similarity_search_with_score(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)

    def update(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        self.vector_store.update(id, embedding, metadata)

class FAISSVectorStore(VectorStore):
    def __init__(self, index_name: str, embedding_model: Any):
        self.vector_store = FAISS.load_local(index_name, embedding_model)
        if self.vector_store is None:
            self.vector_store = FAISS(embedding_model.embed_text, index_name)

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        self.vector_store.add_texts(texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def similarity_search_with_score(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)

    def update(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        self.vector_store.update_document(id, embedding, metadata)

    def save(self):
        self.vector_store.save_local(self.vector_store.index_name)

def create_vector_store(store_type: str, index_name: str, embedding_model: Any) -> VectorStore:
    if store_type == "pinecone":
        return PineconeVectorStore(index_name, embedding_model)
    elif store_type == "faiss":
        return FAISSVectorStore(index_name, embedding_model)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def split_text_advanced(text: str, max_chunk_size: int = CONFIG.general["max_chunk_size"], max_overlap: int = CONFIG.general["max_overlap"]) -> List[str]:
    if bool(BeautifulSoup(text, "html.parser").find()):
        return split_html(text, max_chunk_size, max_overlap)
    else:
        return split_text_semantic_and_syntactic(text, max_chunk_size, max_overlap)

def split_html(html: str, max_chunk_size: int, max_overlap: int) -> List[str]:
    soup = BeautifulSoup(html, 'html.parser')
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for element in soup.find_all(CONFIG.html_processing["tags_to_split"]):
        if element.name.startswith('h'):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            chunks.append(element.get_text())
            continue

        text = element.get_text()
        if current_chunk_size + len(text) > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            overlap = current_chunk[-1] if len(current_chunk) > 0 else ""
            current_chunk = [overlap]
            current_chunk_size = len(overlap)

        current_chunk.append(text)
        current_chunk_size += len(text)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def split_text_semantic_and_syntactic(text: str, max_chunk_size: int = CONFIG.general["max_chunk_size"], max_overlap: int = CONFIG.general["max_overlap"]) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    in_code_block = False

    for paragraph in paragraphs:
        if paragraph.startswith('```') and paragraph.endswith('```'):
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            chunks.append(paragraph)
            continue

        if paragraph.startswith('    ') or paragraph.startswith('\t'):
            if not in_code_block and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            in_code_block = True
            current_chunk.append(paragraph)
            current_chunk_size += len(paragraph)
            if current_chunk_size >= max_chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            continue

        if in_code_block and not (paragraph.startswith('    ') or paragraph.startswith('\t')):
            in_code_block = False
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0

        if re.match(r'^#{1,6}\s', paragraph):
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            chunks.append(paragraph)
            continue

        doc = nlp_models["en"](paragraph)
        for sent in doc.sents:
            sent_text = sent.text
            code_snippets = re.findall(r'`[^`\n]+`', sent_text)
            
            if code_snippets:
                parts = re.split(r'(`[^`\n]+`)', sent_text)
                for part in parts:
                    if part.startswith('`') and part.endswith('`'):
                        if current_chunk_size + len(part) > max_chunk_size and current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = []
                            current_chunk_size = 0
                        current_chunk.append(part)
                        current_chunk_size += len(part)
                    else:
                        words = part.split()
                        for word in words:
                            if current_chunk_size + len(word) + 1 > max_chunk_size and current_chunk:
                                chunks.append("\n\n".join(current_chunk))
                                overlap = current_chunk[-1] if current_chunk else ""
                                current_chunk = [overlap]
                                current_chunk_size = len(overlap)
                            current_chunk.append(word)
                            current_chunk_size += len(word) + 1
            else:
                if current_chunk_size + len(sent_text) > max_chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    overlap = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap]
                    current_chunk_size = len(overlap)
                current_chunk.append(sent_text)
                current_chunk_size += len(sent_text)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def process_documents_with_progress(documents: List[str], model_name: str, progress: Optional[gr.Progress] = None) -> List[List[float]]:
    embedding_model = CachedHuggingFaceEmbeddings(model_name=model_name)
    embeddings = []
    batch_size = CONFIG.processing["batch_size"]
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        if progress:
            progress((i + batch_size) / len(documents), f"Generating embeddings: {i + batch_size}/{len(documents)} documents")

    return embeddings

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        try:
            text += page.extract_text()
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num}: {e}")
    return text

def extract_api_spec(file_content: bytes, file_extension: str) -> str:
    if file_extension == '.json':
        spec = json.loads(file_content)
    elif file_extension in ['.yaml', '.yml']:
        spec = yaml.safe_load(file_content)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    text = f"API Title: {spec.get('info', {}).get('title', 'N/A')}\n"
    text += f"Description: {spec.get('info', {}).get('description', 'N/A')}\n\n"

    for path, methods in spec.get('paths', {}).items():
        text += f"Path: {path}\n"
        for method, details in methods.items():
            text += f"  Method: {method.upper()}\n"
            text += f"  Summary: {details.get('summary', 'N/A')}\n"
            text += f"  Description: {details.get('description', 'N/A')}\n\n"

    return text

def extract_text_from_docx(file_content: bytes) -> str:
    doc = Document(io.BytesIO(file_content))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_rtf(file_content: bytes) -> str:
    rtf_text = file_content.decode('utf-8', errors='ignore')
    return rtf_to_text(rtf_text)

def extract_file_text(file_name: str, file_content: bytes) -> Optional[str]:
    file_extension = os.path.splitext(file_name.lower())[1]
    try:
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_content)
        elif file_extension in ['.json', '.yaml', '.yml']:
            return extract_api_spec(file_content, file_extension)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_content)
        elif file_extension == '.rtf':
            return extract_text_from_rtf(file_content)
        else:
            return file_content.decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to extract text from {file_name}: {str(e)}")
        return None

def process_files_with_progress(
    files: List[Tuple[str, bytes]],
    model_name: str,
    index_name: str,
    store_type: str,
    progress: Optional[gr.Progress] = None
) -> str:
    try:
        embedding_model = CachedHuggingFaceEmbeddings(model_name=model_name)
        vector_store = create_vector_store(store_type, index_name, embedding_model)

        total_files = len(files)
        for i, (file_name, file_content) in enumerate(files):
            if progress:
                progress((i + 1) / total_files, f"Processing file {i+1} of {total_files}")

            file_text = extract_file_text(file_name, file_content)
            if file_text is None:
                continue

            documents = split_text_advanced(file_text)
            document_embeddings = process_documents_with_progress(documents, model_name, progress)
            metadata = [{"file_name": file_name, "chunk_index": j, "timestamp": time.time()} for j in range(len(documents))]
            vector_store.add_texts(documents, document_embeddings, metadata, [f"doc_{i}_{j}" for j in range(len(documents))])

        if isinstance(vector_store, FAISSVectorStore):
            vector_store.save()

        return f"Successfully processed {total_files} files and added to {store_type} index '{index_name}'."

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return f"Error processing files: {str(e)}"

def cached_similarity_search(query: str, index_name: str, top_k: int, store_type: str) -> Tuple[List[Dict[str, Any]], float]:
    cache_key = f"{query}:{index_name}:{top_k}:{store_type}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result), time.time()

    embedding_model = CachedHuggingFaceEmbeddings(model_name=CONFIG.general["default_model_name"])
    vector_store = create_vector_store(store_type, index_name, embedding_model)

    results = vector_store.similarity_search_with_score(query, k=top_k)
    cached_results = ([{"text": result[0].page_content, "score": result[1]} for result in results], time.time())
    
    redis_client.setex(cache_key, CONFIG.search["cache"]["ttl"], json.dumps(cached_results[0]))
    return cached_results

def hybrid_search(query: str, index_name: str, top_k: int = CONFIG.search["default_top_k"], store_type: str = CONFIG.vector_store["default_type"], metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    try:
        embedding_model = CachedHuggingFaceEmbeddings(model_name=CONFIG.general["default_model_name"])
        vector_store = create_vector_store(store_type, index_name, embedding_model)

        semantic_results = vector_store.similarity_search_with_score(query, k=top_k * 2, filter=metadata_filter)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([result[0].page_content for result in semantic_results])
        query_vector = tfidf_vectorizer.transform([query])

        tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

        combined_results = []
        for i, (doc, semantic_score) in enumerate(semantic_results):
            tfidf_score = tfidf_scores[i]
            combined_score = 0.7 * (1 - semantic_score) + 0.3 * tfidf_score
            combined_results.append((doc, combined_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [{"text": result[0].page_content, "score": result[1]} for result in combined_results[:top_k]]

    except Exception as e:
        logger.error(f"Error searching query: {str(e)}")
        return []

def update_document(index_name: str, doc_id: str, new_text: str, model_name: str, store_type: str) -> str:
    try:
        embedding_model = CachedHuggingFaceEmbeddings(model_name=model_name)
        vector_store = create_vector_store(store_type, index_name, embedding_model)

        new_embedding = embedding_model.embed_text(new_text)
        vector_store.update(doc_id, new_embedding, {"text": new_text})

        if isinstance(vector_store, FAISSVectorStore):
            vector_store.save()

        return f"Successfully updated document {doc_id} in {store_type} index '{index_name}'."
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        return f"Error updating document: {str(e)}"

def detect_language(text: str) -> str:
    common_words = {
        'en': ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have'],
        'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser'],
        'fr': ['le', 'la', 'de', 'et', 'un', 'une', 'du', 'des'],
        'de': ['der', 'die', 'das', 'und', 'in', 'zu', 'den', 'für'],
        'it': ['il', 'la', 'di', 'e', 'che', 'un', 'in', 'per']
    }
    
    word_count = {lang: sum(1 for word in text.lower().split() if word in words) for lang, words in common_words.items()}
    return max(word_count, key=word_count.get)

def process_multilingual_text(text: str) -> str:
    lang = detect_language(text)
    nlp = nlp_models.get(lang, nlp_models['en'])
    doc = nlp(text)
    
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown(CONFIG.gradio["markdown_title"])
        
        with gr.Tab(CONFIG.gradio["tabs"][0]):
            gr.Markdown("### Upload API documentation files (Markdown, HTML, Text, PDF, JSON, YAML) for batch embedding generation.")
            
            file_input = gr.File(file_count="multiple", label="Upload files", file_types=CONFIG.general["supported_file_types"])
            model_selector = gr.Dropdown(
                choices=CONFIG.embedding_models,
                value=CONFIG.general["default_model_name"],
                label="Select Embedding Model"
            )
            index_name_input = gr.Textbox(value=CONFIG.general["default_index_name"], label="Index Name")
            store_type_selector = gr.Dropdown(
                choices=CONFIG.vector_store["types"],
                value=CONFIG.vector_store["default_type"],
                label="Select Vector Store Type"
            )
            process_button = gr.Button("Process Files")
            output_text = gr.Textbox(label="Status", interactive=False)
            progress_bar = gr.Progress()

            process_button.click(
                process_files_with_progress,
                inputs=[file_input, model_selector, index_name_input, store_type_selector],
                outputs=output_text,
                show_progress=True
            )

        with gr.Tab(CONFIG.gradio["tabs"][1]):
            gr.Markdown("### Search the vector store for relevant API documentation.")
            search_input = gr.Textbox(label="Search Query")
            search_index_name_input = gr.Textbox(value=CONFIG.general["default_index_name"], label="Index Name")
            search_store_type_selector = gr.Dropdown(
                choices=CONFIG.vector_store["types"],
                value=CONFIG.vector_store["default_type"],
                label="Select Vector Store Type"
            )
            top_k_input = gr.Slider(minimum=1, maximum=20, value=CONFIG.search["default_top_k"], step=1, label="Number of results")
            metadata_filter_input = gr.Textbox(label="Metadata Filter (JSON)", placeholder='{"file_type": "pdf"}')
            search_button = gr.Button("Search")
            search_results = gr.JSON(label="Search Results")

            search_button.click(
                lambda query, index_name, top_k, store_type, metadata_filter: hybrid_search(query, index_name, top_k, store_type, json.loads(metadata_filter) if metadata_filter else None),
                inputs=[search_input, search_index_name_input, top_k_input, search_store_type_selector, metadata_filter_input],
                outputs=search_results
            )

        with gr.Tab(CONFIG.gradio["tabs"][2]):
            gr.Markdown("### Update an existing document in the vector store.")
            update_index_name_input = gr.Textbox(value=CONFIG.general["default_index_name"], label="Index Name")
            update_store_type_selector = gr.Dropdown(
                choices=CONFIG.vector_store["types"],
                value=CONFIG.vector_store["default_type"],
                label="Select Vector Store Type"
            )
            update_doc_id_input = gr.Textbox(label="Document ID")
            update_text_input = gr.Textbox(label="New Document Text", lines=5)
            update_model_selector = gr.Dropdown(
                choices=CONFIG.embedding_models,
                value=CONFIG.general["default_model_name"],
                label="Select Embedding Model"
            )
            update_button = gr.Button("Update Document")
            update_output = gr.Textbox(label="Status", interactive=False)

            update_button.click(
                update_document,
                inputs=[update_index_name_input, update_doc_id_input, update_text_input, update_model_selector, update_store_type_selector],
                outputs=update_output
            )

    app.launch()

if __name__ == "__main__":
    gradio_interface()
