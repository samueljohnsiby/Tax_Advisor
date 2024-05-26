import os
import numpy as np
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain.docstore.document import Document
import faiss

# Assuming RAW_KNOWLEDGE_BASE is a list of paths to PDF files
RAW_KNOWLEDGE_BASE = ["assets/income_tax_india_1961.pdf"]

class PDFLoader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def load_documents(self):
        knowledge_base = []
        for pdf_path in self.pdf_paths:
            try:
                # Open the PDF with PyPDF2
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    # Extract text content from each page
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    knowledge_base.append(text)
            except FileNotFoundError:
                print(f"Error: PDF file not found - {pdf_path}")
        return knowledge_base

class DocumentProcessor:
    def __init__(self, chunk_size, tokenizer_name):
        self.chunk_size = chunk_size
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, knowledge_base):
        docs_processed = []
        for doc in knowledge_base:
            # Convert text to langchain.Document
            lc_doc = Document(page_content=doc)
            docs_processed += self.text_splitter.split_documents([lc_doc])
        
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
        
        return docs_processed_unique

class TextEmbedder:
    def __init__(self, model_name, chunk_size):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_texts(self, docs):
        embeddings = []
        for doc in docs:
            inputs = self.tokenizer(doc.page_content, return_tensors='pt', truncation=True, padding='max_length', max_length=self.chunk_size)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=self.chunk_size)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def save_embeddings(self, embeddings, file_path):
        np.save(file_path, embeddings)

    def load_embeddings(self, file_path):
        return np.load(file_path)

class DocumentSearch:
    def __init__(self, embeddings):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=7):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return indices
class GetContext:
    def __init__(self):
        self.pdf_loader = PDFLoader(RAW_KNOWLEDGE_BASE)
        self.knowledge_base = self.pdf_loader.load_documents()

        # Process documents
        self.chunk_size = 512
        self.processor = DocumentProcessor(self.chunk_size, tokenizer_name="thenlper/gte-small")
        self.docs_processed = self.processor.split_documents(self.knowledge_base)

        # Embed documents and save/load embeddings
        self.embedder = TextEmbedder(model_name="thenlper/gte-small", chunk_size=self.chunk_size)
        self.embeddings_file = "embeddings.npy"

    def get_context(self,query):
        # Load documents
        
        
        if not os.path.exists(self.embeddings_file):
            print("here")
            embeddings = self.embedder.embed_texts(self.docs_processed)
            self.embedder.save_embeddings(embeddings, self.embeddings_file)
        else:
            embeddings = self.embedder.load_embeddings(self.embeddings_file)

        # Search query
        # query = "I have an insurance of 10000, can I write it off, if so under what rule"
        query_embedding = self.embedder.embed_query(query)

        # Perform search
        searcher = DocumentSearch(embeddings)
        top_k = 7
        results = searcher.search(query_embedding, top_k)
        retrieved_docs = [self.docs_processed[i] for i in results[0]]

        # Print the retrieved documents
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        print(context)
        return context

