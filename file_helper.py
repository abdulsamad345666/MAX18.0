# modules/file_helper.py
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import magic  # pip install python-magic
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path  # pip install pdf2image pytesseract
from PIL import Image
import openpyxl
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langning_community.embeddings import HuggingFaceEmbeddings
import shutil
import zipfile
import bcrypt

class FileGod:
    def __init__(self, storage_path="./file_vault"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.vector_db_path = self.storage_path / "vector_db"
        self.encryption_key = b"desi_file_god_2025"  # Real mein .env se lo
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.supported = [
            '.txt', '.pdf', '.docx', '.csv', '.xlsx', '.xls',
            '.pptx', '.json', '.md', '.html', '.py', '.jpg', '.png', '.jpeg'
        ]

    def upload_and_process(self, uploaded_file) -> dict:
        """Upload + Auto-detect + OCR + Index + Encrypt"""
        file_path = self.storage_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        result = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "uploaded_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "type": magic.from_file(str(file_path), mime=True),
            "status": "success"
        }

        # Auto OCR for scanned PDFs/Images
        if result["type"] in ["application/pdf", "image/jpeg", "image/png"]:
            if self._is_scanned_pdf(file_path) or result["type"].startswith("image"):
                text = self._ocr_extract(file_path)
                txt_path = file_path.with_suffix(".extracted.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                result["ocr_done"] = True
                result["extracted_text"] = text[:1000] + "..."

        # Auto index in vector DB for AI search
        self._add_to_rag(file_path)

        # Optional: Encrypt sensitive files
        if any(kw in uploaded_file.name.lower() for kw in ["salary", "bank", "aadhaar", "pan"]):
            self._encrypt_file(file_path)

        return result

    def _is_scanned_pdf(self, file_path):
        """Detect if PDF is scanned (no text layer)"""
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages[:2]:
                    text += page.extract_text() or ""
                return len(text.strip()) < 50
        except:
            return True

    def _ocr_extract(self, file_path):
        """Extract text from scanned PDF or Image using Tesseract"""
        try:
            if file_path.suffix.lower() == ".pdf":
                images = convert_from_path(file_path, dpi=200)
            else:
                images = [Image.open(file_path)]

            text = ""
            for img in images:
                text += pytesseract.image_to_string(img, lang='hin+eng') + "\n"
            return text
        except Exception as e:
            return f"OCR Error: {str(e)}"

    def _add_to_rag(self, file_path):
        """Add file to vector database for AI search"""
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif file_path.suffix.lower() == ".csv":
                loader = CSVLoader(str(file_path))
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            if os.path.exists(self.vector_db_path):
                db = Chroma(persist_directory=str(self.vector_db_path), embedding_function=self.embeddings)
                db.add_documents(splits)
            else:
                Chroma.from_documents(splits, self.embeddings, persist_directory=str(self.vector_db_path))
            db.persist()
        except Exception as e:
            print(f"RAG Index Error: {e}")

    def ai_search_in_files(self, query: str) -> str:
        """AI-powered search across ALL uploaded files"""
        try:
            if not os.path.exists(self.vector_db_path):
                return "Koi file upload nahi kiya abhi tak!"

            db = Chroma(persist_directory=str(self.vector_db_path), embedding_function=self.embeddings)
            results = db.similarity_search(query, k=5)

            response = f"**AI ne ye mila tere files mein:**\n\n"
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                snippet = doc.page_content.replace("\n", " ")[:300]
                response += f"{i+1}. **{Path(source).name}** (Page {page})\n{snippet}...\n\n"
            return response
        except Exception as e:
            return f"Search error: {str(e)}"

    def get_summary(self, file_path):
        """AI-level smart summary"""
        text = self._read_any_file(file_path)
        if len(text) > 100:
            return {
                "preview": text[:600] + "...",
                "words": len(text.split()),
                "characters": len(text),
                "smart_summary": f"Ye file mein mainly {len(text.split())} words hain. Important lagta hai!",
                "file_type": magic.from_file(file_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%d %b %Y")
            }
        return {"full_text": text}

    def _read_any_file(self, file_path):
        """Universal reader"""
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".pdf":
                reader = PdfReader(file_path)
                return " ".join([p.extract_text() or "" for p in reader.pages])
            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                return df.head(10).to_string()
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                return df.head(10).to_string()
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except:
            return "File padh nahi paya bhai!"

    def export_all_files(self):
        """Download all files as ZIP"""
        zip_path = self.storage_path / "my_files_backup.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in self.storage_path.rglob("*"):
                if file.is_file() and "vector_db" not in str(file):
                    zipf.write(file, file.relative_to(self.storage_path.parent))
        return str(zip_path)

    def _encrypt_file(self, file_path):
        """Basic encryption for sensitive files"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            encrypted = bcrypt.hashpw(data, bcrypt.gensalt())
            with open(file_path.with_suffix(".encrypted"), "wb") as f:
                f.write(encrypted)
            os.remove(file_path)  # Original delete
        except:
            pass

    def list_all_files(self):
        """Show all uploaded files with details"""
        files = []
        for file in self.storage_path.glob("*.*"):
            if "vector_db" not in str(file) and ".encrypted" not in str(file):
                stat = file.stat()
                files.append({
                    "name": file.name,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "uploaded": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y"),
                    "type": magic.from_file(str(file), mime=True)
                })
        return files
