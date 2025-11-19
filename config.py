# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load .env file (root folder mein .env bana lena)
load_dotenv()

class Config:
    # ====================== APP INFO ======================
    APP_NAME = "Universal AI Assistant Pro Max"
    VERSION = "9.9.9"
    AUTHOR = "Desi Developer"
    TAGLINE = "Har Kaam Mein Madad • Full Indian AI Power"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # ====================== PATHS ======================
    BASE_DIR = Path(__file__).parent.resolve()
    UPLOAD_DIR = BASE_DIR / "uploads"
    VAULT_DIR = BASE_DIR / "file_vault"
    LOG_DIR = BASE_DIR / "logs"
    VECTOR_DB_PATH = VAULT_DIR / "chroma_db"

    # Auto create directories
    for dir_path in [UPLOAD_DIR, VAULT_DIR, LOG_DIR, VECTOR_DB_PATH]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ====================== SECURITY ======================
    SECRET_KEY = os.getenv("SECRET_KEY", "desi_super_secret_key_2025_change_kar_le")
    ENCRYPTION_SALT = os.getenv("ENCRYPTION_SALT", "desi_salt_masala_2025").encode()

    # ====================== AI MODEL SETTINGS ======================
    class Model:
        # Local Models (Offline Ready)
        LOCAL_SMALL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"           # CPU pe chalega
        LOCAL_MEDIUM = "microsoft/DialoGPT-medium"
        LOCAL_BEST = "Qwen/Qwen2.5-7B-Instruct"                     # 4-bit GPU beast
        LOCAL_VISION = "llava-hf/llava-1.5-7b-hf"                    # Image samajhta hai

        # Online APIs (Free + Blazing Fast)
        GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        DEEPSEEK_MODEL = "deepseek-chat"
        GEMINI_MODEL = "gemini-1.5-pro"
        OPENAI_MODEL = "gpt-4o-mini"

        MAX_TOKENS = 1024
        TEMPERATURE = 0.8
        STREAMING = True

    # ====================== FILE SETTINGS ======================
    class Files:
        MAX_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
        ALLOWED_EXTENSIONS = {
            '.txt', '.pdf', '.docx', '.csv', '.xlsx', '.xls',
            '.pptx', '.jpg', '.jpeg', '.png', '.json', '.md',
            '.py', '.html', '.htm', '.zip'
        }
        OCR_LANGUAGES = "hin+eng"  # Hindi + English OCR

    # ====================== API KEYS ======================
    class API:
        # Free APIs (Sab working 2025 mein)
        GROQ = os.getenv("GROQ_API_KEY")
        DEEPSEEK = os.getenv("DEEPSEEK_API_KEY")
        GEMINI = os.getenv("GEMINI_API_KEY")
        HF_TOKEN = os.getenv("HF_TOKEN")
        SERPER = os.getenv("SERPER_API_KEY")           # Google search
        BRAVE = os.getenv("BRAVE_API_KEY")            # Brave search
        NEWSAPI = os.getenv("NEWS_API_KEY")
        JINA = "https://s.jina.ai/"  # No key needed

        @classmethod
        def is_online_ready(cls):
            return any([cls.GROQ, cls.DEEPSEEK, cls.GEMINI, cls.HF_TOKEN])

    # ====================== FEATURES ======================
    class Features:
        VOICE_MODE = os.getenv("ENABLE_VOICE", "true").lower() == "true"
        RAG_ENABLED = True
        OCR_ENABLED = True
        ENCRYPTION_ENABLED = True
        DARK_MODE = True
        HINGLISH_MODE = True
        AUTO_BACKUP = False

    # ====================== UI SETTINGS ======================
    class UI:
        THEME = "dark"
        PRIMARY_COLOR = "#FF6B6B"
        FONT = "Sans Serif"
        AVATAR_USER = "user
        AVATAR_BOT = "robot"

    # ====================== LOGGING ======================
    LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

    # ====================== HELPER METHOD ======================
    @staticmethod
    def print_banner():
        banner = f"""
╔═══════════════════════════════════════════════════════════════╗
   {Config.APP_NAME} v{Config.VERSION}
   {Config.TAGLINE}
   
   Mode: {'ONLINE + OFFLINE' if Config.API.is_online_ready() else 'FULL OFFLINE'}
   Voice: {'ON' if Config.Features.VOICE_MODE else 'OFF'}
   RAG: {'ON' if Config.Features.RAG_ENABLED else 'OFF'}
   Made with Indian Power by {Config.AUTHOR}
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)

# Auto print banner jab config load ho
Config.print_banner()
