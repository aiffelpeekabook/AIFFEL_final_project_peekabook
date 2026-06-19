import os
from dotenv import load_dotenv
from datetime import timezone, timedelta

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CLOVASTUDIO_API_KEY: str = os.getenv("CLOVASTUDIO_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
LIBRARY_API_KEY: str = os.getenv("LIBRARY_API_KEY", "")
NAVER_CLIENT_ID: str = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET: str = os.getenv("NAVER_CLIENT_SECRET", "")
ALADIN_API_KEY: str = os.getenv("ALADIN_API_KEY", "")
ALADIN_API_KEYS: str = os.getenv("ALADIN_API_KEYS", "")

QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "books_v1")

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

MAX_SLOT_RETRIES: int = 3
SIMILARITY_THRESHOLD: float = 0.5
CONFIDENCE_THRESHOLD: float = 0.6
SIMILAR_SEARCH_K: int = 3
LINK_CANDIDATE_K: int = 5

JUDGE_MODEL = "claude-haiku-4-5-20251001" # peeka_judge
 
MAX_TURNS = 12 # 대화 제어
 
KST = timezone(timedelta(hours=9)) # 한국 시간대 (timestamp / log 표기용)
 
# 시뮬레이션 경로 (env로 오버라이드 가능)
SIMULATION_CHROMA_BASE = os.getenv(
    "SIMULATION_CHROMA_BASE",
    "backend/chroma_db_runs",
)
SIMULATION_RESULTS_DIR = os.getenv(
    "SIMULATION_RESULTS_DIR",
    "research/data/simulation_results",
)
PERSONA_DIR = os.getenv(
    "PERSONA_DIR",
    "backend/data/personas",
)