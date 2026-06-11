# src/embedding/embedder.py

from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass


# ─────────────────────────────────────────
# 로컬 모델
# ─────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    로컬 모델 임베딩 (sentence-transformers 기반)

    Parameters
    ----------
    model_name : HuggingFace 모델명 (기본: BAAI/bge-m3, vector_size=1024)

    예)
    embedder = LocalEmbedder()
    embedder = LocalEmbedder("BAAI/bge-m3")
    """
    def __init__(self, model_name="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()     # 출력 형식 : 단일 리스트

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()    # 출력 형식 : 벡터 여러 개를 담은 리스트


# ─────────────────────────────────────────
# API 기반 모델
# ─────────────────────────────────────────

class APIEmbedder(BaseEmbedder, ABC):
    """
    API 기반 임베딩 베이스 클래스.
    외부 API를 사용하는 임베더는 이 클래스를 상속해 구현.

    구현체 목록:
    - OpenAIEmbedder : OpenAI Embeddings API
    """
    pass


# class OpenAIEmbedder(APIEmbedder):
#     """
#     OpenAI Embeddings API

#     Parameters
#     ----------
#     model_name : 사용할 모델 (기본: text-embedding-3-small)
#     dimensions : 출력 벡터 차원 수 (기본: 1024 → bge-m3과 동일하게 맞춰 컬렉션 재사용 가능)

#     예)
#     embedder = OpenAIEmbedder()
#     embedder = OpenAIEmbedder("text-embedding-3-large", dimensions=1024)
#     """
#     def __init__(self, model_name="text-embedding-3-small", dimensions=1024):
#         from openai import OpenAI
#         load_dotenv()
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         self.model_name = model_name
#         self.dimensions = dimensions

#     def embed(self, text: str) -> list[float]:
#         response = self.client.embeddings.create(
#             model=self.model_name,
#             input=text,
#             dimensions=self.dimensions
#         )
#         return response.data[0].embedding

#     def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         response = self.client.embeddings.create(
#             model=self.model_name,
#             input=texts,
#             dimensions=self.dimensions
#         )
#         return [item.embedding for item in response.data]
