"""
HyDE RAG (Hypothetical Document Embedding) v1

기존 쿼리 변환(step-back / rewrite / decompose) 대신,
LLM이 사용자 프로파일을 보고 '이상적인 도서의 소개글(book_intro)'을 직접 생성합니다.
생성된 가상 소개글을 임베딩하여 벡터 DB를 검색하면,
실제 book_intro 문체와 의미적으로 더 가깝기 때문에 검색 품질이 향상을 기대합니다.
graph.py 호환:
    from app.rag.query_transform_hyde import (
        extract_genre_node, query_transform_rag_node, explain_node, rag_llm_node
    )
    위 한 줄만 바꾸면 기존 graph.py 노드 구성을 그대로 사용할 수 있습니다.
"""
from __future__ import annotations

import json
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.config import QDRANT_COLLECTION_NAME
from app.db.qdrant import QdrantDB
from app.embedding.embedder import LocalEmbedder
from app.reranking.reranker import LocalReranker
from app.state.state_v3 import GraphState

# explain_node / rag_llm_node 는 기존 모듈에서 그대로 가져옵니다.
from app.rag.query_transform import explain_node, rag_llm_node  # noqa: F401

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 가상 소개글을 여러 각도에서 생성해 검색 다양성 확보
# 각도별로 별도 임베딩 → Qdrant 검색 → RRF 병합
SEARCH_LIMIT   = 10   # 각도별 Qdrant 검색 결과 수
RETRIEVE_TOP_N = 10   # 리랭킹 후 최종 반환 수

# ── 카테고리 트리 (v5와 동일) ─────────────────────────────────────────────────
_csv_path = os.path.join(
    os.path.dirname(__file__),
    "../../../research/src/rag/query_transformations/aladin_category.csv",
)
_df = pd.read_csv(_csv_path)
CATEGORY_TREE = (
    _df.groupby("category_large")["category_medium"]
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)
CATEGORY_LARGE_LIST = sorted(CATEGORY_TREE.keys())

# ── 공유 인스턴스 ─────────────────────────────────────────────────────────────
embedder = LocalEmbedder("BAAI/bge-m3")
db       = QdrantDB(vector_size=1024)
llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
reranker = LocalReranker("BAAI/bge-reranker-v2-m3")


# ── 1. 장르 추출 노드 (v5와 동일) ─────────────────────────────────────────────

top_genre_prompt = ChatPromptTemplate.from_template("""
사용자 프로파일을 보고 아래 대분류 목록에서 적합한 것을 최대 2개 선택하세요.
목록에 없는 값은 절대 반환하지 마세요.

대분류 목록: {large_list}
사용자 프로파일: {summary}

JSON으로만 반환: {{"categories": ["소설/시/희곡"]}}
""")

medium_genre_prompt = ChatPromptTemplate.from_template("""
사용자 프로파일을 보고 아래 중분류 목록에서 적합한 것을 최대 3개 선택하세요.
목록에 없는 값은 절대 반환하지 마세요.

중분류 목록: {medium_list}
사용자 프로파일: {summary}

JSON으로만 반환: {{"categories": ["한국소설"]}}
""")


def extract_genre_node(state: GraphState) -> dict:
    summary = state.get("summary", "")

    top_resp = (top_genre_prompt | llm).invoke({
        "large_list": CATEGORY_LARGE_LIST,
        "summary": summary,
    })
    try:
        top_cats = json.loads(top_resp.content)["categories"]
    except (json.JSONDecodeError, KeyError):
        top_cats = []

    if not top_cats:
        print("[Genre] 대분류 추출 실패 → 필터 없음")
        return {"genre_filter": [], "genre_level": "none"}

    medium_candidates = []
    for cat in top_cats:
        medium_candidates.extend(CATEGORY_TREE.get(cat, []))

    if not medium_candidates:
        print(f"[Genre] 대분류 fallback: {top_cats}")
        return {"genre_filter": top_cats, "genre_level": "large"}

    medium_resp = (medium_genre_prompt | llm).invoke({
        "medium_list": medium_candidates,
        "summary": summary,
    })
    try:
        medium_cats = json.loads(medium_resp.content)["categories"]
    except (json.JSONDecodeError, KeyError):
        medium_cats = []

    if not medium_cats:
        print(f"[Genre] 중분류 추출 실패 → 대분류 fallback: {top_cats}")
        return {"genre_filter": top_cats, "genre_level": "large"}

    print(f"[Genre] 대분류: {top_cats} → 중분류: {medium_cats}")
    return {"genre_filter": medium_cats, "genre_level": "medium"}


# ── 2. HyDE 가상 소개글 생성 프롬프트 ─────────────────────────────────────────
#
# 각 프롬프트는 사용자 프로파일의 서로 다른 측면을 강조하여
# 의미적으로 다양한 임베딩 벡터를 만들어냅니다.
# (하나의 프로필 → 여러 검색 벡터 → RRF 병합)

hyde_content_prompt = ChatPromptTemplate.from_template("""
당신은 도서 큐레이터입니다.
아래 사용자 프로파일을 읽고, 이 사용자에게 주제와 내용 면에서 완벽하게 맞는
도서의 소개글(book_intro)이 어떻게 쓰여있을지 작성하세요.

[규칙]
- 실제 출판사 서평이나 도서 소개에 나올 법한 문체와 어휘를 사용하세요.
- 저자명, 책 제목은 만들지 마세요. 내용과 주제만 묘사하세요.
- 사용자의 독서 목적과 선호 장르에 집중하세요.
- 200자 내외로 작성하세요.

사용자 프로파일: {summary}

가상 도서 소개 (주제/내용 측면):
""")

hyde_style_prompt = ChatPromptTemplate.from_template("""
당신은 도서 큐레이터입니다.
아래 사용자 프로파일을 읽고, 이 사용자의 독서 스타일과 난이도 선호에 딱 맞는
도서의 소개글(book_intro)이 어떻게 쓰여있을지 작성하세요.

[규칙]
- 실제 출판사 서평이나 도서 소개에 나올 법한 문체와 어휘를 사용하세요.
- 저자명, 책 제목은 만들지 마세요. 서술 방식, 구성, 난이도만 묘사하세요.
- 사용자의 독서 스타일(속도, 깊이, 형식)과 난이도 선호에 집중하세요.
- 200자 내외로 작성하세요.

사용자 프로파일: {summary}

가상 도서 소개 (독서 스타일/난이도 측면):
""")

hyde_context_prompt = ChatPromptTemplate.from_template("""
당신은 도서 큐레이터입니다.
아래 사용자 프로파일을 읽고, 이 사용자의 현재 상황과 감정에 공명하는
도서의 소개글(book_intro)이 어떻게 쓰여있을지 작성하세요.

[규칙]
- 실제 출판사 서평이나 도서 소개에 나올 법한 문체와 어휘를 사용하세요.
- 저자명, 책 제목은 만들지 마세요. 독자에게 주는 감정적/실용적 가치만 묘사하세요.
- 사용자의 현재 상황(감정 상태, 삶의 맥락)과 독서에서 얻고 싶은 것에 집중하세요.
- 200자 내외로 작성하세요.

사용자 프로파일: {summary}

가상 도서 소개 (상황/감성 측면):
""")

_HYDE_PROMPTS = [
    ("content", hyde_content_prompt),
    # ("style",   hyde_style_prompt),
    # ("context", hyde_context_prompt),
]


def generate_hypothetical_docs(summary: str) -> list[tuple[str, str]]:
    """사용자 프로파일 → [(각도 이름, 가상 book_intro), ...] 생성."""
    results = []
    for angle, prompt in _HYDE_PROMPTS:
        hypo = (prompt | llm).invoke({"summary": summary}).content.strip()
        print(f"  [HyDE/{angle}] {hypo[:80]}...")
        results.append((angle, hypo))
    return results


# ── 3. RRF ────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    scores, payloads = {}, {}
    for results in results_list:
        for rank, r in enumerate(results):
            isbn = r.payload.get("isbn", "")
            if isbn:
                scores[isbn]   = scores.get(isbn, 0) + 1 / (k + rank + 1)
                payloads[isbn] = r.payload
    return [payloads[isbn] for isbn, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ── 4. HyDE RAG 노드 ──────────────────────────────────────────────────────────

def query_transform_rag_node(state: GraphState) -> dict:
    summary    = state.get("summary", "")
    reflection = state.get("reflection", "")
    categories = state.get("genre_filter", [])
    genre_level = state.get("genre_level", "none")  # state_v3 이상에서 사용

    user_profile_query = " ".join(filter(None, [summary, reflection]))

    print("\n[HyDE] 가상 도서 소개글 생성 중...")
    hypo_docs = generate_hypothetical_docs(user_profile_query)

    field_map = {"large": "category_large", "medium": "category_medium"}
    query_filter = None
    if categories and genre_level in field_map:
        query_filter = Filter(
            must=[FieldCondition(key=field_map[genre_level], match=MatchAny(any=categories))]
        )
    elif categories:
        # state_v3 미사용 환경 (기본 state.py) → cate_depth1 필터 fallback
        query_filter = Filter(
            must=[FieldCondition(key="cate_depth1", match=MatchAny(any=categories))]
        )

    all_results = []
    for angle, hypo_text in hypo_docs:
        query_vector = embedder.embed(hypo_text)
        if query_filter:
            results = db.search_with_filter(
                QDRANT_COLLECTION_NAME, query_vector,
                query_filter=query_filter, limit=SEARCH_LIMIT, threshold=0.5,
            )
        else:
            results = db.search(
                QDRANT_COLLECTION_NAME, query_vector,
                limit=SEARCH_LIMIT, threshold=0.5,
            )
        print(f"  [HyDE/{angle}] 검색 결과: {len(results)}건")
        all_results.append(results)

    merged_payloads   = reciprocal_rank_fusion(all_results)
    reranked_payloads = reranker.rerank(query=user_profile_query, books=merged_payloads)

    retrieved_books = [
        {
            "isbn":       p.get("isbn"),
            "title":      p.get("title"),
            "author":     p.get("author"),
            "book_intro": p.get("book_intro"),
            "cate_depth1": p.get("cate_depth1"),
            # v5 필드 (있으면 포함)
            "category_large":  p.get("category_large", ""),
            "category_medium": p.get("category_medium", ""),
            "cover_url":       p.get("cover_url", ""),
        }
        for p in reranked_payloads[:RETRIEVE_TOP_N]
    ]

    hypothetical_doc = hypo_docs[0][1] if hypo_docs else ""

    print(f"\n[HyDE] 최종 검색 결과: {len(retrieved_books)}권")
    return {"retrieved_books": retrieved_books, "hypothetical_doc": hypothetical_doc}
