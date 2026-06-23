"""
# query_transform_main.py

query_transform_v5 기반 + explain_node + rag_llm_node 추가.
웹 데모 최종 응답 생성까지 포함한 전체 RAG 파이프라인.
"""
from __future__ import annotations

import json
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.config import QDRANT_COLLECTION_NAME
from app.db.qdrant import QdrantDB
from app.embedding.embedder import LocalEmbedder
from app.reranking.reranker import LocalReranker
from app.state.state_v3 import GraphState

load_dotenv()

# ── Query Transformation 플래그 ───────────────────────────────────────────────
USE_ORIGINAL  = False
USE_STEP_BACK = False
USE_REWRITE   = False
USE_DECOMPOSE = False
USE_HYDE      = True

# ── 검색 결과 크기 ────────────────────────────────────────────────────────────
SEARCH_LIMIT   = 10  # 쿼리당 Qdrant 검색 결과 수
RETRIEVE_TOP_N = 10  # 리랭킹 후 최종 반환 수

# ── 초기화 ────────────────────────────────────────────────────────────────────

_csv_path = os.path.join(os.path.dirname(__file__), "../../data/aladin_category.csv")
if not os.path.exists(_csv_path):
    _csv_path = os.path.join(os.path.dirname(__file__), "../../../research/src/rag/query_transformations/aladin_category.csv")
_df = pd.read_csv(_csv_path)

CATEGORY_TREE = (
    _df.groupby("category_large")["category_medium"]
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)

CATEGORY_LARGE_LIST = sorted(CATEGORY_TREE.keys())

embedder = LocalEmbedder("BAAI/bge-m3")
db       = QdrantDB(vector_size=1024)
llm      = ChatOpenAI(model="gpt-4o-mini")
reranker = LocalReranker("BAAI/bge-reranker-v2-m3")


# ## 1. 장르 추출 노드

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


# ## 2. Step-back Prompting

step_back_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 검색 쿼리 전문가입니다.

아래 사용자의 원본 질문에서 한 단계 물러나,
이 사용자가 근본적으로 어떤 종류의 독서 경험을 원하는지를 포착하는 상위 질문을 생성하세요.

[규칙]
- 사용자가 언급한 구체적인 장르명, 책 제목, 조건을 그대로 반복하지 마세요.
- 대신, 그 조건들이 가리키는 더 넓은 독서 욕구나 도서 유형의 본질적 특성을 표현하세요.
- 도서 소개글(book_intro)에 실제로 등장할 법한 서술 표현을 사용하세요.
- 3문장 이내로 작성하세요.

사용자 프로파일: {summary}

출력:
""")


def step_back_query(summary: str, llm) -> str:
    return (step_back_prompt | llm).invoke({"summary": summary}).content.strip()


# ## 3. Query Rewriting

rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 검색 쿼리 전문가입니다.

아래 사용자 프로파일을 바탕으로, 벡터 검색에 적합한 도서 검색 쿼리를 작성하세요.

[규칙]
- 도서 소개글(book_intro)이나 출판사 서평에 실제로 등장할 법한 어휘와 표현을 사용하세요.
- 장르, 주제 영역, 서술 방식, 대상 독자층 등 도서 메타데이터와 매칭될 수 있는 조건을 포함하세요.
- 사용자가 언급한 기존 도서가 있다면, 그 도서의 핵심 특성(서술 방식, 주제 범위)을 반영하세요.
- 3문장 이내로 작성하세요.

독서 목적: {summary}

재작성된 검색 쿼리 (두 문장 이내로):
""")


def rewrite_query(summary: str, llm) -> str:
    return (rewrite_prompt | llm).invoke({"summary": summary}).content.strip()


# ## 4. HyDE (Hypothetical Document Embedding)

_hyde_prompt = ChatPromptTemplate.from_template("""
당신은 도서 큐레이터입니다.
아래 사용자 프로파일을 읽고, 이 사용자에게 주제와 내용 면에서 완벽하게 맞는
도서의 소개글(book_intro)이 어떻게 쓰여있을지 작성하세요.

[규칙]
- 실제 출판사 서평이나 도서 소개에 나올 법한 문체와 어휘를 사용하세요.
- 저자명, 책 제목은 만들지 마세요. 내용과 주제만 묘사하세요.
- 사용자의 독서 목적과 선호 장르에 집중하세요.
- 200자 내외로 작성하세요.

사용자 프로파일: {summary}

가상 도서 소개:
""")


def generate_hypothetical_doc(summary: str, llm) -> str:
    return (_hyde_prompt | llm).invoke({"summary": summary}).content.strip()


# ## 5. Sub-query Decomposition

decompose_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 검색 쿼리 전문가입니다.

아래 검색 쿼리를 2~4개의 서브쿼리로 분해하세요.

[핵심 원칙]
- 각 서브쿼리는 서로 다른 독립적 검색 측면을 다뤄야 합니다.
  동일한 의미를 다른 표현으로 반복하는 것은 서브쿼리가 아닙니다.
- 분해 기준 예시: 주제/장르 측면, 서술 방식/구조 측면, 유사 도서 특성 측면, 대상 독자 상황 측면
- 각 서브쿼리는 독립적으로 검색했을 때 서로 다른 후보 도서군을 반환할 수 있어야 합니다.

[작성 규칙]
- 도서 소개글(book_intro)에 등장할 법한 어휘를 사용하세요.
- "이 중에서", "그 중에서" 같은 참조 표현은 사용하지 마세요.
- "추천해주세요", "알고 싶습니다" 같은 요청형 종결은 사용하지 마세요.
- 리뷰, 평점 등 도서 소개글 외의 정보를 요청하지 마세요.

검색 쿼리: {rewritten}

출력 형식 (번호와 텍스트만, 다른 텍스트 없이):
1. [서브쿼리 1]
2. [서브쿼리 2]
3. [서브쿼리 3]
""")


def decompose_query(rewritten: str, llm) -> list:
    response = (decompose_prompt | llm).invoke({"rewritten": rewritten}).content
    return [
        q.strip().lstrip("1234567890. ")
        for q in response.split("\n")
        if q.strip() and q.strip()[0].isdigit()
    ]


# ## 6. Chained Pipeline

def get_chained_queries(user_profile_query: str, llm,
                        use_original: bool = True,
                        use_step_back: bool = True,
                        use_rewrite: bool = True,
                        use_decompose: bool = True,
                        use_hyde: bool = False) -> dict:
    all_queries      = []
    hypothetical_doc = ""

    if use_original:
        all_queries.append(user_profile_query)

    step_back = step_back_query(user_profile_query, llm) if use_step_back else ""
    print(f"  [Step-back]  : {step_back}")
    if use_step_back:
        all_queries.append(step_back)

    rewritten = rewrite_query(user_profile_query, llm) if use_rewrite else user_profile_query
    print(f"  [Rewritten]  : {rewritten}")
    if use_rewrite:
        all_queries.append(rewritten)

    sub_queries = decompose_query(rewritten, llm) if use_decompose else []
    print(f"  [Sub-queries]: {sub_queries}")
    all_queries.extend(sub_queries)

    if use_hyde:
        hypothetical_doc = generate_hypothetical_doc(user_profile_query, llm)
        print(f"  [HyDE]       : {hypothetical_doc[:80]}...")
        all_queries.append(hypothetical_doc)

    return {
        "step_back":        step_back,
        "rewritten":        rewritten,
        "sub_queries":      sub_queries,
        "hypothetical_doc": hypothetical_doc,
        "all":              all_queries,
    }


# ## 7. RRF

def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    scores, payloads = {}, {}
    for results in results_list:
        for rank, r in enumerate(results):
            isbn = r.payload.get("isbn", "")
            if isbn:
                scores[isbn]   = scores.get(isbn, 0) + 1 / (k + rank + 1)
                payloads[isbn] = r.payload
    return [payloads[isbn] for isbn, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ## 8. Query Transform RAG 노드

def query_transform_rag_node(state: GraphState) -> dict:
    summary     = state.get("summary", "")
    reflection  = state.get("reflection", "")
    categories  = state.get("genre_filter", [])
    genre_level = state.get("genre_level", "none")

    print("---------------------")
    print("summary:", summary)
    print("reflection:", reflection)
    print("---------------------")

    user_profile_query = " ".join(filter(None, [summary, reflection]))

    print("\n[Query Transformations]")
    queries = get_chained_queries(
        user_profile_query, llm,
        use_original=USE_ORIGINAL,
        use_step_back=USE_STEP_BACK,
        use_rewrite=USE_REWRITE,
        use_decompose=USE_DECOMPOSE,
        use_hyde=USE_HYDE,
    )
    all_queries      = queries["all"] or [user_profile_query]
    hypothetical_doc = queries.get("hypothetical_doc", "")
    query_transforms = {
        "original":    user_profile_query,
        "step_back":   queries.get("step_back", "") if USE_STEP_BACK else "",
        "rewritten":   queries.get("rewritten", "") if USE_REWRITE else "",
        "sub_queries": queries.get("sub_queries", []),
        "all":         all_queries,
    }

    field_map = {"large": "category_large", "medium": "category_medium"}
    query_filter = None
    if categories and genre_level in field_map:
        query_filter = Filter(
            must=[FieldCondition(key=field_map[genre_level], match=MatchAny(any=categories))]
        )

    all_results = []
    for query in all_queries:
        query_vector = embedder.embed(query)
        if query_filter:
            results = db.search_with_filter(
                QDRANT_COLLECTION_NAME, query_vector,
                query_filter=query_filter, limit=SEARCH_LIMIT, threshold=0.5,
            )
        else:
            results = db.search(QDRANT_COLLECTION_NAME, query_vector, limit=SEARCH_LIMIT, threshold=0.5)
        all_results.append(results)

    merged_payloads   = reciprocal_rank_fusion(all_results)
    reranked_payloads = reranker.rerank(query=user_profile_query, books=merged_payloads)

    retrieved_books = [
        {
            "isbn":            p.get("isbn"),
            "title":           p.get("title"),
            "author":          p.get("author"),
            "book_intro":      p.get("book_intro"),
            "category_large":  p.get("category_large"),
            "category_medium": p.get("category_medium"),
            "cover_url":       p.get("cover_url", ""),
        }
        for p in reranked_payloads[:RETRIEVE_TOP_N]
    ]

    print(f"\n[RAG] 최종 검색 결과: {len(retrieved_books)}권")
    return {"retrieved_books": retrieved_books, "query_transforms": query_transforms, "hypothetical_doc": hypothetical_doc}


# ## 9. Explain 노드 — 각 도서에 대한 사전 분석

explain_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 AI 어시스턴트입니다.
아래 사용자 프로파일과 책 소개를 읽고,
이 책이 이 사용자에게 왜 적합한지 또는 적합하지 않은지 2문장으로 분석하세요.

[주의]
- 책 소개에 없는 내용은 절대 지어내지 마세요.
- 사용자 프로파일의 독서 목적, 선호 장르, 독서 스타일과 연결해서 작성하세요.

[사용자 프로파일]
{summary}

[책 소개]
{book_intro}

분석:
""")

explain_chain = explain_prompt | llm


def explain_node(state: GraphState) -> dict:
    summary    = state.get("summary", "")
    reflection = state.get("reflection", "")
    books      = state.get("retrieved_books", [])

    user_profile_query = " ".join(filter(None, [summary, reflection]))

    for book in books:
        analysis = explain_chain.invoke({
            "summary":    user_profile_query,
            "book_intro": book.get("book_intro", ""),
        }).content
        book["analysis"] = analysis
        print(f"  [분석] {book.get('title')} → {analysis[:60]}...")

    return {"retrieved_books": books}


# ## 10. RAG LLM 노드 — 추천 이유 생성

rag_prompt = ChatPromptTemplate.from_template("""
당신은 도서관 큐레이터 AI입니다.

[규칙]
- [검색된 도서 목록]의 3권 전부에 대해 추천 이유를 작성하세요. 제외하거나 다른 책으로 대체하지 마세요.
- 반드시 JSON 형식으로만 답하세요. 다른 텍스트는 절대 포함하지 마세요.
- 장르는 반드시 [검색된 도서 목록]의 장르 값을 그대로 사용하세요.

[추천 이유 작성 규칙]
- 반드시 [사전분석]과 [소개]에 나온 구체적인 내용을 근거로 작성하세요.
- 사용자 프로파일의 어떤 부분(독서 목적, 선호 장르, 독서 스타일)과 연결되는지 명시하세요.
- [소개]와 [사전분석]에 없는 내용을 임의로 지어내지 마세요.
- 2~3문장으로 작성하세요.

[사용자 프로파일]
{summary}

[검색된 도서 목록]
{context}

[출력 형식]
[
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "category_medium": "장르", "reason": "추천 이유 2~3문장"}},
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "category_medium": "장르", "reason": "추천 이유 2~3문장"}},
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "category_medium": "장르", "reason": "추천 이유 2~3문장"}}
]
""")


def rag_llm_node(state: GraphState) -> dict:
    summary    = state.get("summary", "")
    reflection = state.get("reflection", "")
    books      = state.get("retrieved_books", [])

    user_profile_query = " ".join(filter(None, [summary, reflection]))

    context = "\n\n".join([
        f"ISBN: {b['isbn']}\n"
        f"제목: {b['title']}\n"
        f"저자: {b['author']}\n"
        f"장르: {b.get('category_medium') or b.get('category_large', '')}\n"
        f"표지URL: {b.get('cover_url', '')}\n"
        f"소개: {(b.get('book_intro') or '')[:300]}\n"
        f"사전분석: {b.get('analysis', '')}"
        for b in books[:3]
    ])

    response = (rag_prompt | llm).invoke({
        "context": context,
        "summary": user_profile_query,
    })

    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = response.content

    return {
        "messages":        [AIMessage(content=response.content)],
        "recommendations": recommendations,
    }
