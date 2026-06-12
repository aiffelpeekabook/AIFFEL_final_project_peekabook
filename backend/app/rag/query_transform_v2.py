from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.config import QDRANT_COLLECTION_NAME
from app.db.qdrant import QdrantDB
from app.embedding.embedder import LocalEmbedder
from app.reranking.reranker import LocalReranker
from app.state.state import GraphState, Phase, UserProfile

load_dotenv()

# ── 모듈 레벨 인스턴스 (첫 import 시 초기화) ──
embedder = LocalEmbedder("BAAI/bge-m3")
db = QdrantDB(vector_size=1024)
llm = ChatOpenAI(model="gpt-4o-mini")
reranker = LocalReranker("BAAI/bge-reranker-v2-m3")


CATEGORY_LIST = [
    "소설", "대학교재/전문서적", "어린이", "수험서/자격증", "시/에세이", "종교", "유아", "만화",
    "사회/정치", "경제/경영", "인문", "예술/대중문화", "국어/외국어", "고등학교 참고서",
    "자기계발", "초등학교 참고서", "건강/취미", "컴퓨터/IT", "역사", "자연/과학",
    "청소년", "중학교 참고서", "가정/요리", "여행", "잡지", "전집", "외국도서"
]

genre_prompt = ChatPromptTemplate.from_template("""
사용자 프로파일을 보고 아래 카테고리 목록에서 적합한 것을 최대 3개 선택하세요.
목록에 없는 값은 절대 반환하지 마세요.

카테고리 목록: {category_list}
사용자 프로파일: {summary}

JSON으로만 반환: {{"categories": ["소설"]}}
""")


def extract_genre_node(state: GraphState) -> dict:
    summary = state.get("summary", "")
    chain = genre_prompt | llm
    response = chain.invoke({"category_list": CATEGORY_LIST, "summary": summary})
    try:
        categories = json.loads(response.content)["categories"]
    except (json.JSONDecodeError, KeyError):
        categories = []
    print(f"추출된 장르: {categories}")
    return {"genre_filter": categories}


# ── Query Transformation ──

step_back_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 AI 어시스턴트입니다.
아래 사용자 프로파일에서 한 단계 물러나,
더 넓은 범위의 도서를 검색할 수 있는 일반적인 쿼리를 생성하세요.
특정 장르나 조건에 국한되지 않고 독서 경험과 목적 중심으로 작성하세요.
2문장 이내로 작성하세요.

사용자 프로파일: {summary}

출력:
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 AI 어시스턴트입니다.
아래 독서 목적을 도서 검색에 더 적합하고 구체적인 검색어로 재작성하세요.
장르, 독자 수준, 분위기 등 검색 정확도를 높일 수 있는 표현을 포함하세요.

독서 목적: {step_back}

재작성된 검색 쿼리 (두 문장 이내로):
""")

decompose_prompt = ChatPromptTemplate.from_template("""
당신은 도서 추천 시스템의 검색 쿼리 전문가입니다.
사용자의 독서 취향과 상황을 깊이 이해하여, 벡터 임베딩 검색에 최적화된 서브쿼리를 생성합니다.

아래 검색 쿼리를 2~4개의 서브쿼리로 분해하세요.
각 서브쿼리는 독립적인 문장으로 작성하되, 전체적으로 자연스럽게 맥락이 이어지도록 하세요.
원래 쿼리의 조건을 충실히 반영하면서, 사용자가 미처 생각하지 못했을 관련 관점을 한 개 포함하세요.

[주의]
- "이 중에서", "그 중에서" 같은 참조 표현은 사용하지 마세요.
- 리뷰, 평점, 사용자 의견 등 도서 메타데이터 외의 정보를 요청하지 마세요.
- "추천 도서", "추천해주세요", "포함해주세요" 같은 표현으로 끝내지 마세요.

검색 쿼리: {rewritten}

출력 형식 (번호와 텍스트만, 다른 텍스트 없이):
1. [서브쿼리 1]
2. [서브쿼리 2]
3. [서브쿼리 3]
""")

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

rag_prompt = ChatPromptTemplate.from_template("""
당신은 도서관 큐레이터 AI입니다.

[규칙]
- 반드시 [검색된 도서 목록]에 있는 책만 추천하세요.
- 반드시 JSON 형식으로만 답하세요. 다른 텍스트는 절대 포함하지 마세요.
- 사용자 프로파일을 참고해서 가장 적합한 도서 3권을 추천하세요.
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
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "cate_depth1": "장르", "reason": "추천 이유 2~3문장"}},
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "cate_depth1": "장르", "reason": "추천 이유 2~3문장"}},
    {{"title": "책 제목", "author": "저자", "isbn": "ISBN번호", "cover_url": "표지URL", "book_intro": "책 소개", "cate_depth1": "장르", "reason": "추천 이유 2~3문장"}}
]
""")


def step_back_query(summary: str) -> str:
    chain = step_back_prompt | llm
    return chain.invoke({"summary": summary}).content.strip()


def rewrite_query(step_back: str) -> str:
    chain = rewrite_prompt | llm
    return chain.invoke({"step_back": step_back}).content.strip()


def decompose_query(rewritten: str) -> list:
    chain = decompose_prompt | llm
    response = chain.invoke({"rewritten": rewritten}).content
    sub_queries = [
        q.strip().lstrip("1234567890. ")
        for q in response.split("\n")
        if q.strip() and q.strip()[0].isdigit()
    ]
    return sub_queries


def get_chained_queries(summary: str) -> dict:
    step_back = step_back_query(summary)
    print(f"  [Step-back]  : {step_back}")

    rewritten = rewrite_query(summary)
    print(f"  [Rewritten]  : {rewritten}")

    sub_queries = decompose_query(rewritten)
    print(f"  [Sub-queries]: {sub_queries}")

    return {
        "step_back": step_back,
        "rewritten": rewritten,
        "sub_queries": sub_queries,
        "all": [step_back, rewritten] + sub_queries,
    }


def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    scores, payloads = {}, {}
    for results in results_list:
        for rank, r in enumerate(results):
            isbn = r.payload.get("isbn", "")
            if isbn:
                scores[isbn] = scores.get(isbn, 0) + 1 / (k + rank + 1)
                payloads[isbn] = r.payload
    return [payloads[isbn] for isbn, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def query_transform_rag_node(state: GraphState) -> dict:
    summary = state.get("summary", "")
    reflection = state.get("reflection", "")
    categories = state.get("genre_filter", [])

    user_profile_query = " ".join(filter(None, [summary, reflection]))

    print("\n[Query Transformations]")
    queries = get_chained_queries(user_profile_query)
    all_queries = queries["all"]

    query_filter = None
    if categories:
        query_filter = Filter(
            must=[FieldCondition(key="cate_depth1", match=MatchAny(any=categories))]
        )

    all_results = []
    for query in all_queries:
        query_vector = embedder.embed(query)
        if query_filter:
            results = db.search_with_filter(
                QDRANT_COLLECTION_NAME, query_vector,
                query_filter=query_filter, limit=5, threshold=0.3,
            )
        else:
            results = db.search(QDRANT_COLLECTION_NAME, query_vector, limit=5, threshold=0.3)
        all_results.append(results)

        print(query)
        print(results)
        # for r in results:
        #     print(f"score: {r.score}")
        #     print(f"title: {r.payload.get('title')}")
        #     print(f"book_intro: {r.payload.get('book_intro')}")
        print("---------------")

    merged_payloads = reciprocal_rank_fusion(all_results)
    reranked_payloads = reranker.rerank(query=user_profile_query, books=merged_payloads)

    retrieved_books = [
        {
            "isbn": p.get("isbn"),
            "title": p.get("title"),
            "author": p.get("author"),
            "book_intro": p.get("book_intro"),
            "cate_depth1": p.get("cate_depth1"),
            "cover_url": p.get("cover_url", ""),
        }
        for p in reranked_payloads[:5]
    ]
    return {"retrieved_books": retrieved_books}


def explain_node(state: GraphState) -> dict:
    summary = state.get("summary", "")
    reflection = state.get("reflection", "")
    books = state["retrieved_books"]
    user_profile_query = " ".join(filter(None, [summary, reflection]))

    for book in books:
        analysis = explain_chain.invoke({
            "summary": user_profile_query,
            "book_intro": book.get("book_intro", ""),
        }).content
        book["analysis"] = analysis
        print(f"  [분석] {book.get('title')} → {analysis[:60]}...")

    return {"retrieved_books": books}


def rag_llm_node(state: GraphState) -> dict:
    summary = state.get("summary", "")
    reflection = state.get("reflection", "")
    books = state["retrieved_books"]
    user_profile_query = " ".join(filter(None, [summary, reflection]))

    if not books:
        msg = "검색된 도서가 없어 추천을 제공할 수 없습니다."
        print(f"\n[rag_llm_node] {msg}")
        return {
            "messages": [AIMessage(content=msg)],
            "recommendations": [],
        }

    context = "\n\n".join([
        f"ISBN: {b['isbn']}\n"
        f"제목: {b['title']}\n"
        f"저자: {b['author']}\n"
        f"장르: {b['cate_depth1']}\n"
        f"표지URL: {b.get('cover_url', '')}\n"
        f"소개: {b['book_intro'][:300]}\n"
        f"사전분석: {b.get('analysis', '')}"
        for b in books
    ])

    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "summary": user_profile_query})

    print(f"\n[rag_llm_node 출력]\n{response.content}\n")

    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = response.content

    return {
        "messages": [AIMessage(content=response.content)],
        "recommendations": recommendations,
    }
