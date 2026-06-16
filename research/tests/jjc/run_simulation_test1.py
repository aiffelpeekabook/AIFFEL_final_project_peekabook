"""
books_intro_48k vs books_merged_48k 비교 실험.

RAG 검색 결과(retrieved_books 3권)를 LLM Judge로 1(좋음)/0(안 좋음) 이진 평가.
explain/rag_llm/api_tool_calling 노드를 제거해 토큰 절약.

단일 실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test1.py

Sweep 실행 (2 collection × 3 persona × 10 반복 = 60 runs):
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test1.py --sweep
"""
import asyncio
import copy
import json
import os
import queue
import re
import sys
import tempfile
import threading
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import wandb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

from app.pipeline.graph_test1 import create_app, initial_state
from app.simulation.user_sim import PERSONA_TEMPLATES, UserSimAgent


# ── Sweep 설정 ───────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "mean_score", "goal": "maximize"},
    "parameters": {
        "collection_name": {"values": ["books_intro_48k", "books_merged_48k"]},
        "persona_name":    {"values": ["중년_역사_비문학", "직장인_SF팬", "대학생_문학팬"]},
        "run_index":       {"values": list(range(1, 11))},
    },
}


# ── 시뮬레이션 헬퍼 ──────────────────────────────────────────────────────────
def _extract_ai_responses(state: dict) -> list[str]:
    responses = []
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            break
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            responses.append(msg.content)
    responses.reverse()
    return responses


async def run_crs(app, thread_id: str, user_to_crs: queue.Queue, crs_to_user: queue.Queue, session_id: str = None):
    session_config = {"configurable": {"thread_id": thread_id}}
    state = copy.deepcopy(initial_state)
    if session_id:
        state["session_id"] = session_id
    result = await app.ainvoke(state, config=session_config)

    while True:
        snapshot = app.get_state(session_config)
        if snapshot.next == ():
            crs_to_user.put({"__done__": True, "result": snapshot.values})
            break
        ai_responses = _extract_ai_responses(result)
        if ai_responses:
            crs_to_user.put(ai_responses[-1])
        user_input = user_to_crs.get()
        app.update_state(session_config, {"messages": [HumanMessage(content=user_input)]})
        result = await app.ainvoke(None, config=session_config)


def run_user_sim(persona: dict, result_collector: list, user_to_crs: queue.Queue, crs_to_user: queue.Queue):
    agent = UserSimAgent(persona=persona, verbose=True)
    while True:
        message = crs_to_user.get()
        if isinstance(message, dict) and message.get("__done__"):
            crs_result = message["result"]
            result_collector.append({
                "retrieved_books": crs_result.get("retrieved_books", []),
                "summary":         crs_result.get("summary", ""),
                "reflection":      crs_result.get("reflection", ""),
            })
            break
        user_to_crs.put(agent.answer(message))


async def run_session(app, persona: dict, results: list, thread_id: str, session_id: str = None):
    u2c: queue.Queue = queue.Queue()
    c2u: queue.Queue = queue.Queue()
    t = threading.Thread(target=run_user_sim, args=(persona, results, u2c, c2u))
    t.start()
    await run_crs(app, thread_id, u2c, c2u, session_id=session_id)
    t.join()


# ── LLM Judge ────────────────────────────────────────────────────────────────
def llm_judge(persona: dict, retrieved_books: list, summary: str, reflection: str) -> list[dict]:
    """책 3권 각각에 대해 1(좋음) / 0(안 좋음) 이진 평가."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if not retrieved_books:
        return [{"score": 0, "reason": "검색 결과 없음"}]

    books_text = "\n".join([
        f"{i+1}. {b.get('title', '')} / {b.get('author', '')} "
        f"[{b.get('category_medium') or b.get('category_large', '')}]\n"
        f"   소개: {b.get('book_intro', '')[:200]}"
        for i, b in enumerate(retrieved_books)
    ])

    persona_text = "\n".join(f"- {k}: {v}" for k, v in persona.items())
    profile_text = " ".join(filter(None, [summary, reflection]))

    prompt = f"""당신은 도서 추천 시스템의 평가자입니다.
아래 페르소나와 사용자 프로파일을 보고, 검색된 도서 각각이 이 사용자에게 적합한지 평가하세요.

[페르소나]
{persona_text}

[사용자 프로파일 (summary + reflection)]
{profile_text}

[검색된 도서]
{books_text}

평가 기준:
1. 페르소나의 선호 장르·분위기와 일치하는가
2. 이미 읽은 책과 중복되지 않는가
3. 사용자 프로파일과 실제로 연결되는가

각 책에 대해 1(좋음) 또는 0(안 좋음)으로만 판정하세요.
아래 JSON 배열 형식으로만 답하세요 (책 순서 유지):
[
    {{"score": 1 또는 0, "reason": "판정 근거 1문장"}},
    {{"score": 1 또는 0, "reason": "판정 근거 1문장"}},
    {{"score": 1 또는 0, "reason": "판정 근거 1문장"}}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    match = re.search(r"\[.*\]", response.content, re.DOTALL)
    if match:
        try:
            results = json.loads(match.group())
            return [{"score": int(r["score"]), "reason": r["reason"]} for r in results]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return [{"score": 0, "reason": "파싱 실패"}] * len(retrieved_books)


# ── wandb 로깅 ───────────────────────────────────────────────────────────────
def log_results(persona: dict, result: dict, judgements: list[dict]):
    retrieved_books = result["retrieved_books"]
    summary = result["summary"]
    reflection = result["reflection"]
    scores = [j["score"] for j in judgements]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    table = wandb.Table(columns=["rank", "title", "author", "category", "book_intro", "score", "reason"])
    for i, (book, j) in enumerate(zip(retrieved_books, judgements)):
        table.add_data(
            i + 1,
            book.get("title", ""),
            book.get("author", ""),
            book.get("category_medium") or book.get("category_large", ""),
            book.get("book_intro", "")[:300],
            j["score"],
            j["reason"],
        )

    persona_text = "\n".join(f"{k}: {v}" for k, v in persona.items())

    wandb.log({
        "mean_score":   mean_score,
        "book_1_score": scores[0] if len(scores) > 0 else None,
        "book_2_score": scores[1] if len(scores) > 1 else None,
        "book_3_score": scores[2] if len(scores) > 2 else None,
        "summary":      summary,
        "reflection":   reflection,
        "persona":      persona_text,
    })
    wandb.log({"results_table": table})
    return mean_score, scores


# ── 단일 실행 ────────────────────────────────────────────────────────────────
async def main():
    persona_name = "중년_역사_비문학"
    persona = PERSONA_TEMPLATES[persona_name]

    wandb.init(
        project="peekabook-crs-test1",
        name=f"single_{persona_name}",
        config={
            "persona":         persona_name,
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME", ""),
        },
    )

    session_id = uuid.uuid4().hex[:8]
    thread_id = f"single_test1_{session_id}"
    app = create_app(chroma_db_path=tempfile.mkdtemp())

    results = []
    await run_session(app, persona, results, thread_id=thread_id, session_id=session_id)

    if results:
        r = results[0]
        judgements = llm_judge(persona, r["retrieved_books"], r["summary"], r["reflection"])
        mean_score, scores = log_results(persona, r, judgements)
        for i, (book, j) in enumerate(zip(r["retrieved_books"], judgements)):
            print(f"  [{i+1}] {book.get('title', '')} → {j['score']} ({j['reason']})")
        print(f"\n[LLM Judge] mean_score: {mean_score:.2f}")

    wandb.finish()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    import app.rag.query_transform_v3 as qt_v3

    wandb.init()
    cfg = wandb.config

    # collection 교체 (monkey-patch)
    qt_v3.QDRANT_COLLECTION_NAME = cfg.collection_name

    persona = PERSONA_TEMPLATES[cfg.persona_name]
    session_id = uuid.uuid4().hex[:8]
    thread_id = f"sweep_{cfg.persona_name}_{cfg.collection_name}_{cfg.run_index}_{session_id}"
    app = create_app(chroma_db_path=tempfile.mkdtemp())

    results = []
    asyncio.run(run_session(app, persona, results, thread_id, session_id=session_id))

    if results:
        r = results[0]
        judgements = llm_judge(persona, r["retrieved_books"], r["summary"], r["reflection"])
        mean_score, scores = log_results(persona, r, judgements)
        print(f"\n[Judge] {cfg.persona_name} | {cfg.collection_name} | run {cfg.run_index} → mean: {mean_score:.2f} {scores}")

    wandb.finish()


# ── 진입점 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="peekabook-crs-test1")
        wandb.agent(sweep_id, function=run)
    else:
        asyncio.run(main())
