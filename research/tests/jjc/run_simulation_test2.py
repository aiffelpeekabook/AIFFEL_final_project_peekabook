"""
CRS × UserSim 오케스트레이션 스크립트 (test2).

test1 대비 변경:
- user_sim_v2 사용: DNA 페르소나(PERSONA_BANK) + speaking_style 반영 UserSimAgent
- llm_judge → PeekaJudge (루브릭 기반 5축 평가)
- ChromaDB 경로를 명시적으로 지정 (chroma_db_runs/{timestamp}_{persona_key})

실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test2.py

Sweep 실행 (2 collection × 3 persona × 10 반복 = 60 runs):
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test2.py --sweep
"""
import asyncio
import copy
import json
import os
import queue
import sys
import threading
import uuid
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import wandb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

from app.pipeline.graph_test1 import create_app, initial_state
from app.simulation.user_sim_v2 import PERSONA_BANK, UserSimAgent, PeekaJudge


# ── Sweep 설정 ───────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "mean_score", "goal": "maximize"},
    "parameters": {
        "collection_name": {"values": ["books_intro_48k", "books_merged_48k"]},
        "persona_name":    {"values": ["A_최재원", "B_한미영", "C_오민아"]},
        "run_index":       {"values": list(range(1, 2))},
    },
}


# ── ChromaDB 경로 생성 ────────────────────────────────────────────────────────
def make_chroma_path(persona_id: str, tag: str = "") -> str:
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_key = persona_id.split("_")[0]   # "A_최재원" → "A"
    suffix      = f"_{tag}" if tag else ""
    return os.path.join(
        os.path.dirname(__file__), "../../../backend/chroma_db_runs",
        f"{timestamp}_{persona_key}{suffix}"
    )


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
    t = threading.Thread(target=run_user_sim, args=(persona, results, u2c, c2u), daemon=True)
    t.start()
    await run_crs(app, thread_id, u2c, c2u, session_id=session_id)
    t.join()


# ── PeekaJudge 평가 ───────────────────────────────────────────────────────────
def run_judge(persona: dict, retrieved_books: list) -> list[dict]:
    book_intros = {
        f"{b.get('title', '')} | {b.get('author', '')}": b.get("book_intro", "")
        for b in retrieved_books
        if b.get("book_intro")
    }

    if not book_intros:
        return []

    judge  = PeekaJudge(verbose=True)
    result = judge.evaluate(persona, book_intros)
    return result.get("books_evaluated", [])


# ── wandb 로깅 ───────────────────────────────────────────────────────────────
def log_results(persona: dict, result: dict, judgements: list[dict], chroma_db_path: str):
    retrieved_books = result["retrieved_books"]
    summary         = result["summary"]
    reflection      = result["reflection"]
    scores          = [1 if j.get("match") else 0 for j in judgements]
    mean_score      = sum(scores) / len(scores) if scores else 0.0

    table = wandb.Table(columns=["rank", "title", "author", "category", "book_intro", "score", "reason", "summary", "reflection"])
    for i, (book, j) in enumerate(zip(retrieved_books, judgements)):
        table.add_data(
            i + 1,
            book.get("title", ""),
            book.get("author", ""),
            book.get("category_medium") or book.get("category_large", ""),
            book.get("book_intro", "")[:300],
            1 if j.get("match") else 0,
            j.get("reason", ""),
            summary,
            reflection,
        )

    persona_text = "\n".join(f"{k}: {v}" for k, v in persona.items())

    wandb.log({
        "mean_score":      mean_score,
        "book_1_score":    scores[0] if len(scores) > 0 else None,
        "book_2_score":    scores[1] if len(scores) > 1 else None,
        "book_3_score":    scores[2] if len(scores) > 2 else None,
        "summary":         summary,
        "reflection":      reflection,
        "persona":         persona_text,
        "chroma_db_path":  chroma_db_path,
    })
    wandb.log({"results_table": table})
    return mean_score, scores


# ── 단일 실행 ────────────────────────────────────────────────────────────────
async def main():
    persona_name   = "A_최재원"
    persona        = PERSONA_BANK[persona_name]
    chroma_db_path = make_chroma_path(persona_name)

    wandb.init(
        project="peekabook-crs-test2",
        name=f"single_{persona_name.split('_')[0]}",
        config={
            "persona":         persona_name,
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME", ""),
            "chroma_db_path":  chroma_db_path,
        },
    )

    print(f"[ChromaDB 경로] {chroma_db_path}")

    session_id = uuid.uuid4().hex[:8]
    thread_id  = f"single_test2_{session_id}"
    app        = create_app(chroma_db_path=chroma_db_path)

    results = []
    await run_session(app, persona, results, thread_id=thread_id, session_id=session_id)

    if results:
        r          = results[0]
        judgements = run_judge(persona, r["retrieved_books"])
        if judgements:
            mean_score, scores = log_results(persona, r, judgements, chroma_db_path)
            for i, (book, j) in enumerate(zip(r["retrieved_books"], judgements)):
                mark = "O" if j.get("match") else "X"
                print(f"  [{i+1}] {book.get('title', '')} → [{mark}] {j.get('reason', '')}")
            print(f"\n[PeekaJudge] mean_score: {mean_score:.2f}")

    print(f"\n[ChromaDB 경로] {chroma_db_path}")
    wandb.finish()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    import app.rag.query_transform_v3 as qt_v3

    wandb.init()
    cfg = wandb.config

    qt_v3.QDRANT_COLLECTION_NAME = cfg.collection_name

    persona_name   = cfg.persona_name
    persona        = PERSONA_BANK[persona_name]
    chroma_db_path = make_chroma_path(persona_name, tag=f"{cfg.collection_name}_{cfg.run_index}")

    session_id = uuid.uuid4().hex[:8]
    thread_id  = f"sweep_{persona_name.split('_')[0]}_{cfg.collection_name}_{cfg.run_index}_{session_id}"
    app        = create_app(chroma_db_path=chroma_db_path)

    results = []
    asyncio.run(run_session(app, persona, results, thread_id, session_id=session_id))

    if results:
        r          = results[0]
        judgements = run_judge(persona, r["retrieved_books"])
        if judgements:
            mean_score, scores = log_results(persona, r, judgements, chroma_db_path)
            print(f"\n[Judge] {persona_name} | {cfg.collection_name} | run {cfg.run_index} → mean: {mean_score:.2f} {scores}")

    wandb.finish()


# ── 진입점 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="peekabook-crs-test2")
        wandb.agent(sweep_id, function=run)
    else:
        asyncio.run(main())
