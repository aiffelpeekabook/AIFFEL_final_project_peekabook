"""
CRS × UserSim 오케스트레이션 스크립트 (test6).

test5 대비 변경:
- 쿼리 변환 전략 비교 평가: step_back / rewrite / decompose / rewrite_decompose
- HyDE RAG 평가 포함

실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test6.py

Sweep 실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_simulation_test6.py --sweep
"""
import asyncio
import copy
import gc
import os
import queue
import sys
import threading
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import tiktoken
import wandb
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

from app.pipeline.graph_test3 import create_app, initial_state
from app.simulation.user_sim_v2 import PERSONA_BANK, UserSimAgent, PeekaJudge
import app.rag.query_transform_v5 as qt_v5


# ── Query Transformation 조합 정의 ───────────────────────────────────────────
QUERY_TRANSFORM_CONFIGS = {
    "none":              (False, False, False),
    "step_back":         (True,  False, False),
    "rewrite":           (False, True,  False),
    "decompose":         (False, False, True),
    "rewrite_decompose": (False, True,  True),
}


# ── Sweep 설정 ────────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "mean_score", "goal": "maximize"},
    "parameters": {
        "collection_name":  {"values": ["books_intro_48k"]},
        "persona_name":     {"values": ["A_최재원", "B_한미영", "C_오민아"]},
        "use_genre_filter": {"values": [False]},
        "query_transform":  {"values": ["step_back", "rewrite", "decompose", "rewrite_decompose"]},
        "run_index":        {"values": list(range(1, 10))},
    },
}


# ── 토큰 수 추정 (tiktoken 기반) ─────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def estimate_usersim_tokens(history: list) -> tuple[int, int]:
    """UserSimAgent 대화 히스토리에서 토큰 수 추정.
    Returns: (input_tokens, output_tokens)
    """
    system_tokens = 350  # SYSTEM_PROMPT_TEMPLATE 고정 오버헤드 추정
    input_tokens  = system_tokens + sum(
        _count_tokens(m["content"]) for m in history if m["role"] == "user"
    )
    output_tokens = sum(
        _count_tokens(m["content"]) for m in history if m["role"] == "assistant"
    )
    return input_tokens, output_tokens


def estimate_judge_tokens(persona: dict, book_intros: dict) -> tuple[int, int]:
    """PeekaJudge 평가 토큰 수 추정.
    Returns: (input_tokens, output_tokens)
    """
    persona_str     = "\n".join(f"- {k}: {v}" for k, v in persona.items())
    book_intros_str = "\n\n".join(
        f"📚 {title}\n소개: {intro}" for title, intro in book_intros.items()
    )
    input_tokens  = _count_tokens(persona_str + book_intros_str) + 600  # 프롬프트 템플릿 오버헤드
    output_tokens = 60 * len(book_intros)                                # 책당 ~60 토큰 출력 추정
    return input_tokens, output_tokens


# ── ChromaDB 경로 생성 ────────────────────────────────────────────────────────
def make_chroma_path(persona_id: str, tag: str = "") -> str:
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_key = persona_id.split("_")[0]
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
                "conversation":    agent.get_history(),  # 비용 추정용
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
def log_results(persona: dict, persona_name: str, session_id: str,
                result: dict, judgements: list[dict],
                chroma_db_path: str, run_config: dict,
                token_info: dict | None = None):
    retrieved_books = result["retrieved_books"]
    summary         = result["summary"]
    reflection      = result["reflection"]
    scores          = [1 if j.get("match") else 0 for j in judgements]
    mean_score      = sum(scores) / len(scores) if scores else 0.0
    token_info      = token_info or {}

    table = wandb.Table(columns=[
        "rank", "persona_name", "session_id",
        "title", "author", "category",
        "book_intro", "score", "reason",
        "summary", "reflection",
    ])
    for i, (book, j) in enumerate(zip(retrieved_books, judgements)):
        table.add_data(
            i + 1,
            persona_name,
            session_id,
            book.get("title", ""),
            book.get("author", ""),
            book.get("category_medium") or book.get("category_large", ""),
            book.get("book_intro", ""),
            1 if j.get("match") else 0,
            j.get("reason", ""),
            summary,
            reflection,
        )

    persona_text = "\n".join(f"{k}: {v}" for k, v in persona.items())
    book_scores  = {f"book_{i+1}_score": scores[i] for i in range(len(scores))}

    wandb.log({
        "mean_score":   mean_score,
        "persona_name": persona_name,
        "session_id":   session_id,
        "summary":      summary,
        "reflection":   reflection,
        "persona":      persona_text,
        "chroma_db_path": chroma_db_path,
        **book_scores,
        **token_info,   # 토큰 수 + 구간별/합계 비용
        **run_config,
    })
    wandb.log({"results_table": table})
    return mean_score, scores


# ── 단일 실행 ────────────────────────────────────────────────────────────────
async def main():
    persona_name     = "A_최재원"
    use_genre_filter = True
    query_transform  = "rewrite_decompose"

    persona        = PERSONA_BANK[persona_name]
    chroma_db_path = make_chroma_path(persona_name)

    step_back, rewrite, decompose = QUERY_TRANSFORM_CONFIGS[query_transform]
    qt_v5.USE_STEP_BACK = step_back
    qt_v5.USE_REWRITE   = rewrite
    qt_v5.USE_DECOMPOSE = decompose

    run_config = {
        "persona_name":     persona_name,
        "use_genre_filter": use_genre_filter,
        "query_transform":  query_transform,
        "collection_name":  os.getenv("QDRANT_COLLECTION_NAME", ""),
    }

    wandb.init(
        project="peekabook-crs-test4",
        name=f"single_{persona_name.split('_')[0]}_{query_transform}",
        config={**run_config, "chroma_db_path": chroma_db_path},
    )

    print(f"[ChromaDB 경로] {chroma_db_path}")

    session_id = uuid.uuid4().hex[:8]
    thread_id  = f"single_test4_{session_id}"
    app        = create_app(chroma_db_path=chroma_db_path, use_genre_filter=use_genre_filter,
                            rag_module=qt_v5)

    results = []
    # LangChain(CRS 그래프) 호출 비용은 get_openai_callback으로 추적
    with get_openai_callback() as cb:
        await run_session(app, persona, results, thread_id=thread_id, session_id=session_id)

    if results:
        r = results[0]

        usersim_in, usersim_out = estimate_usersim_tokens(r.get("conversation", []))

        book_intros = {
            f"{b.get('title', '')} | {b.get('author', '')}": b.get("book_intro", "")
            for b in r["retrieved_books"] if b.get("book_intro")
        }
        judgements          = run_judge(persona, r["retrieved_books"])
        judge_in, judge_out = estimate_judge_tokens(persona, book_intros)

        token_info = {
            "crs_input_tokens":      cb.prompt_tokens,
            "crs_output_tokens":     cb.completion_tokens,
            "usersim_input_tokens":  usersim_in,
            "usersim_output_tokens": usersim_out,
            "judge_input_tokens":    judge_in,
            "judge_output_tokens":   judge_out,
            "total_input_tokens":    cb.prompt_tokens + usersim_in + judge_in,
            "total_output_tokens":   cb.completion_tokens + usersim_out + judge_out,
        }

        if judgements:
            mean_score, scores = log_results(
                persona, persona_name, session_id,
                r, judgements, chroma_db_path, run_config,
                token_info=token_info,
            )
            for i, (book, j) in enumerate(zip(r["retrieved_books"], judgements)):
                mark = "O" if j.get("match") else "X"
                print(f"  [{i+1:2d}] {book.get('title', '')} → [{mark}] {j.get('reason', '')}")
            print(f"\n[PeekaJudge] mean_score: {mean_score:.2f}  scores: {scores}")

    print(f"\n[ChromaDB 경로] {chroma_db_path}")
    wandb.finish()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    wandb.init()
    cfg = wandb.config

    qt_v5.QDRANT_COLLECTION_NAME = cfg.collection_name

    step_back, rewrite, decompose = QUERY_TRANSFORM_CONFIGS[cfg.query_transform]
    qt_v5.USE_STEP_BACK = step_back
    qt_v5.USE_REWRITE   = rewrite
    qt_v5.USE_DECOMPOSE = decompose

    persona_name   = cfg.persona_name
    persona        = PERSONA_BANK[persona_name]
    chroma_db_path = make_chroma_path(
        persona_name,
        tag=f"{cfg.collection_name}_{cfg.query_transform}_genre{int(cfg.use_genre_filter)}_{cfg.run_index}"
    )

    run_config = {
        "persona_name":     persona_name,
        "use_genre_filter": cfg.use_genre_filter,
        "query_transform":  cfg.query_transform,
        "collection_name":  cfg.collection_name,
    }

    session_id = uuid.uuid4().hex[:8]
    thread_id  = f"sweep_{persona_name.split('_')[0]}_{cfg.query_transform}_{cfg.run_index}_{session_id}"
    app        = create_app(chroma_db_path=chroma_db_path, use_genre_filter=cfg.use_genre_filter,
                            rag_module=qt_v5)

    results = []
    with get_openai_callback() as cb:
        asyncio.run(run_session(app, persona, results, thread_id, session_id=session_id))

    if results:
        r = results[0]

        usersim_in, usersim_out = estimate_usersim_tokens(r.get("conversation", []))

        book_intros = {
            f"{b.get('title', '')} | {b.get('author', '')}": b.get("book_intro", "")
            for b in r["retrieved_books"] if b.get("book_intro")
        }
        judgements          = run_judge(persona, r["retrieved_books"])
        judge_in, judge_out = estimate_judge_tokens(persona, book_intros)

        token_info = {
            "crs_input_tokens":      cb.prompt_tokens,
            "crs_output_tokens":     cb.completion_tokens,
            "usersim_input_tokens":  usersim_in,
            "usersim_output_tokens": usersim_out,
            "judge_input_tokens":    judge_in,
            "judge_output_tokens":   judge_out,
            "total_input_tokens":    cb.prompt_tokens + usersim_in + judge_in,
            "total_output_tokens":   cb.completion_tokens + usersim_out + judge_out,
        }

        if judgements:
            mean_score, scores = log_results(
                persona, persona_name, session_id,
                r, judgements, chroma_db_path, run_config,
                token_info=token_info,
            )
            print(f"\n[Judge] {persona_name} | {cfg.collection_name} | "
                  f"{cfg.query_transform} | genre={cfg.use_genre_filter} | "
                  f"run {cfg.run_index} → mean: {mean_score:.2f} {scores}")

    wandb.finish()
    gc.collect()


# ── 진입점 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="peekabook-crs-test4")
        wandb.agent(sweep_id, function=run)
    else:
        asyncio.run(main())
