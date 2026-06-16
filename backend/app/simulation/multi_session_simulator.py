"""
멀티세션 시뮬레이션 오케스트레이션

한 페르소나의 N세션을 순서대로 실행하면서:
- 세션별 DNA 추출 → CRS ↔ PeekaReader 대화 → self/judge 평가 → LTM 업데이트
- 페르소나 단위 ChromaDB 격리 (factory + unique path)
- W&B로 세션별 메트릭 로깅

주요 변환:
- Langfuse 제거, W&B 도입 (오케스트레이션 레이어에서만)
- 전역 app → factory (create_app_fn 인자)
- sync app.invoke + Command → async ainvoke + update_state (v4 패턴)
- queue + threading + asyncio 핑퐁
- PERSONA_BANK 전역 의존 제거 → full_persona 명시 주입
"""

from __future__ import annotations

import asyncio
import copy
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

import wandb
from langchain_core.messages import AIMessage, HumanMessage

from app.config import JUDGE_MODEL, KST, MAX_TURNS
from app.simulation.peeka_judge import (
    judge_session,
    update_long_term_memory,
)
from app.simulation.peeka_reader_agent import PeekaReaderAgent, extract_session_dna


# ──────────────────────────────────────────────
# CRS ↔ PeekaReader 핑퐁 (v4 패턴 기반)
# ──────────────────────────────────────────────

def _extract_ai_responses(state: dict[str, Any]) -> list[str]:
    """가장 마지막 user 메시지 이후의 AI 응답들을 시간 순으로 반환함"""
    messages = state.get("messages", [])
    responses = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            break
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            responses.append(msg.content)
    responses.reverse()
    return responses


async def _run_crs(app,
                   thread_id:     str,
                   initial_state: dict,
                   u2c:           queue.Queue,
                   c2u:           queue.Queue) -> None:
    """CRS 그래프를 돌리면서 user input을 queue에서 받아 주입함"""
    session_config = {"configurable": {"thread_id": thread_id}}
    state  = copy.deepcopy(initial_state)
    result = await app.ainvoke(state, config=session_config)

    while True:
        snapshot = app.get_state(session_config)
        if snapshot.next == ():
            c2u.put({"__done__": True, "result": snapshot.values})
            break

        ai_responses = _extract_ai_responses(result)
        if ai_responses:
            c2u.put(ai_responses[-1])

        user_input = u2c.get()
        if user_input is None:  # timeout 신호 → CRS도 종료
            c2u.put({"__done__": True, "result": snapshot.values})
            break

        app.update_state(session_config, {"messages": [HumanMessage(content=user_input)]})
        result = await app.ainvoke(None, config=session_config)


def _run_user_sim(persona_id:  str,
                  persona_dna: dict,
                  collector:   dict,
                  u2c:         queue.Queue,
                  c2u:         queue.Queue,
                  max_turns:   int,
                  verbose:     bool) -> None:
    """PeekaReader 스레드: queue에서 CRS 질문 받고 응답을 보냄"""
    agent = PeekaReaderAgent(persona_id, persona_dna, verbose=verbose)
    collector["agent"]        = agent
    collector["conversation"] = []

    while True:
        message = c2u.get()

        # CRS 종료 신호
        if isinstance(message, dict) and message.get("__done__"):
            collector["crs_result"] = message["result"]
            collector.setdefault("status", "success")
            break

        # max_turns 초과
        if agent.turn_count >= max_turns:
            collector["status"] = "timeout"
            u2c.put(None)  # CRS에 종료 요청
            # CRS의 __done__ 회수
            try:
                m = c2u.get(timeout=10)
                if isinstance(m, dict) and m.get("__done__"):
                    collector["crs_result"] = m["result"]
            except queue.Empty:
                collector["crs_result"] = None
            break

        # 정상 응답
        ans = agent.answer(str(message))
        collector["conversation"].append({
            "turn":    agent.turn_count,
            "csr":     str(message),
            "thought": ans["thought"],
            "user":    ans["utterance"],
        })
        u2c.put(ans["utterance"])


# ──────────────────────────────────────────────
# 한 세션 실행
# ──────────────────────────────────────────────

async def run_session(app,
                      initial_state: dict,
                      persona_id:    str,
                      session_id:    int,
                      persona_dna:   dict,
                      *,
                      max_turns: int  = MAX_TURNS,
                      verbose:   bool = True) -> dict:
    """
    페르소나 한 세션을 끝까지 실행함 (CRS ↔ PeekaReader 핑퐁).
    추천 종료 후 PeekaReader self-evaluation까지 수행.

    Args:
        app:           create_app_fn(...)으로 생성된 컴파일된 LangGraph
        initial_state: CRS 초기 state dict (graph 모듈에서 import)
        persona_id, session_id: 식별자 (thread_id 생성용)
        persona_dna:   extract_session_dna() 결과
        max_turns:     PeekaReader 답변 최대 횟수 (timeout 안전장치)
    """
    u2c: queue.Queue = queue.Queue()
    c2u: queue.Queue = queue.Queue()
    collector: dict  = {"status": "running"}

    thread_id = f"sim_{persona_id}_s{session_id}_{int(time.time())}"
    start     = time.time()

    t = threading.Thread(
        target=_run_user_sim,
        args=(persona_id, persona_dna, collector, u2c, c2u, max_turns, verbose),
        daemon=True,
    )
    t.start()

    try:
        await _run_crs(app, thread_id, initial_state, u2c, c2u)
    except Exception as e:
        collector["status"] = f"error: {e}"
        if verbose:
            print(f"\n[오류] CRS 실행 실패: {e}")
    finally:
        t.join(timeout=30)

    elapsed    = round(time.time() - start, 2)
    agent      = collector.get("agent")
    crs_result = collector.get("crs_result")
    status     = collector.get("status", "unknown")

    # PeekaReader self-evaluation
    self_evaluation     = None
    book_intros: dict   = {}
    recommendation_text = None

    if status == "success" and crs_result and agent is not None:
        messages = crs_result.get("messages", [])
        if messages:
            last = messages[-1]
            recommendation_text = last.content if hasattr(last, "content") else str(last)

        # book_intros 추출
        try:
            recommendations = crs_result.get("recommendations", [])
            retrieved = {
                b["isbn"]: b
                for b in crs_result.get("retrieved_books", [])
                if b.get("isbn")
            }
            for rec in recommendations:
                isbn  = rec.get("isbn", "")
                title = rec.get("title", "")
                if isbn in retrieved and retrieved[isbn].get("book_intro"):
                    book_intros[title] = retrieved[isbn]["book_intro"]
        except Exception as e:
            if verbose:
                print(f"  [book_intro 추출 실패] {e}")

        if verbose:
            print(f"  [book_intro {'로드' if book_intros else '없음 (fallback)'}] "
                  f"{len(book_intros)}권")

        if recommendation_text:
            self_evaluation = agent.evaluate(recommendation_text, book_intros)

    return {
        "persona_id":          persona_id,
        "session_id":          session_id,
        "status":              status,
        "response_time_sec":   elapsed,
        "total_turns":         agent.turn_count if agent else 0,
        "conversation":        collector.get("conversation", []),
        "recommendation_text": recommendation_text,
        "retrieved_books":     crs_result.get("retrieved_books", []) if crs_result else [],
        "recommendations":     crs_result.get("recommendations", []) if crs_result else [],
        "self_evaluation":     self_evaluation,
        "eval_mode":           "book_intro" if book_intros else "fallback",
        "book_intro_loaded":   len(book_intros),
        "simulated_at":        datetime.now(tz=KST).isoformat(),
    }


# ──────────────────────────────────────────────
# 멀티세션 (한 페르소나 전체)
# ──────────────────────────────────────────────

# verdict를 W&B numeric chart용으로 인코딩 (높을수록 만족)
VERDICT_CODES = {
    "satisfied":      4,
    "partial":        3,
    "unsatisfied":    2,
    "too_hard":       1,
    "genre_mismatch": 1,
    "duplicate":      1,
}


def _safe_wandb_log(data: dict, step: Optional[int] = None) -> None:
    """wandb.run이 None이면 no-op. 진입점에서 init 안 한 경우도 안전 동작."""
    if wandb.run is None:
        return
    if step is not None:
        wandb.log(data, step=step)
    else:
        wandb.log(data)


def run_multi_session(persona_id:    str,
                     full_persona:  dict,
                     run_id:        str,
                     create_app_fn: Callable,
                     initial_state: dict,
                     *,
                     chroma_base_dir: str = "backend/chroma_db_runs",
                     judge_model:     str = JUDGE_MODEL,
                     n_sessions:      Optional[int] = None,
                     max_turns:       int  = MAX_TURNS,
                     verbose:         bool = True) -> dict:
    """
    한 페르소나의 N세션을 순서대로 실행하면서 매 세션:
      1) extract_session_dna       (고정속성 + 세션DNA + 누적취향)
      2) run_session               (CRS ↔ PeekaReader 핑퐁 + self-eval)
      3) judge_session             (PeekaJudge 독립 평가)
      4) update_long_term_memory   (규칙 기반 verdict + LTM 누적)
      5) wandb.log                 (세션별 메트릭)

    Factory + unique path로 ChromaDB는 페르소나 단위 격리됨.

    Args:
        persona_id:      예) "A_최재원"
        full_persona:    PERSONA_BANK[persona_id] dict
        run_id:          실행 식별자 (보통 timestamp). 같은 페르소나라도 run마다 chroma 격리
        create_app_fn:   factory 함수 (예: graph_test2.create_app)
        initial_state:   graph 모듈에서 import한 CRS 초기 state
        judge_model:     Judge 모델. default는 Claude Haiku 4.5. sweep으로 모델 비교 가능
        n_sessions:      None이면 페르소나의 모든 세션 실행
    """
    total = len(full_persona["sessions"])
    if n_sessions is not None:
        total = min(total, n_sessions)

    chroma_db_path = f"{chroma_base_dir}/{run_id}_{persona_id}"
    app = create_app_fn(chroma_db_path=chroma_db_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"멀티세션 시작: {persona_id} ({total} 세션)")
        print(f"ChromaDB:    {chroma_db_path}")
        print(f"Judge model: {judge_model}")
        print(f"{'='*60}")

    sessions_log: list = []
    table_rows:   list = []

    for session_spec in full_persona["sessions"][:total]:
        session_id = session_spec["session_id"]

        if verbose:
            memory = full_persona["long_term_memory"]
            print(f"\n{'─'*60}")
            print(f"[세션 {session_id}/{total}] {session_spec.get('preferred_genre', '')}")
            print(f"  [누적 취향] {memory['derived_preferences']}")
            print(f"  [이전 추천] {len(memory['previously_recommended'])}권")
            print(f"{'─'*60}")

        # 1) session DNA 추출
        session_dna = extract_session_dna(full_persona, session_id)

        # 2) 한 세션 실행 (async)
        try:
            session_result = asyncio.run(run_session(
                app=app,
                initial_state=initial_state,
                persona_id=persona_id,
                session_id=session_id,
                persona_dna=session_dna,
                max_turns=max_turns,
                verbose=verbose,
            ))
        except Exception as e:
            print(f"  [오류] run_session 실패: {e}")
            sessions_log.append({"session_id": session_id, "status": f"error: {e}"})
            continue

        # 3) PeekaJudge 평가
        judge_result: dict = {}
        if session_result.get("status") == "success":
            try:
                judge_result = judge_session(
                    session_result=session_result,
                    persona=session_dna,
                    stage="peekajudge",
                    model=judge_model,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [오류] judge_session 실패: {e}")

        # 4) LTM 업데이트
        verdict = None
        if session_result.get("status") == "success":
            try:
                update_long_term_memory(
                    full_persona=full_persona,
                    session_id=session_id,
                    session_dna=session_dna,
                    session_result=session_result,
                    judge_result=judge_result,
                )
                history = full_persona["long_term_memory"]["feedback_history"]
                verdict = history[-1]["verdict"] if history else None
            except Exception as e:
                print(f"  [오류] update_long_term_memory 실패: {e}")

        # 메트릭 계산
        self_eval  = session_result.get("self_evaluation") or {}
        self_books = self_eval.get("books_evaluated", [])
        self_match_rate = (
            sum(1 for b in self_books if b.get("match")) / len(self_books)
            if self_books else 0.0
        )
        judge_match_rate = judge_result.get("book_match_rate", 0.0)

        ltm       = full_persona["long_term_memory"]
        latest_fb = ltm["feedback_history"][-1] if ltm["feedback_history"] else {}

        # 5) W&B log per session
        _safe_wandb_log({
            "match_rate/peekareader_self":  self_match_rate,
            "match_rate/peekajudge":        judge_match_rate,
            "verdict_code":                 VERDICT_CODES.get(verdict, 0),
            "verdict":                      verdict or "unknown",
            "n_difficulty_mismatch":        len(latest_fb.get("difficulty_mismatch", [])),
            "n_genre_mismatch":             len(latest_fb.get("genre_mismatch", [])),
            "n_duplicates":                 len(latest_fb.get("duplicates", [])),
            "derived_prefs_count":          len(ltm["derived_preferences"]),
            "previously_recommended_count": len(ltm["previously_recommended"]),
            "turn_count":                   session_result.get("total_turns", 0),
            "response_time_sec":            session_result.get("response_time_sec", 0.0),
            "status_success":               1 if session_result.get("status") == "success" else 0,
        }, step=session_id)

        # session log 누적 (multi_sim_logger.py에서 저장)
        sessions_log.append({
            "session_id":       session_id,
            "preferred_genre":  session_spec.get("preferred_genre", ""),
            "status":           session_result.get("status", "unknown"),
            "self_match_rate":  self_match_rate,
            "judge_match_rate": judge_match_rate,
            "verdict":          verdict,
            "conversation":     session_result.get("conversation", []),
            "recommendations":  session_result.get("recommendations", []),
        })

        # table row (마지막에 한꺼번에 W&B Table로 log)
        recs_titles = ", ".join(
            r.get("title", "") for r in session_result.get("recommendations", [])
        )
        table_rows.append([
            session_id,
            session_spec.get("preferred_genre", "")[:30],
            session_result.get("status", "unknown"),
            round(self_match_rate, 2),
            round(judge_match_rate, 2),
            verdict or "—",
            recs_titles[:120],
        ])

    # 페르소나 종료 — W&B Table 한 번에 log
    if wandb.run is not None:
        table = wandb.Table(
            columns=["session_id", "preferred_genre", "status",
                     "self_match_rate", "judge_match_rate", "verdict", "recommendations"],
            data=table_rows,
        )
        wandb.log({"sessions_detail": table})

    # 요약 출력
    if verbose:
        print(f"\n{'='*60}")
        print(f"멀티세션 완료: {persona_id}")
        print(f"{'='*60}")
        for s in sessions_log:
            jmr  = s.get("judge_match_rate")
            rate = f"{jmr:.0%}" if jmr is not None else "N/A"
            print(f"  세션 {s['session_id']:2d} | "
                  f"{s.get('preferred_genre', '')[:20]:20s} | "
                  f"judge: {rate:>5s} | verdict: {s.get('verdict') or '—'}")

    return {
        "persona_id":     persona_id,
        "total_sessions": total,
        "sessions":       sessions_log,
        "final_memory":   copy.deepcopy(full_persona["long_term_memory"]),
        "completed_at":   datetime.now(tz=KST).isoformat(),
    }
