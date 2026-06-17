"""
시뮬레이션 결과 저장 및 요약 출력.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from app.config import KST, LLM_MODEL, MAX_TURNS, SIMULATION_RESULTS_DIR


def save_simul_result(result:   dict,
                             save_dir: str = SIMULATION_RESULTS_DIR) -> str:
    """
    멀티세션 시뮬레이션 결과(multi_session_simulator 반환값)를 JSON으로 저장함.

    파일명: simul_{persona_id}_{timestamp}.json
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    persona_id = result["persona_id"]
    timestamp  = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")
    filename   = f"simul_{persona_id}_{timestamp}.json"
    filepath   = os.path.join(save_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[저장 완료] {filepath}")
    return filepath


def save_results(results:  list,
                 save_dir: str = SIMULATION_RESULTS_DIR,
                 filename: str = None) -> str:
    """
    단일세션 결과 list(run_session들의 list)를 JSON으로 저장함.

    metadata로 model, max_turns, version 등을 함께 기록.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")
        filename = f"sim_results_{ts}.json"

    output = {
        "metadata": {
            "version":      "v2",
            "simulated_at": datetime.now(tz=KST).isoformat(),
            "llm_model":    LLM_MODEL,
            "max_turns":    MAX_TURNS,
            "note":         "PeekaReader self-eval + PeekaJudge 독립평가 (Claude Haiku 4.5).",
        },
        "results": results,
    }

    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[저장 완료] {filepath}")
    return filepath


def print_summary(results: list) -> None:
    """
    단일세션 결과 list에서 페르소나별 매치율 요약을 콘솔에 출력함.

    멀티세션 결과의 경우 run_multi_session에서 이미 콘솔에 요약 출력됨 — 단일세션 list 용.
    """
    print("\n" + "=" * 60)
    print("시뮬레이션 결과 요약")
    print("=" * 60)

    for r in results:
        ev      = r.get("self_evaluation") or r.get("evaluation") or {}
        books   = ev.get("books_evaluated", [])
        matched = sum(1 for b in books if b.get("match"))
        total   = len(books)

        print(f"\n[{r.get('persona_id', '?')}] "
              f"{r.get('status', '?')} | "
              f"{r.get('total_turns', 0)}턴 | "
              f"{r.get('response_time_sec', 0)}초")

        if books:
            print(f"  도서 일치: {matched}/{total}권 ({matched / total:.0%})")
            print(f"  평가 모드: {r.get('eval_mode', '-')} "
                  f"(book_intro {r.get('book_intro_loaded', 0)}권)")
            print(f"  총평: {ev.get('overall_reason', '-')}")
        elif r.get("status") == "timeout":
            print(f"  슬롯 채우기 {MAX_TURNS}턴 초과 — 추천 도달 못 함")
        elif str(r.get("status", "")).startswith("error"):
            print(f"  오류: {r['status']}")


def print_multi_session_summary(results: list) -> None:
    """
    여러 페르소나의 run_multi_session 결과 list 요약 출력.

    각 페르소나의 평균 judge_match_rate, final verdict 분포 등.
    """
    print("\n" + "=" * 60)
    print("멀티세션 시뮬레이션 종합 요약")
    print("=" * 60)

    for r in results:
        pid      = r.get("persona_id", "?")
        sessions = r.get("sessions", [])

        if not sessions:
            print(f"\n[{pid}] 세션 없음")
            continue

        judge_rates = [
            s.get("judge_match_rate", 0.0)
            for s in sessions
            if s.get("judge_match_rate") is not None
        ]
        avg_judge = sum(judge_rates) / len(judge_rates) if judge_rates else 0.0

        verdicts = [s.get("verdict") for s in sessions if s.get("verdict")]
        verdict_counts: dict = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        successful = sum(1 for s in sessions if s.get("status") == "success")

        print(f"\n[{pid}] {successful}/{len(sessions)}세션 성공")
        print(f"  평균 judge match_rate: {avg_judge:.0%}")
        print(f"  verdict 분포: {dict(sorted(verdict_counts.items()))}")
