"""
멀티세션 시뮬레이션 진입점 (jjc).

- multi_session_simulator 기반으로 전환 (LTM 누적, self-eval + PeekaJudge 이중 평가)
- 쿼리 변환 전략 비교: none / step_back / rewrite / decompose / rewrite_decompose
- HyDE RAG 평가 포함 예정
- 페르소나별 멀티세션 (N_SESSIONS 고정)

실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_multi_session_simulator.py --persona A_최재원

더미 페르소나로 빠른 검증:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_multi_session_simulator.py --persona-dir ../backend/data/personas/dummy --persona a

Sweep 실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/run_multi_session_simulator.py --sweep
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from datetime import datetime

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import wandb

from app.config import (
    JUDGE_MODEL,
    LLM_MODEL,
    MAX_TURNS,
    PERSONA_DIR,
    SIMULATION_CHROMA_BASE,
    SIMULATION_RESULTS_DIR,
)
from app.pipeline.graph_test3 import create_app, initial_state
from app.simulation.multi_session_simulator_v2 import run_multi_session
from app.simulation.persona_loader import load_persona_bank
import app.rag.query_transform_v5 as qt_v5


# ── 설정 ─────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "peekabook-crs-multisession-test1"
WANDB_ENTITY  = "jjeong3150-aiffel"
WANDB_TAGS    = ["multi_session", "query_transform_eval"]

N_SESSIONS    = 3   # 페르소나당 세션 수 (테스트 시 작게, 본실험 시 늘릴 것)


# ── Query Transformation 조합 정의 ───────────────────────────────────────────
QUERY_TRANSFORM_CONFIGS = {
    "none":              (False, False, False),
    "step_back":         (True,  False, False),
    "rewrite":           (False, True,  False),
    "decompose":         (False, False, True),
    "rewrite_decompose": (False, True,  True),
    # "hyde":            HyDE RAG — query_transform_v7 구현 후 추가 예정
}


# ── Sweep 설정 ────────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "match_rate/peekajudge", "goal": "maximize"},
    "parameters": {
        "persona_name":    {"values": ["A_최재원", "B_한미영", "C_오민아", "D_이수빈", "E_정미희"]},
        "query_transform": {"values": ["none"]},
    },
}


# ── rag 모듈 설정 ─────────────────────────────────────────────────────────────
def _set_query_transform(query_transform: str) -> None:
    step_back, rewrite, decompose = QUERY_TRANSFORM_CONFIGS[query_transform]
    qt_v6.USE_STEP_BACK = step_back
    qt_v6.USE_REWRITE   = rewrite
    qt_v6.USE_DECOMPOSE = decompose


# ── 단일 페르소나 실행 ────────────────────────────────────────────────────────
def run_for_persona(persona_id:    str,
                    persona_bank:  dict,
                    run_id:        str,
                    query_transform: str,
                    use_genre_filter: bool,
                    args:          argparse.Namespace) -> dict:
    full_persona   = copy.deepcopy(persona_bank[persona_id])  # LTM 격리
    chroma_db_path = f"{SIMULATION_CHROMA_BASE}/{run_id}_{persona_id}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{persona_id}_{query_transform}_{run_id}",
        config={
            "persona_name":      persona_id,
            "query_transform":   query_transform,
            "use_genre_filter":  use_genre_filter,
            "n_sessions":        args.n_sessions,
            "llm_model":         LLM_MODEL,
            "judge_model":       JUDGE_MODEL,
            "max_turns":         MAX_TURNS,
            "collection_name":   os.getenv("QDRANT_COLLECTION_NAME", ""),
            "chroma_db_path":    chroma_db_path,
            "run_id":            run_id,
        },
        tags=WANDB_TAGS,
        reinit=True,
    )

    _set_query_transform(query_transform)

    try:
        result = run_multi_session(
            persona_id=persona_id,
            full_persona=full_persona,
            run_id=run_id,
            create_app_fn=lambda chroma_db_path: create_app(
                chroma_db_path=chroma_db_path,
                use_genre_filter=use_genre_filter,
                rag_module=qt_v6,
            ),
            initial_state=initial_state,
            chroma_base_dir=SIMULATION_CHROMA_BASE,
            judge_model=JUDGE_MODEL,
            n_sessions=args.n_sessions,
            max_turns=MAX_TURNS,
            verbose=args.verbose,
        )
        return result
    finally:
        wandb.finish()
        gc.collect()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    wandb.init(tags=WANDB_TAGS)
    cfg = wandb.config

    bank         = load_persona_bank()
    persona_id   = cfg.persona_name
    full_persona = copy.deepcopy(bank[persona_id])
    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")

    _set_query_transform(cfg.query_transform)

    try:
        run_multi_session(
            persona_id=persona_id,
            full_persona=full_persona,
            run_id=run_id,
            create_app_fn=lambda chroma_db_path: create_app(
                chroma_db_path=chroma_db_path,
                use_genre_filter=False,
                rag_module=qt_v6,
            ),
            initial_state=initial_state,
            chroma_base_dir=SIMULATION_CHROMA_BASE,
            judge_model=JUDGE_MODEL,
            n_sessions=N_SESSIONS,
            max_turns=MAX_TURNS,
            verbose=True,
        )
    finally:
        wandb.finish()
        gc.collect()


# ── 단일 실행 (main) ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PeekaReader 멀티세션 시뮬레이션 (jjc)")
    parser.add_argument("--persona",      type=str, help="페르소나 ID (예: A_최재원)")
    parser.add_argument("--all",          action="store_true", help="전체 페르소나 순회")
    parser.add_argument("--persona-dir",  type=str, default=PERSONA_DIR)
    parser.add_argument("--query-transform", type=str, default="none",
                        choices=list(QUERY_TRANSFORM_CONFIGS.keys()))
    parser.add_argument("--use-genre-filter", action="store_true", default=False)
    parser.add_argument("--n-sessions",   type=int, default=N_SESSIONS)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity",  type=str, default=WANDB_ENTITY)
    parser.add_argument("--sweep",        action="store_true", help="Sweep 실행")
    parser.add_argument("--quiet",        dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.wandb_project)
        wandb.agent(sweep_id, function=run)
        return

    if not args.persona and not args.all:
        parser.error("--persona 또는 --all 중 하나는 필수")

    bank   = load_persona_bank(args.persona_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    persona_ids = list(bank.keys()) if args.all else [args.persona]

    for pid in persona_ids:
        print(f"\n{'#'*60}")
        print(f"# 페르소나: {pid} | query_transform: {args.query_transform}")
        print(f"{'#'*60}")
        try:
            run_for_persona(
                persona_id=pid,
                persona_bank=bank,
                run_id=run_id,
                query_transform=args.query_transform,
                use_genre_filter=args.use_genre_filter,
                args=args,
            )
        except Exception as e:
            import traceback
            print(f"\n[페르소나 {pid} 실패] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
