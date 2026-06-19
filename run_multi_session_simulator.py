"""
멀티세션 시뮬레이션 진입점 (jjc).

- multi_session_simulator 기반으로 전환 (LTM 누적, self-eval + PeekaJudge 이중 평가)
- 쿼리 변환 전략 비교: none / step_back / rewrite / decompose / rewrite_decompose / hyde
- HyDE RAG 평가 포함 (graph_test5, use_hyde=True)
- 페르소나별 멀티세션 (N_SESSIONS 고정)

실행:
    cd /home/jjeong3150/work/peekabook
    python run_multi_session_simulator.py --persona A_최재원

더미 페르소나로 빠른 검증:
    cd /home/jjeong3150/work/peekabook
    python run_multi_session_simulator.py --persona-dir backend/data/personas/dummy --persona a

Sweep 실행:
    cd /home/jjeong3150/work/peekabook
    python run_multi_session_simulator.py --sweep
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from datetime import datetime

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = HERE  # 레포 루트에 위치
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import tiktoken
import wandb
from langchain_community.callbacks import get_openai_callback

from app.config import (
    JUDGE_MODEL,
    LLM_MODEL,
    MAX_TURNS,
    SIMULATION_CHROMA_BASE,
)
from app.pipeline.graph_test5 import create_app, initial_state
from app.simulation.multi_session_simulator_v2 import run_multi_session
from app.simulation.persona_loader import load_persona_bank
import app.rag.query_transform_v5 as qt_v5


# ── 토큰 추정 ────────────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _estimate_usersim_tokens(sessions: list) -> tuple[int, int]:
    """세션 대화 기록에서 UserSim 토큰 추정.

    대화 포맷: [{turn, crs(CRS발화→UserSim input), thought, user(UserSim발화→output)}]
    system prompt는 세션당 350토큰으로 고정 추정.
    """
    system_tokens = 350
    total_in  = system_tokens * len(sessions)
    total_out = 0
    for s in sessions:
        for turn in s.get("conversation", []):
            total_in  += _count_tokens(turn.get("crs", ""))
            total_out += _count_tokens(turn.get("user", ""))
    return total_in, total_out


def _estimate_judge_tokens(sessions: list) -> tuple[int, int]:
    """세션 retrieved_books에서 Judge 토큰 추정.

    persona + system prompt를 세션당 600토큰으로 고정 추정.
    """
    total_in  = 0
    total_out = 0
    for s in sessions:
        books = s.get("retrieved_books", [])
        book_intros = {
            f"{b.get('title', '')} | {b.get('author', '')}": b.get("book_intro", "")
            for b in books if b.get("book_intro")
        }
        intros_str = "\n\n".join(
            f"📚 {title}\n소개: {intro}" for title, intro in book_intros.items()
        )
        total_in  += _count_tokens(intros_str) + 600
        total_out += 60 * len(book_intros)
    return total_in, total_out


# ── 설정 ─────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "peekabook-crs-multisession-test1"
WANDB_ENTITY  = "jjeong3150-aiffel"
WANDB_TAGS    = ["multi_session", "query_transform_eval"]

N_SESSIONS    = 3   # 페르소나당 세션 수 (테스트 시 작게, 본실험 시 늘릴 것)


# ── Query Transformation 조합 정의 ───────────────────────────────────────────
# 튜플: (use_original, use_step_back, use_rewrite, use_decompose, use_hyde)
# True인 항목의 쿼리가 all_queries에 포함되어 검색됨
# 조합 예: "none_step_back": (True, True, False, False, False)
QUERY_TRANSFORM_CONFIGS = {
    "none":      (True,  False, False, False, False),
    "step_back": (False, True,  False, False, False),
    "rewrite":   (False, False, True,  False, False),
    "decompose": (False, False, False, True,  False),
    "hyde":      (False, False, False, False, True),
    # "none_decompose_hyde": (True, False, False, True, True),    # 조합 예시
    # "step_back_rewrite": (False, True, True, False, False),     # 조합 예시
}


# # ── Sweep 설정 ────────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "match_rate/peekajudge", "goal": "maximize"},
    "parameters": {
        "persona_name":    {"values": ["A_최재원"]},
        "collection_name": {"values": ["books_intro_48k", "books_pub_review_46k"]},
        "query_transform": {"values": ["none", "step_back", "rewrite", "decompose", "rewrite_decompose", "hyde"]},
        "judge_model":     {"values": ["claude-haiku-4-5-20251001", "gpt-4o-mini"]},
        "use_genre_filter": {"values": [True, False]},
        "run_index":       {"values": list(range(1, 2))},
    },
}


# # ── Sweep 설정 ────────────────────────────────────────────────────────────────
# SWEEP_CONFIG = {
#     "method": "grid",
#     "metric": {"name": "match_rate/peekajudge", "goal": "maximize"},
#     "parameters": {
#         "persona_name":    {"values": ["A_최재원", "B_한미영", "C_오민아", "D_이수빈", "E_정미희"]},
#         "collection_name": {"values": ["books_intro_48k"]},
#         "query_transform": {"values": ["none_decompose_hyde", "step_back_rewrite"]},
#         "judge_model":     {"values": ["claude-haiku-4-5-20251001"]},
#         "use_genre_filter": {"values": [True]},
#         "run_index":       {"values": list(range(1, 2))},
#     },
# }


# ── rag 모듈 설정 ─────────────────────────────────────────────────────────────
def _set_query_transform(query_transform: str) -> None:
    original, step_back, rewrite, decompose, hyde = QUERY_TRANSFORM_CONFIGS[query_transform]
    qt_v5.USE_ORIGINAL  = original
    qt_v5.USE_STEP_BACK = step_back
    qt_v5.USE_REWRITE   = rewrite
    qt_v5.USE_DECOMPOSE = decompose
    qt_v5.USE_HYDE      = hyde


# ── 단일 페르소나 실행 ────────────────────────────────────────────────────────
def run_for_persona(persona_id:      str,
                    persona_bank:    dict,
                    run_id:          str,
                    query_transform: str,
                    use_genre_filter: bool,
                    args:            argparse.Namespace,
                    judge_model:     str = JUDGE_MODEL) -> dict:
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
            "judge_model":       judge_model,
            "max_turns":         MAX_TURNS,
            "collection_name":   os.getenv("QDRANT_COLLECTION_NAME", ""),
            "chroma_db_path":    chroma_db_path,
            "run_id":            run_id,
        },
        tags=WANDB_TAGS,
        reinit=True,
    )

    qt_v5.QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "")
    _set_query_transform(query_transform)

    try:
        with get_openai_callback() as cb:
            result = run_multi_session(
                persona_id=persona_id,
                full_persona=full_persona,
                run_id=run_id,
                create_app_fn=lambda chroma_db_path, _gf=use_genre_filter: create_app(
                    chroma_db_path=chroma_db_path,
                    use_genre_filter=_gf,
                ),
                initial_state=initial_state,
                chroma_base_dir=SIMULATION_CHROMA_BASE,
                judge_model=judge_model,
                n_sessions=args.n_sessions,
                max_turns=MAX_TURNS,
                verbose=args.verbose,
            )

        sessions = result.get("sessions", [])
        usersim_in, usersim_out = _estimate_usersim_tokens(sessions)
        judge_in,   judge_out   = _estimate_judge_tokens(sessions)
        wandb.log({
            "token/crs_input":      cb.prompt_tokens,
            "token/crs_output":     cb.completion_tokens,
            "token/usersim_input":  usersim_in,
            "token/usersim_output": usersim_out,
            "token/judge_input":    judge_in,
            "token/judge_output":   judge_out,
            "token/total_input":    cb.prompt_tokens + usersim_in + judge_in,
            "token/total_output":   cb.completion_tokens + usersim_out + judge_out,
        })
        return result
    finally:
        wandb.finish()
        gc.collect()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    wandb.init(tags=WANDB_TAGS)
    cfg = wandb.config

    bank         = load_persona_bank(os.path.join(REPO_ROOT, "backend/data/personas"))
    persona_id   = cfg.persona_name
    full_persona = copy.deepcopy(bank[persona_id])
    run_id           = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_genre_filter = getattr(cfg, "use_genre_filter", False)
    judge_model      = getattr(cfg, "judge_model", JUDGE_MODEL)
    run_index        = getattr(cfg, "run_index", 1)
    collection_name  = getattr(cfg, "collection_name", os.getenv("QDRANT_COLLECTION_NAME", ""))

    qt_v5.QDRANT_COLLECTION_NAME = collection_name
    _set_query_transform(cfg.query_transform)

    chroma_db_path = f"{SIMULATION_CHROMA_BASE}/{run_id}_{persona_id}"
    wandb.config.update({
        "n_sessions":      N_SESSIONS,
        "llm_model":       LLM_MODEL,
        "judge_model":     judge_model,
        "max_turns":       MAX_TURNS,
        "collection_name": collection_name,
        "chroma_db_path":  chroma_db_path,
        "run_id":          run_id,
        "run_index":       run_index,
    }, allow_val_change=True)

    try:
        with get_openai_callback() as cb:
            result = run_multi_session(
                persona_id=persona_id,
                full_persona=full_persona,
                run_id=run_id,
                create_app_fn=lambda chroma_db_path, _gf=use_genre_filter: create_app(
                    chroma_db_path=chroma_db_path,
                    use_genre_filter=_gf,
                ),
                initial_state=initial_state,
                chroma_base_dir=SIMULATION_CHROMA_BASE,
                judge_model=judge_model,
                n_sessions=N_SESSIONS,
                max_turns=MAX_TURNS,
                verbose=True,
            )

        sessions = result.get("sessions", [])
        usersim_in, usersim_out = _estimate_usersim_tokens(sessions)
        judge_in,   judge_out   = _estimate_judge_tokens(sessions)
        wandb.log({
            "token/crs_input":      cb.prompt_tokens,
            "token/crs_output":     cb.completion_tokens,
            "token/usersim_input":  usersim_in,
            "token/usersim_output": usersim_out,
            "token/judge_input":    judge_in,
            "token/judge_output":   judge_out,
            "token/total_input":    cb.prompt_tokens + usersim_in + judge_in,
            "token/total_output":   cb.completion_tokens + usersim_out + judge_out,
        })
    finally:
        wandb.finish()
        gc.collect()


# ── 단일 실행 (main) ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PeekaReader 멀티세션 시뮬레이션 (jjc)")
    parser.add_argument("--persona",      type=str, help="페르소나 ID (예: A_최재원)")
    parser.add_argument("--all",          action="store_true", help="전체 페르소나 순회")
    parser.add_argument("--persona-dir",  type=str,
                        default=os.path.join(REPO_ROOT, "backend/data/personas"))
    parser.add_argument("--query-transform", type=str, default="none",
                        choices=list(QUERY_TRANSFORM_CONFIGS.keys()))
    parser.add_argument("--use-genre-filter", action="store_true", default=False)
    parser.add_argument("--judge-model",  type=str, default=JUDGE_MODEL,
                        help="Judge LLM 모델 (예: claude-haiku-4-5-20251001, gpt-4o-mini, gemini-1.5-flash, HCX-003)")
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
                judge_model=args.judge_model,
            )
        except Exception as e:
            import traceback
            print(f"\n[페르소나 {pid} 실패] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
