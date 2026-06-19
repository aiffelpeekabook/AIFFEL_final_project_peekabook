"""
멀티세션 시뮬레이션 진입점 (Lael)

(jjc) 의 run_multi_session_simulator.py 베이스에 다음 확장:
- Resume 가능한 체크포인트 (페르소나 단위 저장)
- 복수 페르소나 동시 실행 (multiprocessing)
- 페르소나 선택 옵션 다양화 (--personas, --persona-slice)
- 결과 저장 경로 명시적 지정 (--output-dir)

실행 예:
    # 단일
    python run_multi_simulation.py --persona A_최재원

    # 복수 (콤마 구분)
    python run_multi_simulation.py --personas A_최재원,B_한미영

    # 슬라이싱 (페르소나 인덱스 기준)
    python run_multi_simulation.py --persona-slice 0:3
    python run_multi_simulation.py --persona-slice 3:

    # 전체 + 동시 실행 (Colab T4 권장: 2)
    python run_multi_simulation.py --all --concurrent 2

    # Resume (이전 run 끊겼을 때)
    python run_multi_simulation.py --all --run-id 20260619_103000
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
# research/tests/lael/run_multi_simulation.py → ../../.. → repo root
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
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


# ── 토큰 추정 ────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _estimate_usersim_tokens(sessions: list) -> tuple[int, int]:
    """대화 기록에서 UserSim 토큰 추정"""
    system_tokens = 350
    total_in  = system_tokens * len(sessions)
    total_out = 0
    for s in sessions:
        for turn in s.get("conversation", []):
            total_in  += _count_tokens(turn.get("crs", ""))
            total_out += _count_tokens(turn.get("user", ""))
    return total_in, total_out


def _estimate_judge_tokens(sessions: list) -> tuple[int, int]:
    """retrieved_books에서 Judge 토큰 추정"""
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


# ── 설정 ──────────────────────────────────────────────────────
WANDB_PROJECT = "peekabook-crs-multisession"
WANDB_ENTITY  = "jjeong3150-aiffel"
WANDB_TAGS    = ["multi_session", "colab"]

DEFAULT_N_SESSIONS = 30


# ── Query Transformation 조합 정의 ───────────────────────────
# 튜플: (use_original, use_step_back, use_rewrite, use_decompose, use_hyde)
QUERY_TRANSFORM_CONFIGS = {
    "none":      (True,  False, False, False, False),
    "step_back": (False, True,  False, False, False),
    "rewrite":   (False, False, True,  False, False),
    "decompose": (False, False, False, True,  False),
    "hyde":      (False, False, False, False, True),
}


def _set_query_transform(query_transform: str) -> None:
    original, step_back, rewrite, decompose, hyde = QUERY_TRANSFORM_CONFIGS[query_transform]
    qt_v5.USE_ORIGINAL  = original
    qt_v5.USE_STEP_BACK = step_back
    qt_v5.USE_REWRITE   = rewrite
    qt_v5.USE_DECOMPOSE = decompose
    qt_v5.USE_HYDE      = hyde


# ── 페르소나 선택 ─────────────────────────────────────────────
def _select_personas(bank: dict, args: argparse.Namespace) -> list[str]:
    """CLI 인자 기반 페르소나 ID 리스트 선택.
    우선순위: --all > --personas > --persona-slice > --persona
    """
    all_ids = list(bank.keys())

    if args.all:
        return all_ids

    if args.personas:
        ids = [p.strip() for p in args.personas.split(",") if p.strip()]
        missing = [p for p in ids if p not in bank]
        if missing:
            raise ValueError(f"페르소나 없음: {missing}. 가능: {all_ids}")
        return ids

    if args.persona_slice:
        # "0:3", "3:", ":2" 형태 파싱
        try:
            parts = args.persona_slice.split(":")
            slc = slice(*[int(p) if p else None for p in parts])
            selected = all_ids[slc]
            if not selected:
                raise ValueError(f"빈 슬라이스 결과: {args.persona_slice}")
            return selected
        except (ValueError, TypeError) as e:
            raise ValueError(f"잘못된 slice 형식: {args.persona_slice}. 예: 0:3, 3:, :2 ({e})")

    if args.persona:
        if args.persona not in bank:
            raise ValueError(f"페르소나 없음: {args.persona}. 가능: {all_ids}")
        return [args.persona]

    raise ValueError("--persona, --personas, --persona-slice, --all 중 하나는 필수")


# ── 체크포인트 (페르소나 단위 저장) ────────────────────────────
def _persona_output_path(output_dir: str, run_id: str, persona_id: str) -> Path:
    return Path(output_dir) / run_id / f"{persona_id}.json"


def _meta_path(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / run_id / "_meta.json"


def _is_persona_done(output_dir: str, run_id: str, persona_id: str) -> bool:
    return _persona_output_path(output_dir, run_id, persona_id).exists()


def _save_persona_result(output_dir: str, run_id: str,
                          persona_id: str, result: dict) -> Path:
    path = _persona_output_path(output_dir, run_id, persona_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def _save_meta(output_dir: str, run_id: str, meta: dict) -> Path:
    path = _meta_path(output_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


# ── 단일 페르소나 실행 ────────────────────────────────────────
def run_for_persona(persona_id:       str,
                    persona_bank:     dict,
                    run_id:           str,
                    query_transform:  str,
                    use_genre_filter: bool,
                    n_sessions:       int,
                    max_turns:        int,
                    judge_model:      str,
                    wandb_project:    str,
                    wandb_entity:     str,
                    output_dir:       str,
                    verbose:          bool = True) -> dict:
    """페르소나 1개 처리 후 JSON 저장"""
    full_persona   = copy.deepcopy(persona_bank[persona_id])  # LTM 격리
    chroma_db_path = f"{SIMULATION_CHROMA_BASE}/{run_id}_{persona_id}"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"{persona_id}_{query_transform}_{run_id}",
        config={
            "persona_name":     persona_id,
            "query_transform":  query_transform,
            "use_genre_filter": use_genre_filter,
            "n_sessions":       n_sessions,
            "llm_model":        LLM_MODEL,
            "judge_model":      judge_model,
            "max_turns":        max_turns,
            "collection_name":  os.getenv("QDRANT_COLLECTION_NAME", ""),
            "chroma_db_path":   chroma_db_path,
            "run_id":           run_id,
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
                n_sessions=n_sessions,
                max_turns=max_turns,
                verbose=verbose,
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

        # 체크포인트 저장 (페르소나 완료 시 즉시)
        save_path = _save_persona_result(output_dir, run_id, persona_id, result)
        if verbose:
            print(f"  [체크포인트] 저장됨: {save_path}")

        return result
    finally:
        wandb.finish()
        gc.collect()


# ── multiprocessing worker ───────────────────────────────────
def _worker_run_persona(kwargs: dict) -> tuple[str, str]:
    """multiprocessing pool에서 호출되는 wrapper.
    bank 전체를 보낼 필요 없이 해당 페르소나 dict만 패킹해서 전송함.
    """
    persona_id   = kwargs["persona_id"]
    persona_data = kwargs.pop("persona_data")
    kwargs["persona_bank"] = {persona_id: persona_data}
    try:
        run_for_persona(**kwargs)
        return persona_id, "success"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return persona_id, f"error: {e}"


# ── 동시 실행 dispatcher ──────────────────────────────────────
def run_personas_concurrent(persona_ids:   list[str],
                             persona_bank: dict,
                             concurrent:   int,
                             common_kwargs: dict) -> list[tuple[str, str]]:
    """페르소나 리스트를 concurrent 개수만큼 동시 실행 (spawn context)"""
    tasks = []
    for pid in persona_ids:
        kwargs = {
            "persona_id":   pid,
            "persona_data": persona_bank[pid],
            **common_kwargs,
        }
        tasks.append(kwargs)

    # CUDA 자식 프로세스 호환을 위해 spawn 사용 (fork 안 됨)
    ctx = get_context("spawn")
    with ctx.Pool(processes=concurrent) as pool:
        results = pool.map(_worker_run_persona, tasks)
    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PeekaReader 멀티세션 시뮬레이션")

    # 페르소나 선택
    parser.add_argument("--persona",       type=str,
                        help="단일 페르소나 ID (예: A_최재원)")
    parser.add_argument("--personas",      type=str,
                        help="복수 페르소나 ID (콤마 구분, 예: A_최재원,B_한미영)")
    parser.add_argument("--persona-slice", type=str,
                        help="페르소나 슬라이싱 (예: 0:3, 3:, :2)")
    parser.add_argument("--all",           action="store_true",
                        help="전체 페르소나 순회")
    parser.add_argument("--persona-dir",   type=str,
                        default=os.path.join(REPO_ROOT, "backend/data/personas"))

    # 시뮬레이션 설정
    parser.add_argument("--query-transform",  type=str, default="none",
                        choices=list(QUERY_TRANSFORM_CONFIGS.keys()))
    parser.add_argument("--use-genre-filter", action="store_true", default=False)
    parser.add_argument("--judge-model",      type=str, default=JUDGE_MODEL)
    parser.add_argument("--n-sessions",       type=int, default=DEFAULT_N_SESSIONS)
    parser.add_argument("--max-turns",        type=int, default=MAX_TURNS)

    # 동시 실행
    parser.add_argument("--concurrent",       type=int, default=1,
                        help="동시 실행 페르소나 수 (Colab T4 권장: 2)")

    # Resume / 출력
    parser.add_argument("--run-id",    type=str, default=None,
                        help="Resume할 run ID (생략 시 새로 생성)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "research/data/simulation_results"))
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="이미 완료된 페르소나도 무시하고 처음부터")

    # W&B
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity",  type=str, default=WANDB_ENTITY)

    # 기타
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    # ── 페르소나 선택
    bank     = load_persona_bank(args.persona_dir)
    selected = _select_personas(bank, args)

    # ── run_id 결정
    run_id    = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    is_resume = args.run_id is not None

    # ── 이미 완료된 페르소나 skip (resume)
    if is_resume and not args.no_resume:
        skipped, remaining = [], []
        for pid in selected:
            if _is_persona_done(args.output_dir, run_id, pid):
                skipped.append(pid)
            else:
                remaining.append(pid)
        if skipped:
            print(f"\n[Resume] {len(skipped)}개 페르소나 이미 완료 (skip): {skipped}")
        if not remaining:
            print(f"[Resume] 모든 페르소나가 이미 완료되었습니다. 종료.")
            return
        selected = remaining

    # ── 메타 저장 (started_at)
    meta = {
        "run_id":           run_id,
        "started_at":       datetime.now().isoformat(),
        "personas":         selected,
        "n_sessions":       args.n_sessions,
        "query_transform":  args.query_transform,
        "use_genre_filter": args.use_genre_filter,
        "judge_model":      args.judge_model,
        "concurrent":       args.concurrent,
        "output_dir":       args.output_dir,
        "is_resume":        is_resume,
    }
    _save_meta(args.output_dir, run_id, meta)

    # ── 실행 시작 배너
    print(f"\n{'='*60}")
    print(f"멀티세션 시뮬레이션 시작")
    print(f"  run_id:     {run_id}")
    print(f"  personas:   {selected}")
    print(f"  concurrent: {args.concurrent}")
    print(f"  output:     {args.output_dir}/{run_id}/")
    print(f"{'='*60}")

    # ── 공통 kwargs (페르소나마다 동일)
    common_kwargs = {
        "run_id":           run_id,
        "query_transform":  args.query_transform,
        "use_genre_filter": args.use_genre_filter,
        "n_sessions":       args.n_sessions,
        "max_turns":        args.max_turns,
        "judge_model":      args.judge_model,
        "wandb_project":    args.wandb_project,
        "wandb_entity":     args.wandb_entity,
        "output_dir":       args.output_dir,
        "verbose":          args.verbose,
    }

    # ── 실행
    if args.concurrent > 1 and len(selected) > 1:
        results = run_personas_concurrent(
            persona_ids=selected,
            persona_bank=bank,
            concurrent=args.concurrent,
            common_kwargs=common_kwargs,
        )
        print(f"\n{'='*60}")
        print("실행 결과 요약")
        print(f"{'='*60}")
        for pid, status in results:
            mark = "✓" if status == "success" else "✗"
            print(f"  {mark} {pid}: {status}")
    else:
        # 시퀀셜 (concurrent=1 또는 페르소나 1개)
        for pid in selected:
            print(f"\n{'#'*60}")
            print(f"# 페르소나: {pid}")
            print(f"{'#'*60}")
            try:
                run_for_persona(
                    persona_id=pid,
                    persona_bank=bank,
                    **common_kwargs,
                )
            except Exception as e:
                import traceback
                print(f"\n[페르소나 {pid} 실패] {e}")
                traceback.print_exc()

    # ── 메타 업데이트 (completed_at)
    meta["completed_at"] = datetime.now().isoformat()
    _save_meta(args.output_dir, run_id, meta)

    print(f"\n[완료] {args.output_dir}/{run_id}/")


if __name__ == "__main__":
    main()
