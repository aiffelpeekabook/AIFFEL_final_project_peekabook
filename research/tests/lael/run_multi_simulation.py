"""
멀티세션 시뮬레이션 진입점 (Lael).

페르소나 1명을 끊어 실행하거나 (--persona A_최재원),
전체 페르소나를 순회 실행 (--all)할 수 있음.

W&B는 placeholder project name 사용 — 본인 project 이름이 정해지면
--wandb-project / --wandb-entity 인자로 덮어쓰면 됨.

Colab에서 호출 예:
    !cd /content/peekabook/research/tests/lael && \\
        python run_multi_simulation.py --persona A_최재원 \\
            --wandb-project peekareader-multi-session-lael

전체 페르소나 (Pro 또는 GPU 환경):
    python run_multi_simulation.py --all

로컬/Codespaces:
    cd research/tests/lael && python run_multi_simulation.py --persona A_최재원

더미 페르소나로 빠른 검증:
    python run_multi_simulation.py --persona-dir backend/data/personas/dummy --persona a
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# repo root의 backend/를 import path에 추가
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
from app.simulation.multi_session_simulator_v2 import run_multi_session
from app.simulation.persona_loader          import load_persona_bank
from app.simulation.multi_sim_logger        import (
    print_multi_session_summary,
    save_simul_result,
)
from app.simulation.profile_visualizer      import (
    log_clusters_to_wandb,
    visualize_profile_clusters,
)


# ──────────────────────────────────────────────
# 그래프 모듈 import (factory + initial_state)
# ──────────────────────────────────────────────
# 현재는 팀원 graph_test3이 factory를 export함.
# 그래프가 graph_v4 등으로 promote되면 여기 한 줄만 변경.
from app.pipeline.graph_test3 import create_app, initial_state
import app.rag.query_transform_v5 as qt_v5


# ──────────────────────────────────────────────
# Placeholders — 본인 wandb 정보로 바꾸면 됨
# ──────────────────────────────────────────────
DEFAULT_WANDB_PROJECT = "peekareader-simul-test1"
DEFAULT_WANDB_ENTITY  = "jjeong3150-aiffel"   # None이면 wandb 기본 entity 사용


def run_for_persona(persona_id:   str,
                    persona_bank: dict,
                    run_id:       str,
                    args:         argparse.Namespace) -> dict:
    """페르소나 1명 시뮬레이션 + W&B run 생성 + 결과 저장."""
    full_persona = persona_bank[persona_id]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{persona_id}_{run_id}",
        config={
            "persona_id":  persona_id,
            "n_sessions":  args.n_sessions or len(full_persona["sessions"]),
            "llm_model":   LLM_MODEL,
            "judge_model": args.judge_model,
            "max_turns":   MAX_TURNS,
            "run_id":      run_id,
        },
        reinit=True,   # 같은 프로세스에서 여러 페르소나 → 여러 run
    )

    try:
        result = run_multi_session(
            persona_id=persona_id,
            full_persona=full_persona,
            run_id=run_id,
            create_app_fn=lambda chroma_db_path: create_app(
                chroma_db_path=chroma_db_path,
                use_genre_filter=False,
                rag_module=qt_v5,
            ),
            initial_state=initial_state,
            chroma_base_dir=SIMULATION_CHROMA_BASE,
            judge_model=args.judge_model,
            n_sessions=args.n_sessions,
            max_turns=MAX_TURNS,
            verbose=args.verbose,
        )

        # 결과 JSON 저장
        save_simul_result(result, save_dir=SIMULATION_RESULTS_DIR)

        # 클러스터 시각화 (optional, 세션 누적 후)
        chroma_db_path = f"{SIMULATION_CHROMA_BASE}/{run_id}_{persona_id}"
        if args.viz:
            try:
                png = visualize_profile_clusters(
                    chroma_path=chroma_db_path,
                    persona_id=persona_id,
                )
                if png:
                    log_clusters_to_wandb(png, key=f"clusters/{persona_id}")
            except Exception as e:
                print(f"[viz 실패] {e}")

        return result
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="PeekaReader 멀티세션 시뮬레이션 진입점"
    )
    parser.add_argument(
        "--persona", type=str,
        help="실행할 페르소나 ID (예: A_최재원). --all과 둘 중 하나는 필수",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="모든 페르소나를 순회 실행",
    )
    parser.add_argument(
        "--persona-dir", type=str, default=PERSONA_DIR,
        help=f"페르소나 JSON 디렉토리 (default: {PERSONA_DIR})",
    )
    parser.add_argument(
        "--n-sessions", type=int, default=None,
        help="페르소나당 세션 수. 미지정 시 페르소나의 모든 세션",
    )
    parser.add_argument(
        "--judge-model", type=str, default=JUDGE_MODEL,
        help=f"PeekaJudge 모델 (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--wandb-project", type=str,
        default=os.getenv("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
    )
    parser.add_argument(
        "--wandb-entity", type=str,
        default=os.getenv("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
    )
    parser.add_argument(
        "--viz", action="store_true",
        help="페르소나 종료 후 클러스터 시각화 PNG 생성 + W&B에 push",
    )
    parser.add_argument(
        "--quiet", dest="verbose", action="store_false",
        help="자세한 콘솔 출력 끄기",
    )
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    if not args.persona and not args.all:
        parser.error("--persona 또는 --all 중 하나는 필수")

    # 페르소나 풀 로드
    bank = load_persona_bank(args.persona_dir)
    print(f"\n[페르소나 풀] {len(bank)}개 로드: {list(bank.keys())}")

    # 실행할 페르소나 ID 결정
    if args.all:
        persona_ids = list(bank.keys())
    else:
        if args.persona not in bank:
            print(f"[오류] '{args.persona}' 페르소나 없음. 사용 가능: {list(bank.keys())}")
            sys.exit(1)
        persona_ids = [args.persona]

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for pid in persona_ids:
        print(f"\n{'#'*60}")
        print(f"# 페르소나: {pid}")
        print(f"{'#'*60}")
        try:
            r = run_for_persona(pid, bank, run_id, args)
            results.append(r)
        except Exception as e:
            print(f"\n[페르소나 {pid} 실패] {e}")
            import traceback
            traceback.print_exc()

    # 전체 요약
    if len(results) > 1:
        print_multi_session_summary(results)


if __name__ == "__main__":
    main()
