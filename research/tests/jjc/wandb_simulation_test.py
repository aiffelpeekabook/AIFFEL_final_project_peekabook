"""
wandb + LLM judge 테스트.

단일 실행:
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/wandb_simulation_test.py

Sweep 실행 (2 collection × 3 persona × 3 반복 = 18 runs):
    cd /home/jjeong3150/work/peekabook/backend
    python ../research/tests/jjc/wandb_simulation_test.py --sweep
"""
import asyncio
import copy
import json
import os
import queue
import re
import sys
import threading
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import wandb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

from app.pipeline.graph import app, initial_state
from app.simulation.user_sim import PERSONA_TEMPLATES, UserSimAgent


# ── Sweep 설정 ───────────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "judge_score", "goal": "maximize"},
    "parameters": {
        "collection_name": {"values": ["books_intro_48k", "books_merged_48k"]},
        "persona_name":    {"values": ["중년_역사_비문학", "직장인_SF팬", "대학생_문학팬"]},
        "run_index":       {"values": [1, 2, 3]},
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


async def run_crs(thread_id: str, user_to_crs: queue.Queue, crs_to_user: queue.Queue):
    session_config = {"configurable": {"thread_id": thread_id}}
    state = copy.deepcopy(initial_state)
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
                "recommendations": crs_result.get("recommendations", []),
                "summary":         crs_result.get("summary", ""),
            })
            break
        user_to_crs.put(agent.answer(message))


async def run_session(persona: dict, results: list, thread_id: str):
    u2c: queue.Queue = queue.Queue()
    c2u: queue.Queue = queue.Queue()
    t = threading.Thread(target=run_user_sim, args=(persona, results, u2c, c2u))
    t.start()
    await run_crs(thread_id, u2c, c2u)
    t.join()


# ── LLM Judge ────────────────────────────────────────────────────────────────
def llm_judge(persona: dict, recommendations: list, summary: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if isinstance(recommendations, list) and recommendations:
        rec_text = "\n".join([
            f"{i+1}. {r.get('title', '')} / {r.get('author', '')} — {r.get('reason', '')}"
            for i, r in enumerate(recommendations)
        ])
    elif isinstance(recommendations, str) and recommendations:
        rec_text = recommendations  # raw string fallback
    else:
        rec_text = "(추천 없음)"

    persona_text = "\n".join(f"- {k}: {v}" for k, v in persona.items())

    prompt = f"""당신은 도서 추천 시스템의 평가자입니다.
아래 페르소나와 사용자 프로파일 요약을 보고, 추천된 도서 3권이 얼마나 적합한지 평가하세요.

[페르소나]
{persona_text}

[사용자 프로파일 요약]
{summary}

[추천 도서]
{rec_text}

평가 기준:
1. 페르소나의 선호 장르·분위기와 일치하는가
2. 이미 읽은 책과 중복되지 않는가
3. 추천 이유가 프로파일과 실제로 연결되는가

아래 JSON 형식으로만 답하세요:
{{
    "score": 0~10 사이 숫자,
    "reason": "평가 근거 2~3문장"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match:
        result = json.loads(match.group())
        return {"score": float(result["score"]), "reason": result["reason"]}
    return {"score": 0.0, "reason": "파싱 실패"}


# ── 단일 실행 ────────────────────────────────────────────────────────────────
async def main():
    persona_name = "중년_역사_비문학"
    persona = PERSONA_TEMPLATES[persona_name]

    wandb.init(
        project="peekabook-crs",
        name=f"sim_{persona_name}",
        config={
            "persona":         persona_name,
            "llm_model":       os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME", ""),
        },
    )

    results = []
    await run_session(persona, results, thread_id="single_run_001")

    if results:
        r = results[0]
        judge = llm_judge(persona, r["recommendations"], r["summary"])
        print(f"\n[LLM Judge] score: {judge['score']} / 10")
        print(f"            reason: {judge['reason']}")
        wandb.log({"judge_score": judge["score"], "judge_reason": judge["reason"]})

    wandb.finish()


# ── Sweep 단위 실행 ───────────────────────────────────────────────────────────
def run():
    """sweep agent가 반복 호출하는 단위 실행 함수."""
    import app.rag.query_transform as qt

    wandb.init()
    cfg = wandb.config

    # collection 교체 (monkey-patch)
    qt.QDRANT_COLLECTION_NAME = cfg.collection_name

    persona = PERSONA_TEMPLATES[cfg.persona_name]
    thread_id = f"sweep_{cfg.persona_name}_{cfg.collection_name}_{cfg.run_index}_{uuid.uuid4().hex[:6]}"

    results = []
    asyncio.run(run_session(persona, results, thread_id))

    if results:
        r = results[0]
        judge = llm_judge(persona, r["recommendations"], r["summary"])
        print(f"\n[Judge] {cfg.persona_name} | {cfg.collection_name} | run {cfg.run_index} → {judge['score']}")
        wandb.log({"judge_score": judge["score"], "judge_reason": judge["reason"]})

    wandb.finish()


# ── 진입점 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="peekabook-crs")
        wandb.agent(sweep_id, function=run)
    else:
        asyncio.run(main())
