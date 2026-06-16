"""
CRS × UserSim 오케스트레이션 스크립트 (v4).

v3 대비 변경:
- user_sim_v2 사용: DNA 페르소나(PERSONA_BANK) + speaking_style 반영 UserSimAgent
- graph_test2.create_app()으로 실행마다 독립된 ChromaDB 경로 사용
- 세션 종료 후 PeekaJudge 평가 자동 실행

실행:
    cd /home/jjeong3150/work/peekabook/research/tests/jjc
    python run_simulation_v4.py
"""
import asyncio
import copy
import json
import os
import queue
import sys
import threading
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

from langchain_core.messages import AIMessage, HumanMessage

from app.pipeline.graph_test2 import create_app, initial_state
from app.simulation.user_sim_v2 import PERSONA_BANK, UserSimAgent, judge_session


user_to_crs: queue.Queue = queue.Queue()
crs_to_user: queue.Queue = queue.Queue()
eval_results: list = []


def _extract_ai_responses(state: dict[str, Any]) -> list[str]:
    messages = state.get("messages", [])
    responses = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            break
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            responses.append(msg.content)
    responses.reverse()
    return responses


async def run_crs(app, thread_id: str):
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


def run_user_sim(persona: dict, result_collector: list):
    agent = UserSimAgent(persona=persona, verbose=True)
    while True:
        message = crs_to_user.get()
        if isinstance(message, dict) and message.get("__done__"):
            crs_result = message["result"]
            result_collector.append({
                "persona":         persona,
                "user_profile":    crs_result.get("user_profile", {}),
                "summary":         crs_result.get("summary", ""),
                "reflection":      crs_result.get("reflection", ""),
                "recommendations": crs_result.get("recommendations", []),
                "final_message":   (
                    crs_result["messages"][-1].content
                    if crs_result.get("messages") else ""
                ),
                "conversation":    agent.get_history(),
            })
            break
        response = agent.answer(message)
        user_to_crs.put(response)


async def run_session(app, persona: dict, results: list, thread_id: str):
    t_user = threading.Thread(target=run_user_sim, args=(persona, results), daemon=True)
    t_user.start()
    await run_crs(app, thread_id)
    t_user.join()


async def main():
    persona_id = "A_최재원"
    persona    = PERSONA_BANK[persona_id]

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_key    = persona_id.split("_")[0]   # "A_최재원" → "A"
    chroma_db_path = os.path.join(
        os.path.dirname(__file__), "../../../backend/chroma_db_runs",
        f"{timestamp}_{persona_key}"
    )
    thread_id = f"eval_session_v4_{timestamp}_{persona_key}"

    print(f"[ChromaDB 경로] {chroma_db_path}")

    app = create_app(chroma_db_path=chroma_db_path)
    await run_session(app, persona, eval_results, thread_id)

    print("\n" + "="*50)
    print("세션 종료")
    print("="*50)

    if not eval_results:
        return

    r = eval_results[-1]
    print("\n[페르소나]")
    print(json.dumps(r["persona"], ensure_ascii=False, indent=2))
    print("\n[추출된 프로필]")
    print(r["user_profile"])
    print("\n[요약]")
    print("summary    :", r["summary"])
    print("reflection :", r["reflection"])
    print("\n[추천 결과]")
    print(r["final_message"])

    # Judge 평가
    print("\n" + "="*50)
    print("PeekaJudge 평가")
    print("="*50)
    judge_result = judge_session(session_result=r, persona=persona, verbose=True)

    if judge_result:
        print(f"\n최종 match_rate: {judge_result.get('book_match_rate', 0):.0%}")

    print(f"\n[ChromaDB 경로] {chroma_db_path}")


if __name__ == "__main__":
    asyncio.run(main())
