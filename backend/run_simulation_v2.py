"""
CRS × UserSim 오케스트레이션 스크립트 (v2).

cover_url 기반 이미지 표시 + search_library/check_book_availability 전용 api_tools_v2 사용.

실행:
    cd backend
    python run_simulation_v2.py
"""
import asyncio
import copy
import json
import queue
import threading
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.pipeline.graph_v2 import app, initial_state, config
from app.simulation.user_sim import PERSONA_TEMPLATES, UserSimAgent


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


async def run_crs(thread_id: str):
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
                "persona": persona,
                "user_profile": crs_result.get("user_profile", {}),
                "summary": crs_result.get("summary", ""),
                "reflection": crs_result.get("reflection", ""),
                "recommendations": crs_result.get("recommendations", []),
                "final_message": (
                    crs_result["messages"][-1].content
                    if crs_result.get("messages") else ""
                ),
                "conversation": agent.get_history(),
            })
            break
        response = agent.answer(message)
        user_to_crs.put(response)


async def run_session(persona: dict, results: list, thread_id: str):
    t_user = threading.Thread(target=run_user_sim, args=(persona, results))
    t_user.start()
    await run_crs(thread_id)
    t_user.join()


async def main():
    persona = PERSONA_TEMPLATES["중년_역사_비문학"]
    thread_id = "eval_session_v2_001"

    await run_session(persona, eval_results, thread_id)

    print("\n" + "="*50)
    print("세션 종료")
    print("="*50)

    if eval_results:
        r = eval_results[-1]
        print("\n[페르소나]")
        print(json.dumps(r["persona"], ensure_ascii=False, indent=2))
        print("\n[추출된 프로필]")
        print(r["user_profile"])
        print("\n[요약]")
        print("summary :", r["summary"])
        print("reflection :", r["reflection"])
        print("\n[추천 결과]")
        print(r["final_message"])


if __name__ == "__main__":
    asyncio.run(main())
