"""
graph_demo.py — 웹 데모용 그래프 어댑터.

기존 backend/app/pipeline/graph_test3.py를 그대로 import해서 사용하되,
웹 챗봇 환경에 맞게 다음 두 가지를 처리한다:

1. interrupt_before 우회
   - 원본은 process_slot_answer 등 3개 노드 직전에 멈춰서 인간 입력을 기다리는 구조.
   - 웹에서는 매 요청이 (유저 메시지) → (AI 응답) 한 쌍으로 완결돼야 함.
   - 따라서 stream API + Command(resume) 패턴 대신, 명시적인 상태 inject로 처리.

2. 세션 누적 (localStorage 기반)
   - 백엔드는 stateless하게 동작.
   - 클라이언트가 이전 세션의 UserProfile + book_experiences + summary를 보내주면
     그것을 initial_state에 inject한 뒤 그래프 실행.
   - 그래프 종료 후 업데이트된 프로파일을 다시 클라이언트에 반환.

이렇게 하면 ChromaDB 영속성 없이도 "세션을 넘나드는 프로파일 누적"의
시각적 효과를 정확히 재현할 수 있음.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

# 원본 그래프 모듈은 절대 건드리지 않고 import만.
from app.pipeline import graph_test3
from app.state.state import (
    UserProfile,
    BookExperience,
    Phase,
    SLOT_NAMES,
)


# ────────────────────────────────────────────────────────────────────────────
# 그래프 인스턴스 (모듈 로드 시 1회만 생성)
# ────────────────────────────────────────────────────────────────────────────
# 데모는 ChromaDB 영속성을 쓰지 않으므로 임시 디렉토리에 만들고 매 호출마다 새로 시작.
# 진짜 영속성은 클라이언트 localStorage가 담당.
_DEMO_CHROMA_PATH = os.environ.get("DEMO_CHROMA_PATH", "/tmp/peekabook_demo_chroma")

_compiled_graph = graph_test3.create_app(
    chroma_db_path=_DEMO_CHROMA_PATH,
    use_genre_filter=True,
    rag_module=None,  # 기본값 query_transform_v4 사용
)


# ────────────────────────────────────────────────────────────────────────────
# 프로파일 직렬화 / 역직렬화 헬퍼
# ────────────────────────────────────────────────────────────────────────────
def serialize_profile_payload(state: dict) -> dict:
    """그래프 실행 후 클라이언트에 돌려줄 누적 가능한 프로파일 페이로드.

    이 dict가 그대로 localStorage에 JSON으로 저장되고, 다음 세션 시작 시
    그대로 받아서 deserialize_profile_payload로 복원된다.
    """
    profile: UserProfile = state.get("user_profile") or UserProfile()
    book_experiences: list[BookExperience] = state.get("book_experiences") or []

    return {
        "user_profile": profile.to_display_dict(),  # state.py에 이미 정의됨
        "book_experiences": [
            {
                "book_name": be.book_name,
                "impression": be.impression,
                "context": be.context,
            }
            for be in book_experiences
        ],
        "summary": state.get("summary", "") or "",
        "reflection": state.get("reflection", "") or "",
    }


def deserialize_profile_payload(payload: Optional[dict]) -> dict:
    """클라이언트가 보낸 페이로드를 그래프 initial_state 형태로 변환."""
    if not payload:
        return {
            "user_profile": UserProfile(),
            "book_experiences": [],
            "summary": "",
            "reflection": "",
        }

    profile = UserProfile.from_dict(payload.get("user_profile") or {})
    book_experiences = [
        BookExperience(
            book_name=be.get("book_name", ""),
            impression=be.get("impression", ""),
            context=be.get("context", ""),
        )
        for be in (payload.get("book_experiences") or [])
    ]

    return {
        "user_profile": profile,
        "book_experiences": book_experiences,
        "summary": payload.get("summary", "") or "",
        "reflection": payload.get("reflection", "") or "",
    }


# ────────────────────────────────────────────────────────────────────────────
# 핵심: 한 턴 실행
# ────────────────────────────────────────────────────────────────────────────
async def run_one_turn(
    user_message: str,
    session_id: str,
    prior_profile_payload: Optional[dict] = None,
    thread_id: Optional[str] = None,
) -> dict:
    """유저 메시지 한 번에 대해 그래프를 한 턴 실행하고 결과를 반환.

    Parameters
    ----------
    user_message : 유저가 방금 입력한 메시지.
    session_id   : 현재 세션 ID (클라이언트가 발급/관리).
    prior_profile_payload : 이전 세션에서 누적된 프로파일 (없으면 빈 상태로 시작).
    thread_id    : LangGraph checkpoint thread (같은 브라우저 세션 내 멀티턴용).
                   클라이언트가 첫 메시지에서 None을 보내면 새로 만들어 응답에 포함.

    Returns
    -------
    {
        "ai_response": str,                 # 사용자에게 보여줄 AI 답변
        "recommendations": list,            # 책 추천 결과 (있을 때)
        "phase": str,                       # 현재 phase (UI에서 단계 표시용)
        "profile_payload": dict,            # localStorage 저장용 누적 페이로드
        "thread_id": str,                   # 같은 세션 내 다음 호출에 그대로 전달
        "session_done": bool,               # 그래프가 END에 도달했는지
    }
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # 현재 checkpoint 상태를 확인 — 같은 thread에서 이어지는 호출인지 첫 호출인지 판단
    current_snapshot = _compiled_graph.get_state(config)
    is_first_turn_in_thread = (
        current_snapshot is None or not current_snapshot.values
    )

    if is_first_turn_in_thread:
        # 첫 턴: initial_state에 prior profile inject 후 시작
        prior = deserialize_profile_payload(prior_profile_payload)

        initial = dict(graph_test3.initial_state)  # 얕은 복사
        initial["session_id"] = session_id
        initial["messages"] = [HumanMessage(content=user_message)]
        
        # 세션 2부터는 슬롯 빈 상태로 시작
        initial["user_profile"] = UserProfile()
        initial["book_experiences"] = []

        # 이전 세션 요약만 컨텍스트로 주입
        if prior.get("summary"):
            initial["summary"] = f"[이전 세션 요약] {prior['summary']}"
        if prior.get("reflection"):
            initial["reflection"] = f"[이전 세션 인사이트] {prior['reflection']}"

        # 누적 프로파일이 있으면 슬롯필링 일부를 건너뛰고 바로 추천으로 가는 것도
        # 가능하지만, 데모에서는 "기억하고 있음을 보여주는" 효과가 더 중요하므로
        # 정상 흐름을 유지하되 generate_slot_question 노드가 이미 채워진 슬롯을
        # 인지하도록 둔다 (그래프 로직이 filled_slots()를 이미 확인함).

        await _compiled_graph.ainvoke(initial, config=config)
    else:
        # 같은 thread의 후속 턴: interrupt_before로 멈춰있는 상태에 유저 메시지만 추가
        _compiled_graph.update_state(
            config,
            {"messages": [HumanMessage(content=user_message)]},
        )
        await _compiled_graph.ainvoke(None, config=config)

    # 실행 후 상태 조회
    snapshot = _compiled_graph.get_state(config)
    state_values = snapshot.values

    # next가 비어있으면 그래프가 END에 도달한 것
    session_done = not snapshot.next

    return {
        "ai_response": state_values.get("ai_response", ""),
        "recommendations": state_values.get("recommendations", []),
        "phase": _phase_to_str(state_values.get("phase")),
        "profile_payload": serialize_profile_payload(state_values),
        "thread_id": thread_id,
        "session_done": session_done,
    }


def _phase_to_str(phase) -> str:
    """Phase enum 또는 문자열을 안전하게 문자열로 변환."""
    if phase is None:
        return ""
    if isinstance(phase, Phase):
        return phase.value
    return str(phase)


# ────────────────────────────────────────────────────────────────────────────
# 헬스체크 — HF Spaces 슬립 방지용
# ────────────────────────────────────────────────────────────────────────────
def healthcheck() -> dict:
    """그래프가 정상 컴파일됐는지만 가볍게 확인."""
    return {
        "graph_compiled": _compiled_graph is not None,
        "slot_names": SLOT_NAMES,
    }
