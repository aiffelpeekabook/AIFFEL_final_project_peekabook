from __future__ import annotations

from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages

# ── 공통 타입은 state.py에서 그대로 가져온다 (Pydantic 클래스 동일성 유지) ──
from app.state.state import (
    SlotStatus, Phase, SLOT_NAMES, SLOT_DESCRIPTIONS,
    ProfileSlot, UserProfile, BookExperience, SessionMemory, MemoryLink,
)

__all__ = [
    "SlotStatus", "Phase", "SLOT_NAMES", "SLOT_DESCRIPTIONS",
    "ProfileSlot", "UserProfile", "BookExperience", "SessionMemory", "MemoryLink",
    "GraphState",
]


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    phase: Phase
    turn_count: int
    user_profile: UserProfile
    current_slot: Optional[str]
    similar_profiles: list[dict[str, Any]]
    matched_profile_id: Optional[str]
    book_experiences: list[BookExperience]
    asked_book_experience: bool
    summary: str
    reflection: str
    links: list[MemoryLink]
    ai_response: str
    retrieved_books: list
    recommendations: list
    genre_filter: list
    genre_level: str
    availability_results: Optional[str]
    hypothetical_doc: str
    query_transforms: dict  # {original, step_back, rewritten, sub_queries, all}
