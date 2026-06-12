from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SlotStatus(str, Enum):
    EMPTY = "empty"
    FILLED = "filled"
    UNCLEAR = "unclear"


class Phase(str, Enum):
    SLOT_FILLING = "slot_filling"
    SIMILAR_SEARCH = "similar_search"
    MATCH_CONFIRM = "match_confirm"
    BOOK_EXPERIENCE = "book_experience"
    SUMMARY = "summary"
    REFLECTION = "reflection"
    DONE = "done"


SLOT_NAMES: list[str] = [
    "reading_goal",
    "preferred_genre",
    "reading_style",
    "difficulty_level",
    "current_context",
]

SLOT_DESCRIPTIONS: dict[str, str] = {
    "reading_goal": "사용자가 현재 왜 책을 읽고 싶어하는지 (동기/목적)",
    "preferred_genre": "사용자가 현재 관심을 가지는 장르나 분야",
    "reading_style": "사용자가 어떤 스타일의 독서를 선호하는지 (깊이, 속도, 형식 등)",
    "difficulty_level": "사용자가 선호하는 책의 난이도",
    "current_context": "사용자의 현재 상황이나 감정 상태",
}


class ProfileSlot(BaseModel):
    value: str = ""
    status: SlotStatus = SlotStatus.EMPTY
    retry_count: int = 0
    MAX_RETRIES: int = 3


class UserProfile(BaseModel):
    reading_goal: ProfileSlot = Field(default_factory=ProfileSlot)
    preferred_genre: ProfileSlot = Field(default_factory=ProfileSlot)
    reading_style: ProfileSlot = Field(default_factory=ProfileSlot)
    difficulty_level: ProfileSlot = Field(default_factory=ProfileSlot)
    current_context: ProfileSlot = Field(default_factory=ProfileSlot)

    def get_slot(self, name: str) -> ProfileSlot:
        return getattr(self, name)

    def set_slot(self, name: str, slot: ProfileSlot) -> None:
        setattr(self, name, slot)

    def filled_slots(self) -> list[str]:
        return [s for s in SLOT_NAMES if self.get_slot(s).status == SlotStatus.FILLED]

    def empty_slots(self) -> list[str]:
        return [s for s in SLOT_NAMES if self.get_slot(s).status == SlotStatus.EMPTY]

    def all_filled_or_unclear(self) -> bool:
        return all(
            self.get_slot(s).status in (SlotStatus.FILLED, SlotStatus.UNCLEAR)
            for s in SLOT_NAMES
        )

    def reading_goal_filled(self) -> bool:
        return self.reading_goal.status == SlotStatus.FILLED

    def to_embedding_text(self) -> str:
        parts = []
        for name in SLOT_NAMES:
            slot = self.get_slot(name)
            if slot.value:
                parts.append(f"{SLOT_DESCRIPTIONS[name]}: {slot.value}")
        return " | ".join(parts)

    def to_display_dict(self) -> dict:
        return {
            name: {
                "value": self.get_slot(name).value,
                "status": self.get_slot(name).status.value,
            }
            for name in SLOT_NAMES
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        profile = cls()
        for name in SLOT_NAMES:
            if name in data:
                slot_data = data[name]
                slot = ProfileSlot(
                    value=slot_data.get("value", ""),
                    status=SlotStatus(slot_data.get("status", "empty")),
                    retry_count=slot_data.get("retry_count", 0),
                )
                profile.set_slot(name, slot)
        return profile


class BookExperience(BaseModel):
    book_name: str
    impression: str
    context: str = ""


class SessionMemory(BaseModel):
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    profile: UserProfile = Field(default_factory=UserProfile)
    book_experiences: list[BookExperience] = Field(default_factory=list)
    summary: str = ""
    reflection: str = ""
    linked_session_ids: list[str] = Field(default_factory=list)


class MemoryLink(BaseModel):
    source_session_id: str
    target_session_id: str
    link_reason: str
    strength: float = 0.0


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
    availability_results: Optional[str]
