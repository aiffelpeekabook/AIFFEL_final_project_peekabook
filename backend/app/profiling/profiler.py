from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.state.state import (
    BookExperience,
    GraphState,
    MemoryLink,
    Phase,
    ProfileSlot,
    SessionMemory,
    SlotStatus,
    SLOT_DESCRIPTIONS,
    SLOT_NAMES,
    UserProfile,
)

load_dotenv()


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

SLOT_QUESTION_PROMPT = """\
당신은 친절한 도서 큐레이터입니다.
사용자의 도서 선호 프로파일을 파악하기 위해 자연스러운 질문을 생성해야 합니다.

현재까지 수집된 프로필 정보:
{filled_profile}

이번에 파악해야 할 항목: {slot_name}
항목 설명: {slot_description}

이전 대화 맥락:
{conversation_context}

{retry_instruction}

요구사항:
- 한국어로 작성
- 이미 수집된 정보를 자연스럽게 참조하며 대화를 이어가세요
- 직접적이지 않고 대화체로 물어보세요
- 1~3문장 이내

질문:"""

RETRY_INSTRUCTION_TEMPLATE = """\
주의: 이전에 이 항목에 대해 질문했으나 사용자가 명확한 답변을 하지 않았습니다. (재시도 {retry_count}/{max_retries}회)
다른 각도에서, 더 구체적인 예시를 들어 질문을 재구성하세요."""

EXTRACT_SLOT_PROMPT = """\
당신은 도서 큐레이션을 위한 정보 추출기입니다.
사용자의 응답에서 아래 항목들에 대한 정보를 추출하세요.

추출 대상 항목:
{target_slots}

사용자 응답: "{user_message}"

대화 맥락:
{conversation_context}

아래 JSON 형식으로 응답하세요. 정보가 없는 항목은 null로 표시합니다:
{{
    "extracted": {{
        "<slot_name>": {{
            "value": "추출된 값 또는 null"
        }}
    }}
}}

JSON 응답:"""

SIMILAR_PROFILE_PRESENT_PROMPT = """\
이전에 비슷한 맥락에서 도서를 찾으셨던 기록이 있습니다.

이전 프로파일 요약:
{profile_summary}

이 정보를 사용자에게 자연스럽게 설명하고, 이번에도 비슷한 도서를 찾고 있는지 물어보세요.

요구사항:
- 한국어로 작성
- 이전 기록을 간결하게 정리
- 이번에도 같은 종류의 책을 찾는지 yes/no로 답할 수 있게 질문
- 2~4문장 이내

응답:"""

MATCH_CONFIRM_PROMPT = """\
사용자의 응답이 이전 프로파일과 같은 종류의 책을 찾고 있다는 긍정적 답변인지 판단하세요.

사용자 응답: "{user_message}"

아래 JSON 형식으로 응답하세요:
{{
    "is_match": true 또는 false,
    "reason": "판단 근거"
}}

JSON 응답:"""

BOOK_EXPERIENCE_PROMPT = """\
당신은 친절한 도서 큐레이터입니다.
사용자의 프로필 정보를 바탕으로, 비슷한 맥락에서 이전에 읽었던 책이 있는지 물어보세요.

현재 사용자 프로필:
{profile_summary}

기존에 수집된 독서 경험:
{existing_experiences}

요구사항:
- 한국어로 작성
- 프로필 맥락에 맞는 자연스러운 질문
- 책 제목과 간단한 소감을 함께 물어보세요
- 2~3문장 이내

질문:"""

EXTRACT_BOOK_EXPERIENCE_PROMPT = """\
사용자의 응답에서 이전 독서 경험을 추출하세요.

사용자 응답: "{user_message}"

아래 JSON 형식으로 응답하세요. 책 경험이 없으면 빈 리스트를 반환합니다:
{{
    "experiences": [
        {{
            "book_name": "책 이름",
            "impression": "소감/감상",
            "context": "어떤 맥락에서 읽었는지"
        }}
    ],
    "has_more": true 또는 false
}}

JSON 응답:"""

SUMMARY_PROMPT = """\
아래 정보를 바탕으로, 사용자가 현재 어떤 책을 원하고 있는지를 간략하게 요약하세요.

사용자 프로필:
{profile}

이전 독서 경험:
{book_experiences}

대화 히스토리 요약:
{conversation_summary}

요구사항:
- 한국어로 3~5문장의 간결한 요약문
- 사용자의 독서 목적, 선호 장르, 스타일, 난이도, 현재 상황을 자연스럽게 통합
- 구체적인 도서 추천을 위한 근거가 될 수 있도록 작성

요약문:"""

REFLECTION_PROMPT = """\
당신은 도서 큐레이션 시스템의 메모리 분석기입니다.
아래의 세션 메모리와 연결된 이전 메모리들을 분석하여 고도화된 인사이트를 추출하세요.

현재 세션 메모리:
- 프로필: {current_profile}
- 요약: {current_summary}
- 독서 경험: {current_experiences}

연결된 이전 메모리들 (1-hop):
{linked_memories}

다음의 심화 질문들에 답하며 reflection을 수행하세요:
1. 사용자의 독서 패턴에서 반복되는 테마나 욕구가 있는가?
2. 시간에 따른 독서 선호의 변화가 감지되는가?
3. 사용자가 명시적으로 말하지 않았지만 추론 가능한 잠재적 선호가 있는가?
4. 이전 경험과 현재 요구 사이의 연결점은 무엇인가?

인사이트를 JSON 형식으로 응답하세요:
{{
    "recurring_themes": "반복되는 테마/패턴",
    "preference_evolution": "선호 변화 추이",
    "latent_preferences": "잠재적 선호",
    "connections": "이전 경험과 현재 요구의 연결점",
    "reflection_summary": "종합 인사이트 (3~5문장)"
}}

JSON 응답:"""

LINK_JUDGMENT_PROMPT = """\
두 세션 메모리 사이에 의미 있는 연결(link)이 존재하는지 판단하세요.

세션 A (현재):
{session_a}

세션 B (후보):
{session_b}

다음 기준으로 판단하세요:
1. 같은 reading_goal을 가지고 있는가? (가중치: 0.3)
2. 비슷한 감정/상황 context인가? (가중치: 0.25)
3. 추천된 책 장르가 겹치는가? (가중치: 0.25)
4. 독서 스타일이나 난이도 선호가 유사한가? (가중치: 0.2)

아래 JSON 형식으로 응답하세요:
{{
    "should_link": true 또는 false,
    "link_reason": "연결 이유",
    "criteria_scores": {{
        "reading_goal_match": 0.0~1.0,
        "context_similarity": 0.0~1.0,
        "genre_overlap": 0.0~1.0,
        "style_similarity": 0.0~1.0
    }},
    "overall_strength": 0.0~1.0
}}

JSON 응답:"""


# ──────────────────────────────────────────────
# LLM utilities
# ──────────────────────────────────────────────

async def llm_call(llm: BaseChatModel, prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    return response.content.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def format_conversation_context(messages: list, last_n: int = 10) -> str:
    recent = messages[-last_n:] if len(messages) > last_n else messages
    lines = []
    for msg in recent:
        role = "사용자" if (isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human") else "큐레이터"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# MemoryStore
# ──────────────────────────────────────────────

class MemoryStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.sessions = self.client.get_or_create_collection(
            name="session_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self.links = self.client.get_or_create_collection(
            name="memory_links",
            metadata={"hnsw:space": "cosine"},
        )

    def save_session(self, memory: SessionMemory) -> None:
        embedding_text = self._build_embedding_text(memory)
        metadata = {
            "session_id": memory.session_id,
            "timestamp": memory.timestamp,
            "summary": memory.summary,
            "reflection": memory.reflection,
            "profile_json": memory.profile.model_dump_json(),
            "experiences_json": json.dumps(
                [e.model_dump() for e in memory.book_experiences], ensure_ascii=False
            ),
            "linked_ids": json.dumps(memory.linked_session_ids),
        }
        self.sessions.upsert(
            ids=[memory.session_id],
            documents=[embedding_text],
            metadatas=[metadata],
        )

    def search_similar_profiles(
        self,
        profile: UserProfile,
        k: int = 3,
        exclude_session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_text = profile.to_embedding_text()
        if not query_text.strip():
            return []

        results = self.sessions.query(query_texts=[query_text], n_results=k)
        similar = []
        if results and results["ids"] and results["ids"][0]:
            for i, sid in enumerate(results["ids"][0]):
                if exclude_session_id and sid == exclude_session_id:
                    continue
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else None
                similar.append({
                    "session_id": sid,
                    "summary": meta.get("summary", ""),
                    "reflection": meta.get("reflection", ""),
                    "profile": json.loads(meta.get("profile_json", "{}")),
                    "experiences": json.loads(meta.get("experiences_json", "[]")),
                    "distance": distance,
                    "timestamp": meta.get("timestamp", ""),
                })
        return similar

    def search_by_summary(
        self,
        summary: str,
        k: int = 5,
        exclude_session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not summary.strip():
            return []

        results = self.sessions.query(query_texts=[summary], n_results=k)
        similar = []
        if results and results["ids"] and results["ids"][0]:
            for i, sid in enumerate(results["ids"][0]):
                if exclude_session_id and sid == exclude_session_id:
                    continue
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else None
                similar.append({
                    "session_id": sid,
                    "summary": meta.get("summary", ""),
                    "reflection": meta.get("reflection", ""),
                    "profile": json.loads(meta.get("profile_json", "{}")),
                    "experiences": json.loads(meta.get("experiences_json", "[]")),
                    "distance": distance,
                    "timestamp": meta.get("timestamp", ""),
                })
        return similar

    def save_link(self, link: MemoryLink) -> None:
        link_id = f"{link.source_session_id}___{link.target_session_id}"
        self.links.upsert(
            ids=[link_id],
            documents=[link.link_reason],
            metadatas=[{
                "source_id": link.source_session_id,
                "target_id": link.target_session_id,
                "strength": link.strength,
                "reason": link.link_reason,
            }],
        )

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        try:
            result = self.sessions.get(ids=[session_id])
            if result and result["metadatas"] and result["metadatas"][0]:
                meta = result["metadatas"][0]
                return {
                    "session_id": session_id,
                    "summary": meta.get("summary", ""),
                    "reflection": meta.get("reflection", ""),
                    "profile": json.loads(meta.get("profile_json", "{}")),
                    "experiences": json.loads(meta.get("experiences_json", "[]")),
                    "timestamp": meta.get("timestamp", ""),
                }
        except Exception:
            pass
        return None

    def _build_embedding_text(self, memory: SessionMemory) -> str:
        parts = [memory.profile.to_embedding_text()]
        if memory.summary:
            parts.append(f"요약: {memory.summary}")
        if memory.reflection:
            parts.append(f"인사이트: {memory.reflection}")
        for exp in memory.book_experiences:
            parts.append(f"독서경험: {exp.book_name} - {exp.impression}  {exp.context}")
        return " | ".join(parts)


# ──────────────────────────────────────────────
# Node factory
# ──────────────────────────────────────────────

def create_nodes(llm: BaseChatModel, memory_store: MemoryStore):
    """LLM과 MemoryStore를 클로저로 캡처한 노드 함수들을 반환한다."""

    async def generate_slot_question(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        current_slot = state["current_slot"]

        if current_slot is None:
            current_slot = _get_next_empty_slot(profile)
            if current_slot is None:
                return {"current_slot": None}

        slot = profile.get_slot(current_slot)
        retry_instruction = ""
        if slot.retry_count > 0:
            retry_instruction = RETRY_INSTRUCTION_TEMPLATE.format(
                retry_count=slot.retry_count,
                max_retries=slot.MAX_RETRIES,
            )

        filled_info = _format_filled_profile(profile)
        prompt = SLOT_QUESTION_PROMPT.format(
            filled_profile=filled_info if filled_info else "아직 수집된 정보 없음",
            slot_name=current_slot,
            slot_description=SLOT_DESCRIPTIONS[current_slot],
            conversation_context=format_conversation_context(state["messages"]),
            retry_instruction=retry_instruction,
        )
        question = await llm_call(llm, prompt)
        return {
            "messages": [AIMessage(content=question)],
            "ai_response": question,
            "current_slot": current_slot,
        }

    async def process_slot_answer(state: GraphState) -> dict[str, Any]:
        user_msg = _get_last_human_message(state)
        profile = state["user_profile"]
        current_slot = state["current_slot"]

        profile = await _extract_slots_from_message(llm, user_msg, profile, state["messages"])

        if current_slot and profile.get_slot(current_slot).status == SlotStatus.EMPTY:
            slot = profile.get_slot(current_slot)
            slot.retry_count += 1
            if slot.retry_count >= slot.MAX_RETRIES:
                slot.status = SlotStatus.UNCLEAR
            profile.set_slot(current_slot, slot)

        next_slot = _get_next_empty_slot(profile)
        return {"user_profile": profile, "current_slot": next_slot}

    async def search_similar_profiles(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        similar = memory_store.search_similar_profiles(
            profile=profile, k=3, exclude_session_id=state["session_id"]
        )
        filtered = [s for s in similar if s.get("distance", 1.0) < 0.7]

        if not filtered:
            return {"similar_profiles": [], "phase": Phase.SLOT_FILLING}

        best = filtered[0]
        prompt = SIMILAR_PROFILE_PRESENT_PROMPT.format(
            profile_summary=best.get("summary", "정보 없음")
        )
        question = await llm_call(llm, prompt)
        return {
            "similar_profiles": filtered,
            "messages": [AIMessage(content=question)],
            "ai_response": question,
            "phase": Phase.MATCH_CONFIRM,
        }

    async def process_match_confirm(state: GraphState) -> dict[str, Any]:
        user_msg = _get_last_human_message(state)
        prompt = MATCH_CONFIRM_PROMPT.format(user_message=user_msg)
        result = parse_json_response(await llm_call(llm, prompt))
        is_match = result.get("is_match", False)

        if is_match and state["similar_profiles"]:
            best = state["similar_profiles"][0]
            matched_profile = UserProfile.from_dict(best["profile"])
            current_profile = state["user_profile"]
            matched_profile.set_slot("reading_goal", current_profile.get_slot("reading_goal"))
            prev_experiences = [BookExperience(**e) for e in best.get("experiences", [])]
            transition_msg = "이전에 비슷한 맥락으로 도서를 찾으셨던 기록이 있어서 해당 프로필을 불러왔습니다!"
            return {
                "user_profile": matched_profile,
                "matched_profile_id": best["session_id"],
                "book_experiences": prev_experiences,
                "current_slot": None,
                "messages": [AIMessage(content=transition_msg)],
                "ai_response": transition_msg,
                "phase": Phase.BOOK_EXPERIENCE,
            }
        else:
            next_slot = _get_next_empty_slot(state["user_profile"])
            return {
                "similar_profiles": [],
                "current_slot": next_slot,
                "phase": Phase.SLOT_FILLING,
            }

    async def ask_book_experience(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        existing = state.get("book_experiences", [])
        prompt = BOOK_EXPERIENCE_PROMPT.format(
            profile_summary=profile.to_embedding_text(),
            existing_experiences=(
                "\n".join(f"- {e.book_name}: {e.impression} {e.context}" for e in existing)
                if existing else "없음"
            ),
        )
        question = await llm_call(llm, prompt)
        return {
            "messages": [AIMessage(content=question)],
            "ai_response": question,
            "asked_book_experience": True,
        }

    async def process_book_experience(state: GraphState) -> dict[str, Any]:
        user_msg = _get_last_human_message(state)
        prompt = EXTRACT_BOOK_EXPERIENCE_PROMPT.format(user_message=user_msg)
        result = parse_json_response(await llm_call(llm, prompt))

        experiences = list(state.get("book_experiences", []))
        for exp_data in result.get("experiences", []):
            experiences.append(BookExperience(**exp_data))

        has_more = result.get("has_more", False)
        return {
            "book_experiences": experiences,
            "phase": Phase.BOOK_EXPERIENCE if has_more else Phase.SUMMARY,
        }

    async def generate_summary(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        experiences = state.get("book_experiences", [])
        prompt = SUMMARY_PROMPT.format(
            profile=profile.to_embedding_text(),
            book_experiences=(
                "\n".join(f"- {e.book_name}: {e.impression} {e.context}" for e in experiences)
                if experiences else "없음"
            ),
            conversation_summary=format_conversation_context(state["messages"], last_n=10),
        )
        summary = await llm_call(llm, prompt)
        return {"summary": summary, "phase": Phase.REFLECTION}

    async def perform_reflection(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        summary = state["summary"]
        experiences = state.get("book_experiences", [])
        session_id = state["session_id"]

        similar = memory_store.search_by_summary(summary=summary, k=5, exclude_session_id=session_id)
        links: list[MemoryLink] = []
        linked_session_ids: list[str] = []

        current_experiences_text = (
            "\n".join(f"- {e.book_name}: {e.impression} {e.context}" for e in experiences)
            if experiences else "없음"
        )
        current_session_text = (
            f"프로필: {profile.to_embedding_text()}\n"
            f"요약: {summary}\n"
            f"독서 경험: {current_experiences_text}"
        )

        for candidate in similar:
            cand_profile = candidate.get("profile", {})
            cand_profile_text = " | ".join(
                f"{SLOT_DESCRIPTIONS.get(k, k)}: {v.get('value', '')}"
                for k, v in cand_profile.items()
                if isinstance(v, dict) and v.get("value")
            )
            cand_experiences = candidate.get("experiences", [])
            cand_experiences_text = (
                "\n".join(
                    f"- {e.get('book_name', '')}: {e.get('impression', '')} {e.get('context', '')}"
                    for e in cand_experiences
                )
                if cand_experiences else "없음"
            )
            candidate_text = (
                f"프로필: {cand_profile_text}\n"
                f"요약: {candidate.get('summary', '')}\n"
                f"독서 경험: {cand_experiences_text}"
            )

            prompt = LINK_JUDGMENT_PROMPT.format(
                session_a=current_session_text,
                session_b=candidate_text,
            )
            result = parse_json_response(await llm_call(llm, prompt))

            if result.get("should_link", False):
                link = MemoryLink(
                    source_session_id=session_id,
                    target_session_id=candidate["session_id"],
                    link_reason=result.get("link_reason", ""),
                    strength=result.get("overall_strength", 0.0),
                )
                links.append(link)
                linked_session_ids.append(candidate["session_id"])
                memory_store.save_link(link)

        linked_memories = []
        for lid in linked_session_ids:
            sess = memory_store.get_session(lid)
            if sess:
                linked_memories.append(sess)

        prompt = REFLECTION_PROMPT.format(
            current_profile=profile.to_embedding_text(),
            current_summary=summary,
            current_experiences=current_experiences_text,
            linked_memories=(
                "\n---\n".join(
                    f"세션 {m['session_id']}:\n요약: {m.get('summary', '')}\nReflection: {m.get('reflection', '')}"
                    for m in linked_memories
                )
                if linked_memories else "연결된 이전 메모리 없음"
            ),
        )
        reflection_result = parse_json_response(await llm_call(llm, prompt))
        reflection_text = reflection_result.get(
            "reflection_summary", json.dumps(reflection_result, ensure_ascii=False)
        )

        session_memory = SessionMemory(
            session_id=session_id,
            profile=profile,
            book_experiences=experiences,
            summary=summary,
            reflection=reflection_text,
            linked_session_ids=linked_session_ids,
        )
        memory_store.save_session(session_memory)

        done_msg = (
            f"프로필 분석이 완료되었습니다!\n\n"
            f"📋 요약: {summary}\n\n"
            f"💡 인사이트: {reflection_text}"
        )
        return {
            "reflection": reflection_text,
            "links": links,
            "messages": [AIMessage(content=done_msg)],
            "ai_response": done_msg,
            "phase": Phase.DONE,
        }

    async def _extract_slots_from_message(
        llm_inst: BaseChatModel,
        user_msg: str,
        profile: UserProfile,
        messages: list,
    ) -> UserProfile:
        target_slots = profile.empty_slots()
        if not target_slots:
            return profile

        slot_desc = "\n".join(f"- {s}: {SLOT_DESCRIPTIONS[s]}" for s in target_slots)
        prompt = EXTRACT_SLOT_PROMPT.format(
            target_slots=slot_desc,
            user_message=user_msg,
            conversation_context=format_conversation_context(messages),
        )
        result = parse_json_response(await llm_call(llm_inst, prompt))
        extracted = result.get("extracted", {})

        for slot_name, info in extracted.items():
            if slot_name not in SLOT_NAMES or info is None:
                continue
            value = info.get("value")
            if value and value != "null" and value.strip():
                slot = profile.get_slot(slot_name)
                slot.value = value
                slot.status = SlotStatus.FILLED
                profile.set_slot(slot_name, slot)

        return profile

    return {
        "generate_slot_question": generate_slot_question,
        "process_slot_answer": process_slot_answer,
        "search_similar_profiles": search_similar_profiles,
        "process_match_confirm": process_match_confirm,
        "ask_book_experience": ask_book_experience,
        "process_book_experience": process_book_experience,
        "generate_summary": generate_summary,
        "perform_reflection": perform_reflection,
    }


# ──────────────────────────────────────────────
# Edge routing functions
# ──────────────────────────────────────────────

def route_after_slot_processing(state: GraphState) -> str:
    profile = state["user_profile"]
    similar_profiles = state.get("similar_profiles")
    phase = state.get("phase")

    if profile.all_filled_or_unclear():
        return "ask_book_experience"

    if (
        profile.reading_goal_filled()
        and similar_profiles is None
        and phase != Phase.MATCH_CONFIRM
    ):
        return "search_similar_profiles"

    return "generate_slot_question"


def route_after_similar_search(state: GraphState) -> str:
    phase = state.get("phase")
    if phase == Phase.MATCH_CONFIRM:
        return "process_match_confirm"
    return "generate_slot_question"


def route_after_match_confirm(state: GraphState) -> str:
    phase = state.get("phase")
    if phase == Phase.BOOK_EXPERIENCE:
        return "ask_book_experience"
    return "generate_slot_question"


def route_after_book_experience(state: GraphState) -> str:
    phase = state.get("phase")
    if phase == Phase.SUMMARY:
        return "generate_summary"
    return "ask_book_experience"


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _get_last_human_message(state: GraphState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return msg.content
    return ""


def _get_next_empty_slot(profile: UserProfile) -> str | None:
    remaining = profile.empty_slots()
    return remaining[0] if remaining else None


def _format_filled_profile(profile: UserProfile) -> str:
    lines = []
    for name in SLOT_NAMES:
        slot = profile.get_slot(name)
        if slot.status == SlotStatus.FILLED:
            lines.append(f"- {SLOT_DESCRIPTIONS[name]}: {slot.value}")
    return "\n".join(lines) if lines else ""
