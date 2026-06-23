from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from chromadb.utils import embedding_functions

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

[주의]
  - 반드시 {{"value": "..."}} 형태를 유지하세요.
  - <slot_name>의 값은 반드시 객체 {{"value": "문자열"}} 이어야 합니다. 문자열을 직접 값으로 쓰지 마세요.
  - value는 반드시 문자열. 객체나 배열 금지. 정보 없으면 null

추출 대상 항목:
{target_slots}

사용자 응답: "{user_message}"

대화 맥락:
{conversation_context}

아래 JSON 반드시 형식으로 응답하세요. 정보가 없는 항목은 null로 표시합니다:
{{
    "extracted": {{
        "<slot_name>": {{
            "value": "추출된 문자열"
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
- 인삿말 금지

응답:"""

# ──────────────────────────────────────────────
# 4-1. 매칭 후 현재 상황 질문
# ──────────────────────────────────────────────
POST_MATCH_CONTEXT_PROMPT = """\
당신은 친절한 도서 큐레이터입니다.
사용자가 이전에 비슷한 맥락에서 도서를 찾았던 기록이 있어 해당 프로필을 불러왔습니다.
이전 프로필의 선호에서 달라진 점이 있는지를 자연스럽게 물어보세요.
 
불러온 이전 프로필:
{matched_profile}
 
이전 대화 맥락:
{conversation_context}
 
질문에 포함할 관점:
- 이전과 비교하여 독서 스타일, 장르,난이도에서 달라진 점이 있는지
- 현재 어떤 상황이나 감정에서 책을 읽으려 하는지
 
요구사항:
- 한국어로 작성
- 선호의 변화를 물어보세요
- 부담 없이 대답할 수 있도록 대화체로 작성
- 1~3문장 이내
- 인삿말 금지
 
질문:"""

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
- 프로필 정보를 기반으로 간략한 요약 제시후, 질문
- 프로필 맥락에 맞는 자연스러운 질문
- 예시를 들지 말 것
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


요구사항:
- 한국어로 3~5문장의 간결한 요약문
- 사용자의 독서 목적, 선호 장르, 스타일, 난이도, 현재 상황을 자연스럽게 통합
- 구체적인 도서 추천을 위한 근거가 될 수 있도록 작성
- 반드시 포함해야 할 요소:
  (1) 책을 찾게 된 상황적·감정적 맥락 (왜 지금 이 책이 필요한가)
  (2) 원하는 책의 구체적 특성 (장르, 테마, 분위기, 문체, 난이도)
  (3) 이전 독서 경험이 있다면 그것과의 관계
- 이 요약문이 향후 유사한 독서 요구를 가진 다른 세션을 검색하는 데 사용되므로, 감정적 맥락과 독서 목적을 명확하게 서술하세요

요약문:"""

REFLECTION_PROMPT = """\
아래의 현재 세션 정보와 연결된 이전 메모리들을 분석하여, 사용자의 프로필 정보(독서 목적, 선호 장르, 독서 스타일, 난이도, 현재 맥락)만으로는 파악할 수 없는 **추가적인 독서 선호 특성**을 최대 5가지 도출하세요.

중요: 프로필에 이미 명시된 정보(장르, 난이도, 스타일 등)를 단순 반복하지 마세요.
연결된 이전 메모리와의 비교·대조를 통해서만 추론 가능한, 프로필에는 담기지 않은 새로운 선호 차원와 숨겨진 욕구를 도출하는 것이 목적입니다.

현재 세션 프로필:
{current_profile}

현재 세션 요약:
{current_summary}

현재 세션 독서 경험:
{current_experiences}

연결된 이전 메모리들 (1-hop):
{linked_memories}

다음 두 단계로 추론하세요.

[1단계: 연결 메모리 기반 분석]
연결된 이전 메모리들과 현재 세션을 비교·대조하여 다음을 분석하세요:
- 유사한 감정/상황 맥락에서 사용자가 어떤 다른 선택을 했는가?
- 이전 독서 경험에서 만족했던 요소는 무엇인가?
- 현재 프로필에는 드러나지 않지만, 연결된 메모리를 통해 추론 가능한 숨겨진 선호는 무엇인가?

[2단계: 인사이트 도출]
1단계의 분석 결과를 바탕으로, 다음 4가지 관점에서 프로필에 없는 추가 선호를 도출하세요:
1. 서사 구조: 사용자가 선호할 가능성이 높은 서사 구조 (예: 성장 서사, 반전 구조 등)
2. 톤과 분위기: 사용자가 원하는 책의 톤 (예: 유머러스, 따뜻한, 긴장감 있는 등)
3. 소재와 테마: 프로필의 장르 안에서 특히 끌릴 세부 소재나 테마 (예: 동료 간 유대, 자아 탐색 등)
4. 독서 후 기대 감정: 사용자가 기대하는 독서 후 감정 (예: 카타르시스, 위로, 동기부여 등)

작성 기준:
- 제공된 정보에서 근거를 찾을 수 없는 항목은 도출하지 마세요. 5가지 모두 채울 필요는 없습니다.
- 프로필 슬롯(독서 목적, 장르, 스타일, 난이도, 맥락)에 이미 담긴 내용은 절대 단순 반복하지 마세요.
- 2단계 인사이트에 이전 메모리와의 비교 내용, "이전에는 ~했지만" 등의 대비 표현은 출력에 포함하지 마세요.
- 현재 선호 방향만 서술하세요. 반대 방향의 키워드가 포함되면 유사도 검색이 왜곡됩니다.


아래 JSON 형식으로 응답하세요:
{{
    "analysis": "1단계 분석 내용을 자유롭게 서술",
    "insights": [
        "2단계 인사이트 1",
        "2단계 인사이트 2",
        "2단계 인사이트 3",
        "2단계 인사이트 4",
        "2단계 인사이트 5"
    ]
}}

JSON 응답:"""

MEMORY_LINK_PROMPT = """\
당신은 도서 큐레이션 시스템의 메모리 link 생성 에이전트입니다.
새로운 세션 메모리의 요약과 유사한 이웃 메모리들의 요약을 비교하여, 어떤 이웃 메모리와 연결(link)을 생성해야 하는지 판단하세요.

현재 세션 요약:
{current_summary}

유사한 이웃 메모리들:
{nearest_neighbors}

절대 규칙:
- 반드시 세션 메모리의 요약과 유사한 이웃 메모리들의 요약만을 근거로 판단한다.
- 요약에 명시적으로 쓰여 있지 않은 동기나 감정을 추론하여 연결 근거로 사용하지 마세요.
- 다음은 link의 근거가 될 수 없다:
  · 난이도, 독서 스타일, 문체 선호 등 표면적 속성의 유사성
  · 주제를 "~적 관점", "~적 접근"으로 재표현하여 공통점을 만드는 것
- 기본값은 should_link: false 입니다. 명확한 근거가 있을 때만 true로 판단하세요.

판단 기준 — 다음을 모두 충족해야 연결을 생성하세요:
1. 필수 조건: 책을 찾게 된 동기, 목적 또는 상황적 배경이 겹치는가?
2. 추가 조건: 위 조건을 충족한 상태에서, 이웃 메모리를 참고하면 현재 사용자가 원하는 책의 본질을 더 깊이 이해할 수 있는가?
   - 유사한 감정 상황에서 다른 장르, 분야, 난이도를 선택한 경험이 있어 대비가 가능한 경우
   - 유사한 감정 상황에서의 이전 독서 경험이 현재 원하는 책의 특성을 구체화하는 경우

비연결 판단 예시 (should_link: false):
- "경제 공부 후 위로받을 에세이" vs "직장 소통을 심리학으로 이해" → 독서 동기가 다름 (위로 vs 지적 탐구)
- "의사결정 편향 이해" vs "직장 관계 심리학 이해" → 같은 심리학이지만 동기와 맥락이 완전히 다름 (인지 편향 vs 대인관계)

연결 판단 예시 (should_link: true):
- "직장 번아웃 → 판타지로 도피" vs "직장 갈등 → 자기계발서로 대처" → 직장 스트레스 해소라는 구체적 감정 동기 공유, 다른 전략 비교 가능
- "이별 후 위로받을 소설" vs "친구와 소원해진 후 관계 문학" → 관계 상실이라는 감정적 맥락 공유
- "이직 준비 중 재정 점검" vs "이직 후 소비 패턴 정리" → 커리어 전환이라는 상황적 맥락 공유, 동일 목적의 전후 비교 가능

아래 JSON 형식으로 응답하세요:
{{
    "evolution_decisions": [
        {{
            "neighbor_session_id": "이웃 세션 ID",
            "should_link": false,
            "link_reason": "현재 세션의 이해에 어떻게 도움이 되는지 (1~2문장)",
            "link_strength": 0.0~1.0
        }}
    ]
}}

JSON 응답:"""

single_query_prompt = """
당신은 도서 추천 시스템의 검색 질의 생성기입니다.

아래는 사용자와 AI 큐레이터의 전체 대화입니다.

{conversation}

목표:
- 사용자의 현재 관심사
- 독서 목적
- 선호하는 주제
- 원하는 난이도
- 이전 독서 경험

를 파악하여 도서 검색을 위한 query를 생성하세요.

조건:
- 2~3문장으로 작성
- 자연어 형태로 작성
- 발화문 형태로 작성
- 책 제목은 생성하지 말 것
- 사용자의 현재 관심사를 중심으로 작성
- 벡터 검색(RAG)에 바로 사용할 수 있도록 핵심 정보만 포함

출력은 query만 작성하세요.

query:"""


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
        # 한국어 특화 임베딩 모델 (HuggingFace sentence-transformers)

        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="dragonkue/bge-m3-ko"
        )

        self.sessions = self.client.get_or_create_collection(
            name="session_memories",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )
        self.links = self.client.get_or_create_collection(
            name="memory_links",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
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
        parts = [f"요약: {memory.summary}"]
        if memory.reflection:
            parts.append(f"인사이트: {memory.reflection}")
        #parts = [memory.profile.to_embedding_text()]
        #if memory.summary:
        #    parts.append(f"요약: {memory.summary}")
        #if memory.reflection:
        #    parts.append(f"인사이트: {memory.reflection}")
        #for exp in memory.book_experiences:
        #    parts.append(f"독서경험: {exp.book_name} - {exp.impression}  {exp.context}")
        return " | ".join(parts)


# ──────────────────────────────────────────────
# Node factory
# ──────────────────────────────────────────────

def create_nodes(llm: BaseChatModel, memory_store: MemoryStore):
    """LLM과 MemoryStore를 클로저로 캡처한 노드 함수들을 반환한다."""

    async def generate_slot_question(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        current_slot = state["current_slot"]

        # 방어 코드: 라우팅에서 걸러지므로 정상적으로는 도달하지 않음
        if current_slot is None:
            current_slot = _get_next_empty_slot(profile)
            if current_slot is None:
                # 진짜 모든 슬롯이 완료됨 — 빈 응답 반환 (라우팅이 처리)
                return {"current_slot": None}

        slot = profile.get_slot(current_slot)

        # 매칭 후 current_context 질문인 경우 → 전용 프롬프트 사용
        if (
            current_slot == "current_context"
            and state.get("matched_profile_id") is not None
        ):
            # 이전 프로필 정보를 similar_profiles에서 가져옴
            matched = state["similar_profiles"][0]
            matched_profile_text = _format_profile_from_dict(matched.get("profile", {}))

            prompt = POST_MATCH_CONTEXT_PROMPT.format(
                matched_profile=matched_profile_text,
                conversation_context=format_conversation_context(state["messages"], last_n=100),
            )
        else:
            # 재시도 안내 구성
            retry_instruction = ""
            if slot.retry_count > 0:
                retry_instruction = RETRY_INSTRUCTION_TEMPLATE.format(
                    retry_count=slot.retry_count,
                    max_retries=slot.MAX_RETRIES,
                )

            # 이미 채워진 프로필 정보 포매팅
            filled_info = _format_filled_profile(profile)

            prompt = SLOT_QUESTION_PROMPT.format(
                filled_profile=filled_info if filled_info else "아직 수집된 정보 없음",
                slot_name=current_slot,
                slot_description=SLOT_DESCRIPTIONS[current_slot],
                conversation_context=format_conversation_context(state["messages"], last_n=100),
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

        # 매칭 후 응답 처리: 사용자가 언급하지 않은 빈 슬롯은 이전 프로필로 채움
        if state.get("matched_profile_id") and state.get("similar_profiles"):
            matched_data = state["similar_profiles"][0].get("profile", {})
            for slot_name in SLOT_NAMES:
                if profile.get_slot(slot_name).status == SlotStatus.EMPTY:
                    matched_slot = matched_data.get(slot_name, {})
                    if isinstance(matched_slot, dict) and matched_slot.get("value"):
                        slot = ProfileSlot(
                            value=matched_slot["value"],
                            status=SlotStatus.FILLED,
                        )
                        profile.set_slot(slot_name, slot)

        next_slot = _get_next_empty_slot(profile)
        return {"user_profile": profile, "current_slot": next_slot}

    async def search_similar_profiles(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        similar = memory_store.search_similar_profiles(
            profile=profile, k=3, exclude_session_id=state["session_id"]
        )
        filtered = [s for s in similar if s.get("distance", 1.0) < 0.5]

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
            prev_experiences = [BookExperience(**e) for e in best.get("experiences", [])]
            transition_msg = "이전에 비슷한 맥락으로 도서를 찾으셨던 기록이 있어서 해당 프로필을 불러왔습니다!"
            return {
                "matched_profile_id": best["session_id"],
                "book_experiences": prev_experiences,
                "current_slot": "current_context",
                "messages": [AIMessage(content=transition_msg)],
                "ai_response": transition_msg,
                "phase": Phase.SLOT_FILLING,
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
        chat_history = format_conversation_context(state["messages"], last_n=100)
        prompt = single_query_prompt.format(
            conversation = chat_history
        )
        
        summary = await llm_call(llm, prompt)
        return {"summary": summary, "phase": Phase.REFLECTION}

    async def perform_reflection(state: GraphState) -> dict[str, Any]:
        profile = state["user_profile"]
        summary = state["summary"]
        experiences = state.get("book_experiences", [])
        session_id = state["session_id"]
 
        # 1) summary 기반으로 유사 메모리 검색
        similar = memory_store.search_by_summary(
            summary=summary,
            k=5,
            exclude_session_id=session_id,
        )
 
        # 현재 세션 독서 경험 텍스트 구성
        current_experiences_text = (
            "\n".join(f"- {e.book_name}: {e.impression}" for e in experiences)
            if experiences
            else "없음"
        )
 
        # 2) 이웃 메모리 요약 포매팅 + 단일 호출로 링크 판단
        links: list[MemoryLink] = []
        linked_session_ids: list[str] = []
 
        if similar:
            neighbor_texts = []
            for idx, cand in enumerate(similar):
                neighbor_texts.append(
                    f"[이웃 {idx + 1}] 세션 ID: {cand['session_id']}\n"
                    f"  요약: {cand.get('summary', '')}"
                )
 
            evolution_prompt = MEMORY_LINK_PROMPT.format(
                current_summary=summary,
                nearest_neighbors="\n---\n".join(neighbor_texts),
            )
            evolution_result = parse_json_response(await llm_call(llm, evolution_prompt))
 
            decisions = evolution_result.get("evolution_decisions", [])
            for decision in decisions:
                if decision.get("should_link", False):
                    target_id = decision.get("neighbor_session_id", "")
                    valid_ids = {c["session_id"] for c in similar}
                    if target_id in valid_ids:
                        link = MemoryLink(
                            source_session_id=session_id,
                            target_session_id=target_id,
                            link_reason=decision.get("link_reason", ""),
                            strength=decision.get("link_strength", 0.0),
                        )
                        links.append(link)
                        linked_session_ids.append(target_id)
                        memory_store.save_link(link)
 
        # 3) 1-hop 연결 메모리 수집 + link reason 매핑
        link_reasons = {
            link.target_session_id: link.link_reason for link in links
        }
        linked_memories = []
        for lid in linked_session_ids:
            sess = memory_store.get_session(lid)
            if sess:
                sess["link_reason"] = link_reasons.get(lid, "")
                linked_memories.append(sess)
 
        # 4) Reflection 수행 — 5가지 인사이트 리스트 추출
        # linked_memories 포매팅 (연결 이유 + 프로필 + 요약 + 독서경험 + 이전 reflection)
        def _format_linked_memory(m: dict) -> str:
            parts = [f"세션 {m['session_id']}:"]
            # 연결 이유
            if m.get("link_reason"):
                parts.append(f"연결 이유: {m['link_reason']}")
            # 프로필 정보
            m_profile = m.get("profile", {})
            profile_items = []
            for k, v in m_profile.items():
                if isinstance(v, dict) and v.get("value"):
                    desc = SLOT_DESCRIPTIONS.get(k, k)
                    profile_items.append(f"{desc}: {v['value']}")
            if profile_items:
                parts.append(f"프로필: {' | '.join(profile_items)}")
            # 요약
            if m.get("summary"):
                parts.append(f"요약: {m['summary']}")
            # 독서 경험
            m_exps = m.get("experiences", [])
            if m_exps:
                exp_lines = [f"  - {e.get('book_name', '')}: {e.get('impression', '')}" for e in m_exps]
                parts.append(f"독서 경험:\n" + "\n".join(exp_lines))
            # 이전 reflection
            if m.get("reflection"):
                parts.append(f"이전 인사이트: {m['reflection']}")
            return "\n".join(parts)

        if len(linked_session_ids) > 0:
            prompt = REFLECTION_PROMPT.format(
                current_profile=profile.to_embedding_text(),
                current_summary=summary,
                current_experiences=current_experiences_text,
                linked_memories=(
                    "\n---\n".join(
                        _format_linked_memory(m) for m in linked_memories
                    )
                    if linked_memories
                    else "연결된 이전 메모리 없음"
                ),
            )
            reflection_raw = await llm_call(llm, prompt)
    
            # 리스트 형태 파싱
            reflection_parsed = parse_json_response(reflection_raw)
            if isinstance(reflection_parsed, list):
                reflection_list = reflection_parsed
            else:
                # fallback: 파싱 실패 시 원문을 단일 항목 리스트로
                reflection_list = [reflection_raw.strip()]
    
            reflection_text = " ".join(reflection_list)
            
        else:
            reflection_text = ""
 
        # 5) ChromaDB에 저장
        session_memory = SessionMemory(
            session_id=session_id,
            profile=profile,
            book_experiences=experiences,
            summary=summary,
            reflection=reflection_text,
            linked_session_ids=linked_session_ids,
        )
        memory_store.save_session(session_memory)
 
        # 6) 완료 메시지
        
        done_msg = (
            f"프로필 분석이 완료되었습니다!\n\n"
            f"📋 요약: {summary}\n\n"
            f"💡 인사이트:\n{reflection_text}"
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


def _format_profile_from_dict(profile_dict: dict) -> str:
    """ChromaDB에서 가져온 dict 형태의 프로필을 문자열로 포매팅."""
    lines = []
    for name in SLOT_NAMES:
        if name in profile_dict:
            slot_data = profile_dict[name]
            if isinstance(slot_data, dict) and slot_data.get("value"):
                lines.append(f"- {SLOT_DESCRIPTIONS[name]}: {slot_data['value']}")
    return "\n".join(lines) if lines else ""
