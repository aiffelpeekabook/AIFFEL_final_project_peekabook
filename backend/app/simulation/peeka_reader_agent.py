"""
PeekaReader: 페르소나 DNA 기반 사용자 시뮬레이션 에이전트

CRS의 슬롯 질문에 자동 응답하고 추천 결과를 도서별로 평가함.
ReAct 패턴(Thought + Action)으로 발화를 생성함.
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI

from app.config import LLM_MODEL


# OpenAI 클라이언트는 모듈 임포트 시점에 만들지 않고 지연 초기화함
# (테스트 시 환경변수 주입 순서 문제 회피)
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """OpenAI 클라이언트를 지연 초기화하여 반환함"""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def extract_session_dna(full_persona: dict, session_id: int) -> dict:
    """
    페르소나 풀에서 특정 세션의 DNA를 추출함.

    고정 속성 + 세션별 DNA + 누적 메모리(derived_preferences)를 합산함.
    반환값은 PeekaReader / PeekaJudge가 받는 persona 딕셔너리 형태와 동일함.

    Args:
        full_persona: 페르소나 전체 dict (demographics, sessions, long_term_memory 포함)
        session_id:   추출할 세션 번호 (1부터 시작)

    Returns:
        Judge 루브릭 5개 축 + 고정 속성을 합친 DNA dict
    """
    session = full_persona["sessions"][session_id - 1]
    memory  = full_persona["long_term_memory"]

    # Judge 루브릭 5개 축 + 고정 속성
    dna = {
        "reading_goal":     session["reading_goal"],
        "preferred_genre":  session["preferred_genre"],
        "reading_style":    session["reading_style"],
        "difficulty_level": session["difficulty_level"],
        "current_context":  session["current_context"],
        "demographics":     full_persona["demographics"],
        "speaking_style":   full_persona["speaking_style"],
        "disliked":         full_persona["disliked"],
        "pain_points":      full_persona["pain_points"],
    }

    # 누적 취향이 있으면 current_context에 자연스럽게 덧붙임
    # → PeekaReader가 "지난번에 어려운 책을 받았어서요..." 발화를 만들어냄
    if memory["derived_preferences"]:
        prefs_str = " / ".join(memory["derived_preferences"])
        dna["current_context"] += f" (이전 경험: {prefs_str})"

    return dna


class PeekaReaderAgent:
    """
    CRS 슬롯 질문에 페르소나 DNA 기반으로 자동 응답하고,
    추천 결과(3권)를 도서별로 평가하는 ReAct 에이전트.
    """

    ANSWER_SYSTEM = """\
당신은 도서 추천 챗봇과 대화하는 실제 사용자를 연기하는 에이전트입니다.

## 당신의 페르소나 (DNA)
{persona_str}

## 행동 규칙 (ReAct)
먼저 속으로 한 문장 생각(Thought)한 뒤, 그 생각에 기반해 답(Action)하세요. 아래 형식을 반드시 따르세요.
- Thought: "나는 [DNA 특성]이고 [현재 상황]이라서 [원하는 것]이 필요하다"
- Action: 챗봇 질문에 대한 자연스러운 답변

## 답변 규칙
1. 페르소나에 충실하되, DNA 단어를 그대로 복붙하지 말고 자연스럽게 풀어 쓰세요.
   예) "경제 교양서" -> "실생활에 도움 되는 책"
2. speaking_style을 지켜 실제 사람처럼 구어체로, 1~3문장 이내로.
3. 페르소나에 없는 정보는 DNA와 일관된 방향으로 자연스럽게 지어내세요.
4. 챗봇의 질문에만 답하세요. 먼저 책 추천을 요청하지 마세요.

## 출력 형식 (JSON)
{{"thought": "속마음 한 문장", "utterance": "실제 발화"}}
"""

    EVAL_SYSTEM = """\
당신은 아래 DNA를 가진 도서관 이용자입니다.

## 당신의 페르소나 (DNA)
{persona_str}

## 과제
추천된 도서 각각이 당신의 DNA와 맞는지 판단하세요.

## 평가 방법
아래 각 도서의 소개글을 읽고 DNA 기준으로 직접 판단하세요.
추천 이유나 도서관 정보는 무시하세요.

## 추천 도서 및 소개글
{book_intros_str}

## 판단 기준 (권당)
- 관심 장르/주제와 맞는가
- 난이도가 내 수준에 맞는가
- 현재 상황(목적)에 적합한가

## 출력 형식 (JSON만 출력)
{{
  "books_evaluated": [
    {{"title": "책 제목", "match": true, "reason": "내 입장에서 한 문장"}}
  ],
  "overall_reason": "전체 소감 한 문장"
}}
"""

    EVAL_SYSTEM_FALLBACK = """\
당신은 아래 DNA를 가진 도서관 이용자입니다.

## 당신의 페르소나 (DNA)
{persona_str}

## 과제
시스템이 추천한 도서 각각이 당신의 DNA와 맞는지 직접 판단하세요.

## 판단 기준 (권당)
- 관심 장르/주제와 맞는가
- 난이도가 내 수준에 맞는가
- 현재 상황(목적)에 적합한가

## 주의
판단 기준은 오직 위의 DNA입니다.
추천 이유 문구나 도서관 정보는 무시하세요.
도서 제목과 저자만 보고 DNA 기준으로 직접 판단하세요.

## 출력 형식 (JSON만 출력)
{{
  "books_evaluated": [
    {{"title": "책 제목", "match": true, "reason": "내 입장에서 한 문장"}}
  ],
  "overall_reason": "전체 소감 한 문장"
}}
"""

    EVAL_SYSTEM_BOTH = """\
당신은 아래 DNA를 가진 도서관 이용자입니다.

## 당신의 페르소나 (DNA)
{persona_str}

## 과제
추천된 도서 각각이 당신의 DNA와 맞는지 판단하세요.

## 평가 방법
아래 두 가지 정보를 함께 참고하세요.
1. 도서 소개글 (출판사 작성 고정 텍스트)
2. 추천 이유 (큐레이터 AI 작성)

단, 판단의 주된 근거는 DNA와 도서 소개글이며,
추천 이유는 보조 참고 자료로만 활용하세요.

## 도서 소개글
{book_intros_str}

## 판단 기준 (권당)
- 관심 장르/주제와 맞는가
- 난이도가 내 수준에 맞는가
- 현재 상황(목적)에 적합한가

## 출력 형식 (JSON만 출력)
{{
  "books_evaluated": [
    {{"title": "책 제목", "match": true, "reason": "DNA 기준으로 한 문장"}}
  ],
  "overall_reason": "전체 소감 한 문장"
}}
"""

    def __init__(self, persona_id: str, persona: dict, verbose: bool = True):
        self.persona_id  = persona_id
        self.persona     = persona
        self.verbose     = verbose
        self.history: list = []
        self.turn_count  = 0
        self.persona_str = "\n".join(f"- {k}: {v}" for k, v in persona.items())

    def answer(self, question: str) -> dict:
        """CRS 슬롯 질문에 DNA 기반 자동 응답 (최대 3회 재시도)"""
        self.turn_count += 1
        self.history.append({"role": "user", "content": question})

        thought   = ""
        utterance = ""

        for attempt in range(3):
            resp = _get_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system",
                     "content": self.ANSWER_SYSTEM.format(
                         persona_str=self.persona_str)},
                    *self.history
                ],
                temperature=0.3,  # 원래 0.7
                # seed=42,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content  # 응답 미생성 시 fallback
            raw = content.strip() if content else "{}"

            try:
                parsed    = json.loads(raw)
                thought   = parsed.get("thought", "")
                utterance = parsed.get("utterance", "")
            except json.JSONDecodeError:
                thought, utterance = "", ""

            if utterance:
                break

            if self.verbose and attempt < 2:
                print(f"  [재시도 {attempt + 1}] utterance 비어있음 — 재생성 중...")

        if not utterance:
            utterance = "잘 모르겠어요."
            thought   = ""
            if self.verbose:
                print(f"  [경고] utterance 생성 실패 — 기본값 사용")

        # history에는 발화만 누적 (CRS가 받는 건 utterance뿐)
        self.history.append({"role": "assistant", "content": utterance})

        if self.verbose:
            print(f"\n[Turn {self.turn_count}]")
            print(f"  CSR     : {question}")
            print(f"  THOUGHT : {thought}")
            print(f"  USER    : {utterance}")

        return {"thought": thought, "utterance": utterance}

    def evaluate(self, recommendation_text: str,
                 book_intros: Optional[dict] = None,
                 mode: str = "both") -> dict:
        """
        mode:
          "recommendation" — CRS 추천 이유만 보고 평가
          "book_intro"     — 서지 소개글만 보고 평가 (기본값)
          "both"           — 추천 이유 + 서지 소개글 둘 다 보고 평가
        """

        if mode == "recommendation":
            system_content = self.EVAL_SYSTEM_FALLBACK.format(
                persona_str=self.persona_str
            )
            user_content = f"시스템 추천 결과:\n{recommendation_text}"

        elif mode == "book_intro":
            if not book_intros:
                system_content = self.EVAL_SYSTEM_FALLBACK.format(
                    persona_str=self.persona_str
                )
                user_content = f"시스템 추천 결과:\n{recommendation_text}"
            else:
                book_intros_str = "\n\n".join([
                    f"📚 {title}\n소개: {intro}"
                    for title, intro in book_intros.items()
                ])
                system_content = self.EVAL_SYSTEM.format(
                    persona_str=self.persona_str,
                    book_intros_str=book_intros_str
                )
                user_content = "추천 도서 소개글을 바탕으로 평가해주세요."

        elif mode == "both":
            book_intros_str = "\n\n".join([
                f"📚 {title}\n소개: {intro}"
                for title, intro in (book_intros or {}).items()
            ])
            system_content = self.EVAL_SYSTEM_BOTH.format(
                persona_str=self.persona_str,
                book_intros_str=book_intros_str if book_intros else "소개글 없음"
            )
            user_content = (
                f"[추천 이유]\n{recommendation_text}\n\n"
                f"[도서 소개글]\n{book_intros_str if book_intros else '없음'}"
            )

        else:
            raise ValueError(f"지원하지 않는 mode: {mode}")

        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content
        raw = content.strip() if content else \
            '{"books_evaluated": [], "overall_reason": "응답 없음"}'

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"books_evaluated": [], "overall_reason": raw}

        books   = result.get("books_evaluated", [])
        matched = sum(1 for b in books if b.get("match"))

        if self.verbose:
            print(f"\n[Evaluation — mode: {mode}]")
            for b in books:
                mark = "O" if b.get("match") else "X"
                print(f"  [{mark}] {b.get('title', '?')} — {b.get('reason', '')}")
            print(f"  총평: {result.get('overall_reason', '')}")
            if books:
                print(f"  match_rate: {matched}/{len(books)}권 "
                      f"({matched / len(books):.0%})")

        return result

    def reset(self):
        self.history    = []
        self.turn_count = 0
