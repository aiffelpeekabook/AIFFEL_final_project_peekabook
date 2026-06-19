"""
PeekaJudge: 추천 도서 적합성을 평가하는 독립 LLM-as-Judge

기본 모델: Claude Haiku 4.5 (Anthropic)
멀티 프로바이더 지원: Anthropic / OpenAI / Gemini / ClovaStudio
모델 이름 접두사로 자동 감지 (claude-* / gemini-* / HCX-* / 그 외→openai)

주요 구성:
- PeekaJudge          : LLM-as-Judge 평가자
- judge_session       : 세션 결과에서 book_intros 추출해 Judge 실행
- _check_*            : 규칙 기반 불일치/중복 감지
- _determine_verdict  : 규칙(주) + Judge match_rate(보조) 조합 판정
- update_long_term_memory : 페르소나 long_term_memory를 in-place 업데이트
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

from anthropic import Anthropic

from app.config import JUDGE_MODEL, KST


# ── 프로바이더 감지 ────────────────────────────────────────────────────────────

def _detect_provider(model: str) -> str:
    """모델 이름 접두사로 API 프로바이더를 결정함.

    claude-*       → anthropic
    gemini-*       → gemini   (OpenAI-compat endpoint)
    HCX-* / clova* → clovastudio (OpenAI-compat endpoint)
    그 외           → openai
    """
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("HCX-") or model.lower().startswith("clova"):
        return "clovastudio"
    return "openai"


# ── 클라이언트 지연 초기화 ──────────────────────────────────────────────────────

_anthropic_client: Optional[Anthropic] = None
_openai_clients: dict = {}  # provider → openai.OpenAI instance


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def _get_openai_client(provider: str):
    """OpenAI-호환 클라이언트 (openai / gemini / clovastudio) 지연 초기화"""
    if provider in _openai_clients:
        return _openai_clients[provider]

    import openai

    if provider == "gemini":
        client = openai.OpenAI(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif provider == "clovastudio":
        client = openai.OpenAI(
            api_key=os.getenv("CLOVASTUDIO_API_KEY", ""),
            base_url="https://clovastudio.stream.ntruss.com/v1",
        )
    else:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    _openai_clients[provider] = client
    return client


# ──────────────────────────────────────────────
# Judge 프롬프트
# ──────────────────────────────────────────────

PEEKAJUDGE_RUBRIC_PROMPT = """\
당신은 도서 추천 시스템의 독립적인 평가자입니다.

아래 사용자 프로파일(DNA)과 도서 소개글을 보고
각 도서가 이 사용자에게 적합한 추천인지 판단하세요.

## 사용자 프로파일 (DNA)
{persona_str}

## 평가 대상 도서 소개글
{book_intros_str}

## 판단 루브릭 (5개 축을 순서대로 확인)
1. reading_goal     : 소개글의 내용이 DNA의 독서 목적에 부합하는가?
2. preferred_genre  : 소개글의 주제/장르가 DNA의 선호 장르와 맞는가?
3. reading_style    : 소개글의 구성 방식이 DNA의 독서 스타일에 맞는가?
4. difficulty_level : 소개글의 난이도가 DNA의 수준에 적합한가?
5. current_context  : 지금 이 사람의 현재 상황에 실제로 도움이 되는가?

## 절대 규칙
- 반드시 소개글에 명시된 내용만을 근거로 판단한다
- 책 제목·저자로 알고 있는 외부 지식, 리뷰, 배경지식은 사용 금지다

## 판단 순서 (반드시 이 순서대로)
① 루브릭 5개 축을 각각 만족하는지 판단한다
② 5개 중 3개 이상 충족 시 match=true, 미만이면 match=false
③ 최종 판단 근거를 한 문장으로 요약한다

## 출력 형식 (JSON만 출력, 다른 텍스트 절대 금지)
{{
  "books_evaluated": [
    {{
      "title": "책 제목",
      "match":  true,
      "reason": "판단 근거"
    }}
  ]
}}
"""

JUDGE_PROMPTS = {
    "peekajudge": PEEKAJUDGE_RUBRIC_PROMPT,
}


# ──────────────────────────────────────────────
# JSON 파싱 헬퍼 (Anthropic은 response_format 없음)
# ──────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Claude 응답에서 JSON 객체만 추출함.
    ```json ... ``` 래핑이나 앞뒤 설명문이 있어도 첫 { ... 마지막 } 까지를 잡음.
    """
    # 코드 펜스 제거
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", raw.strip(), re.DOTALL)
    if fence:
        raw = fence.group(1)

    # 첫 '{' 부터 마지막 '}' 까지를 시도
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else raw

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"books_evaluated": [], "overall_reason": raw[:200]}


# ──────────────────────────────────────────────
# PeekaJudge
# ──────────────────────────────────────────────

class PeekaJudge:
    """
    페르소나 DNA + book_intros로 추천 도서 적합성을 평가하는 LLM-as-Judge.

    PeekaReader(gpt-4o-mini)와 모델 패밀리를 분리(Claude)해 Preference Leakage 회피.
    """

    def __init__(self,
                 stage:   str  = "peekajudge",
                 model:   str  = JUDGE_MODEL,
                 verbose: bool = True):
        assert stage in JUDGE_PROMPTS, \
            f"stage는 {list(JUDGE_PROMPTS.keys())} 중 하나여야 함."
        self.stage   = stage
        self.model   = model
        self.verbose = verbose

    def evaluate(self,
                 persona:     dict,
                 book_intros: dict,
                 temperature: float = 0.0,
                 max_tokens:  int   = 2048) -> dict:
        """
        Args:
            persona:     extract_session_dna() 결과 dict
            book_intros: {"제목 | 저자": "소개글"} 형태
            temperature: 0.0이 default (재현성)
            max_tokens:  도서 3~5권 평가에 충분한 토큰 (default 2048)

        Returns:
            books_evaluated, book_match_rate, stage, model, evaluated_at 포함 dict
        """
        if not book_intros:
            return {"books_evaluated": [], "book_match_rate": 0.0}

        persona_str     = "\n".join(f"- {k}: {v}" for k, v in persona.items())
        book_intros_str = "\n\n".join([
            f"📚 {title}\n소개: {intro}"
            for title, intro in book_intros.items()
        ])

        prompt = JUDGE_PROMPTS[self.stage].format(
            persona_str=persona_str,
            book_intros_str=book_intros_str,
        )

        provider = _detect_provider(self.model)

        if provider == "anthropic":
            resp = _get_anthropic_client().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=prompt,
                messages=[{"role": "user", "content": "위 도서 소개글을 평가해주세요."}],
            )
            raw = resp.content[0].text.strip() if resp.content else ""
        else:
            resp = _get_openai_client(provider).chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system",  "content": prompt},
                    {"role": "user",    "content": "위 도서 소개글을 평가해주세요."},
                ],
            )
            raw = resp.choices[0].message.content or "" if resp.choices else ""
        result = _extract_json(raw)

        books = result.get("books_evaluated", [])

        # title에 저자가 " | "로 붙어 들어왔으면 분리
        for b in books:
            title = b.get("title", "")
            if " | " in title:
                b["title"] = title.split(" | ")[0].strip()

        matched = sum(1 for b in books if b.get("match"))
        result["book_match_rate"] = round(matched / len(books), 2) if books else 0.0
        result["stage"]           = self.stage
        result["model"]           = self.model
        result["evaluated_at"]    = datetime.now(tz=KST).isoformat()

        if self.verbose:
            print(f"\n[PeekaJudge — {self.stage} / {self.model}]")
            for b in books:
                mark = "O" if b.get("match") else "X"
                print(f"  [{mark}] {b.get('title', '?')} — {b.get('reason', '')}")
            if "overall_reason" in result:
                print(f"  총평: {result['overall_reason']}")
            print(f"  match_rate: {matched}/{len(books)}권 "
                  f"({result['book_match_rate']:.0%})")

        return result


def judge_session(session_result: dict,
                  persona:        dict,
                  *,
                  stage:          str  = "peekajudge",
                  model:          str  = JUDGE_MODEL,
                  verbose:        bool = True) -> dict:
    """
    run_session() 결과에서 book_intros를 추출해 PeekaJudge 실행함.

    Args:
        session_result: run_session 반환 dict (recommendations, retrieved_books 포함)
        persona:        extract_session_dna() 결과 (세션 DNA)
        stage:          Judge 프롬프트 종류 (현재 "peekajudge"만)
        model:          Judge 모델. 기본은 Claude Haiku 4.5. sweep으로 모델 비교 시 변경 가능
    """
    recommendations = session_result.get("recommendations", [])
    retrieved       = {
        b["isbn"]: b
        for b in session_result.get("retrieved_books", [])
        if b.get("isbn")
    }

    book_intros = {}
    for rec in recommendations:
        isbn   = rec.get("isbn", "")
        title  = rec.get("title", "")
        author = rec.get("author", "")
        if isinstance(author, list):
            author = ", ".join(author)
        key = f"{title} | {author}"
        if isbn in retrieved and retrieved[isbn].get("book_intro"):
            book_intros[key] = retrieved[isbn]["book_intro"]

    if not book_intros:
        print("  [Judge 스킵] book_intro 없음")
        return {}

    print(f"\n  [book_intros 로드 완료] {len(book_intros)}권")
    for title, intro in book_intros.items():
        print(f"    📚 {title}")
        print(f"       {intro[:60]}...")

    judge = PeekaJudge(stage=stage, model=model, verbose=verbose)
    return judge.evaluate(persona, book_intros)


# ──────────────────────────────────────────────
# 규칙 기반 불일치/중복 감지 (Judge 보상 해킹 차단용)
# ──────────────────────────────────────────────

# 난이도 불일치 감지용 키워드
HARD_KEYWORDS   = ["전공", "학술", "원론", "개론", "이론", "연구", "심층"]
SIMPLE_KEYWORDS = ["입문", "에세이", "쉽게", "처음", "기초", "청소년", "초등", "중학"]


def _check_difficulty_mismatch(session_dna:     dict,
                               recommendations: list,
                               retrieved:       dict) -> list:
    """difficulty_level 불일치 도서 목록 반환함."""
    preferred  = session_dna.get("difficulty_level", "")
    wants_easy = any(kw in preferred for kw in ["입문", "에세이", "부담", "쉬운", "가볍"])
    mismatch   = []

    for rec in recommendations:
        isbn  = rec.get("isbn", "")
        intro = retrieved.get(isbn, {}).get("book_intro", "")
        if wants_easy and any(kw in intro for kw in HARD_KEYWORDS):
            mismatch.append(rec.get("title", ""))

    return mismatch


def _check_genre_mismatch(session_dna:     dict,
                          recommendations: list,
                          retrieved:       dict) -> list:
    """preferred_genre 불일치 도서 목록 반환함."""
    preferred      = session_dna.get("preferred_genre", "")
    genre_keywords = [g.strip() for g in preferred.replace("(", "").replace(")", "").split(",")]
    mismatch       = []

    for rec in recommendations:
        isbn  = rec.get("isbn", "")
        intro = retrieved.get(isbn, {}).get("book_intro", "")
        if intro and not any(kw in intro for kw in genre_keywords):
            mismatch.append(rec.get("title", ""))

    return mismatch


def _check_duplicates(recommendations:        list,
                      previously_recommended: list) -> list:
    """이전 세션에서 이미 추천된 도서 목록 반환함."""
    prev_isbns = {b["isbn"] for b in previously_recommended if b.get("isbn")}
    return [
        rec.get("title", "")
        for rec in recommendations
        if rec.get("isbn") in prev_isbns
    ]


def _determine_verdict(rule_result: dict, judge_match_rate: float) -> str:
    """
    규칙 기반 결과(주) + Judge match_rate(보조)로 verdict 결정함.
    규칙이 명확한 불일치를 감지하면 Judge 결과 무시 → 보상 해킹 차단.
    """
    if len(rule_result["difficulty_mismatch"]) >= 2:
        return "too_hard"
    # genre_mismatch는 Judge도 낮을 때만 적용
    if len(rule_result["genre_mismatch"]) >= 2 and judge_match_rate < 0.67:
        return "genre_mismatch"
    if rule_result["duplicates"]:
        return "duplicate"
    # 규칙이 감지 못 한 경우 Judge 결과 보조 활용
    if judge_match_rate >= 0.67:
        return "satisfied"
    elif judge_match_rate >= 0.34:
        return "partial"
    else:
        return "unsatisfied"


def update_long_term_memory(full_persona:    dict,
                            session_id:      int,
                            session_dna:     dict,
                            session_result:  dict,
                            judge_result:    dict) -> None:
    """
    세션 종료 후 long_term_memory를 in-place 업데이트함.
    규칙 기반(주) + Judge match_rate(보조) 조합.

    Args:
        full_persona: PERSONA_BANK[persona_id] 전체 dict (long_term_memory 포함).
                      이 함수가 full_persona["long_term_memory"]를 직접 수정함.
        session_id, session_dna, session_result, judge_result: 세션 정보
    """
    memory          = full_persona["long_term_memory"]
    recommendations = session_result.get("recommendations", [])
    retrieved       = {
        b["isbn"]: b
        for b in session_result.get("retrieved_books", [])
        if b.get("isbn")
    }

    # 1단계: 규칙 기반 불일치 감지
    rule_result = {
        "difficulty_mismatch": _check_difficulty_mismatch(
            session_dna, recommendations, retrieved),
        "genre_mismatch":      _check_genre_mismatch(
            session_dna, recommendations, retrieved),
        "duplicates":          _check_duplicates(
            recommendations, memory["previously_recommended"]),
    }

    # 2단계: previously_recommended 업데이트
    for rec in recommendations:
        if rec.get("isbn") and rec.get("title"):
            memory["previously_recommended"].append({
                "isbn":  rec["isbn"],
                "title": rec["title"],
            })

    # 3단계: verdict 결정
    judge_match_rate = judge_result.get("book_match_rate", 0.0) \
                       if judge_result else 0.0
    verdict = _determine_verdict(rule_result, judge_match_rate)

    # 4단계: feedback_history 기록
    memory["feedback_history"].append({
        "session_id":          session_id,
        "verdict":             verdict,
        "difficulty_mismatch": rule_result["difficulty_mismatch"],
        "genre_mismatch":      rule_result["genre_mismatch"],
        "duplicates":          rule_result["duplicates"],
        "judge_match_rate":    judge_match_rate,
        "note":                judge_result.get("overall_reason", "") if judge_result else "",
    })

    # 5단계: derived_preferences 업데이트 (중복 방지)
    def add_pref(pref: str):
        if pref not in memory["derived_preferences"]:
            memory["derived_preferences"].append(pref)

    if verdict == "too_hard":
        add_pref(f"난이도 높은 책 거부 확인 (세션 {session_id})")
    elif verdict == "genre_mismatch":
        add_pref(f"장르 불일치 경험 (세션 {session_id})")
    elif verdict == "satisfied":
        genre = session_dna.get("preferred_genre", "")
        add_pref(f"세션 {session_id} 추천 만족 — {genre}")
    elif verdict == "partial":
        add_pref(f"세션 {session_id} 추천 일부 만족")

    print(f"  [memory 업데이트] verdict={verdict} | "
          f"derived_preferences: {len(memory['derived_preferences'])}개")
