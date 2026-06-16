import json
import os
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

KST       = timezone(timedelta(hours=9))
LLM_MODEL = "gpt-4o-mini"

# ──────────────────────────────────────────────
# 페르소나 뱅크 (DNA 구조)
# ──────────────────────────────────────────────

PERSONA_BANK = {
    "A_최재원": {
        "reading_goal":     "이직을 준비하며 커리어 방향을 다시 잡는 데 도움이 될 통찰을 얻고 싶음",
        "preferred_genre":  "경제, 자기계발, 철학, 심리 (교양서)",
        "reading_style":    "핵심이 분명하고 실생활에 연결되는 글을 선호",
        "difficulty_level": "성인 일반 수준, 너무 두껍거나 학술적인 책은 부담",
        "current_context":  "평일 저녁 퇴근 후, 지치고 막막한 상태에서 방향을 찾고 싶음",
        "demographics":     "32세, 스타트업 마케터(5년차), 서울 성동구 거주, 1인 가구",
        "disliked":         "다들 읽는 뻔한 베스트셀러, 전형적인 자기계발서",
        "pain_points":      "추천 근거 없으면 손이 안 감 / 선택지 많으면 못 고름",
        "speaking_style":   "구어체 존댓말, 망설임 표현 자연스럽게('그냥...', '음...'), 1~3문장",
    },
    "B_한미영": {
        "reading_goal":     "두 자녀(초5, 중1)에게 맞는 책을 찾아주고 싶음",
        "preferred_genre":  "자녀용: 초5는 역사·과학, 중1은 AI·로봇",
        "reading_style":    "아이가 흥미를 잃지 않고 끝까지 읽을 수 있는 구성",
        "difficulty_level": "각 자녀 학년 수준에 정확히 맞는 난이도",
        "current_context":  "자녀 교육 목적, 아이 수준 판단이 어려워 도움이 필요함",
        "demographics":     "43세, 전업주부(전직 초등교사), 경기도 수원시 거주",
        "disliked":         "아이 수준에 안 맞는 너무 어렵거나 유치한 책",
        "pain_points":      "수준 판단 기준 없음 / 두 아이 관심사가 달라 따로 추천 원함",
        "speaking_style":   "차분하고 정중한 구어체, 자녀 이야기를 구체적으로 설명",
    },
    "C_오민아": {
        "reading_goal":     "인디 게임 개발에 입문하고 싶음 (코딩 거의 처음)",
        "preferred_genre":  "게임 개발·코딩 (이전에는 웹소설·콘텐츠 크리에이터 관심)",
        "reading_style":    "단계적으로 따라갈 수 있는 입문서, 자기 언어로 시작",
        "difficulty_level": "고등학생 입문 수준, 너무 전문적이면 부담",
        "current_context":  "관심사가 빠르게 바뀌는 시기, 최신 트렌드 도서를 원함",
        "demographics":     "17세, 고등학생, 인천시 연수구 거주",
        "disliked":         "옛날 책, 지나치게 전문적인 기술서",
        "pain_points":      "검색해도 옛날 책만 나옴 / 자기 수준에 맞는지 판단 어려움",
        "speaking_style":   "10대 구어체, 솔직하고 직설적, 짧은 문장",
    },
}

# 하위 호환: 기존 코드가 PERSONA_TEMPLATES를 참조하는 경우
PERSONA_TEMPLATES = PERSONA_BANK


# ──────────────────────────────────────────────
# UserSimAgent
# ──────────────────────────────────────────────

class UserSimAgent:
    """
    CRS 시스템의 프로파일링 질문에 DNA 페르소나 기반으로 자동 응답하는 에이전트.

    v1 대비 변경:
    - speaking_style을 시스템 프롬프트에 명시적으로 반영
    - current_context를 대화에 자연스럽게 녹이도록 지시
    """

    SYSTEM_PROMPT_TEMPLATE = """\
당신은 도서 추천 챗봇과 대화하는 실제 사용자를 시뮬레이션하는 에이전트입니다.

## 당신의 페르소나
{persona_str}

## 행동 규칙
1. 위 페르소나에 충실하게 답변하세요.
2. speaking_style에 명시된 말투와 표현 방식을 반드시 따르세요.
3. current_context(현재 상황)를 자연스럽게 대화에 녹여내세요.
4. 페르소나에 없는 정보는 페르소나와 일관된 방향으로 자연스럽게 만들어내세요.
5. 챗봇의 질문에만 답하세요. 책 추천을 먼저 요청하지 마세요.
"""

    def __init__(self, persona: dict, model: str = LLM_MODEL, verbose: bool = True):
        self.persona     = persona
        self.model       = model
        self.verbose     = verbose
        self.history     = []
        self.turn_count  = 0

        persona_str        = "\n".join(f"- {k}: {v}" for k, v in persona.items())
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(persona_str=persona_str)
        self.client        = OpenAI()

    def answer(self, question: str) -> str:
        self.turn_count += 1
        self.history.append({"role": "user", "content": question})

        if self.verbose:
            print(f"\n[Turn {self.turn_count}]")
            print(f"  CRS  : {question}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.history,
            ],
            temperature=0.7,
        )
        answer_text = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": answer_text})

        if self.verbose:
            print(f"  USER : {answer_text}")

        return answer_text

    def get_history(self) -> list:
        return self.history


# ──────────────────────────────────────────────
# PeekaJudge
# ──────────────────────────────────────────────

JUDGE_PROMPT = """\
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

## 출력 형식 (JSON만 출력)
{{
  "books_evaluated": [
    {{
      "title":  "책 제목",
      "match":  true,
      "reason": "판단 근거 한 문장"
    }}
  ]
}}
"""


class PeekaJudge:
    """
    페르소나 DNA와 book_intro 기반으로 추천 도서 적합성을 평가하는 Judge.

    입력: 페르소나 DNA + book_intros ({"제목 | 저자": "소개글"})
    출력: books_evaluated 목록 + book_match_rate
    """

    def __init__(self, model: str = LLM_MODEL, verbose: bool = True):
        self.model   = model
        self.verbose = verbose
        self.client  = OpenAI()

    def evaluate(self,
                 persona: dict,
                 book_intros: dict,
                 temperature: float = 0.0,
                 seed: int = 42) -> dict:
        if not book_intros:
            return {"books_evaluated": [], "book_match_rate": 0.0}

        persona_str     = "\n".join(f"- {k}: {v}" for k, v in persona.items())
        book_intros_str = "\n\n".join(
            f"📚 {title}\n소개: {intro}"
            for title, intro in book_intros.items()
        )

        prompt = JUDGE_PROMPT.format(
            persona_str=persona_str,
            book_intros_str=book_intros_str,
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "위 도서 소개글을 평가해주세요."},
            ],
            temperature=temperature,
            seed=seed,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content
        raw     = content.strip() if content else '{"books_evaluated": []}'

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"books_evaluated": []}

        books   = result.get("books_evaluated", [])
        matched = sum(1 for b in books if b.get("match"))

        result["book_match_rate"] = round(matched / len(books), 2) if books else 0.0
        result["model"]           = self.model
        result["evaluated_at"]    = datetime.now(tz=KST).isoformat()

        if self.verbose:
            print(f"\n[PeekaJudge / {self.model}]")
            for b in books:
                mark = "O" if b.get("match") else "X"
                print(f"  [{mark}] {b.get('title', '?')} — {b.get('reason', '')}")
            print(f"  match_rate: {matched}/{len(books)}권 ({result['book_match_rate']:.0%})")

        return result


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def extract_book_intros(session_result: dict) -> dict:
    """세션 결과에서 추천 도서의 book_intro를 추출한다."""
    book_intros = {}
    for rec in session_result.get("recommendations", []):
        title  = rec.get("title", "")
        author = rec.get("author", "")
        key    = f"{title} | {author}"
        if rec.get("book_intro"):
            book_intros[key] = rec["book_intro"]
    return book_intros


def judge_session(session_result: dict, persona: dict, verbose: bool = True) -> dict:
    """세션 결과에서 book_intros를 추출해 PeekaJudge 평가를 실행한다."""
    book_intros = extract_book_intros(session_result)

    if not book_intros:
        if verbose:
            print("  [Judge 스킵] book_intro 없음")
        return {}

    if verbose:
        print(f"\n  [book_intros 로드] {len(book_intros)}권")
        for title in book_intros:
            print(f"    📚 {title}")

    judge = PeekaJudge(verbose=verbose)
    return judge.evaluate(persona, book_intros)
