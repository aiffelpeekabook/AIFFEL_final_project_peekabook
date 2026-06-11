import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PERSONA_TEMPLATES = {
    "직장인_SF팬": {
        "age_group": "30대 초반",
        "job": "IT 스타트업 개발자, 평일엔 바빠서 주말에 몰아 읽는 편",
        "favorite_genre": "SF, 테크 스릴러",
        "reading_frequency": "한 달에 1~2권",
        "mood": "머리를 비우고 싶어서 가볍게 읽히는 책을 원함",
        "disliked_genre": "로맨스, 자기계발",
        "reading_experience": "최근에 '프로젝트 헤일메리'를 읽고 재미있었음"
    },
    "대학생_문학팬": {
        "age_group": "20대 초반",
        "job": "대학교 국문학과 재학 중",
        "favorite_genre": "한국 현대소설, 시",
        "reading_frequency": "일주일에 1권 이상",
        "mood": "감성적이고 문장이 아름다운 책을 원함",
        "disliked_genre": "판타지, 무협",
        "reading_experience": "한강 작가의 채식주의자를 읽고 깊은 인상을 받음"
    },
    "중년_역사_비문학": {
        "age_group": "40대 중반",
        "job": "중학교 역사 교사",
        "favorite_genre": "역사, 교양 비문학",
        "reading_frequency": "한 달에 3~4권",
        "mood": "수업에 활용할 수 있는 흥미로운 역사 이야기",
        "disliked_genre": "공포, 오컬트",
        "reading_experience": "유발 하라리의 사피엔스를 인상 깊게 읽음"
    }
}


class UserSimAgent:
    """
    CSR 시스템의 프로파일링 질문에 페르소나 기반으로 자동 응답하는 에이전트.
    """

    SYSTEM_PROMPT_TEMPLATE = """\
당신은 도서 추천 챗봇과 대화하는 실제 사용자를 시뮬레이션하는 에이전트입니다.

## 당신의 페르소나
{persona_str}

## 행동 규칙
1. 위 페르소나에 충실하게 답변하세요.
2. 실제 사람처럼 자연스럽고 구어체로 답변하세요. (단답 가능)
3. 페르소나에 없는 정보는 페르소나와 일관된 방향으로 자연스럽게 만들어내세요.
4. 챗봇의 질문에만 답하세요. 책 추천을 먼저 요청하지 마세요.
5. 답변은 1~3문장 이내로 간결하게.
"""

    def __init__(self, persona: dict, model: str = "gpt-4o-mini", verbose: bool = True):
        self.persona = persona
        self.model = model
        self.verbose = verbose
        self.history = []
        self.turn_count = 0

        persona_str = "\n".join(f"- {k}: {v}" for k, v in persona.items())
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(persona_str=persona_str)
        self.client = OpenAI()

    def answer(self, question: str) -> str:
        self.turn_count += 1
        self.history.append({"role": "user", "content": question})

        if self.verbose:
            print(f"\n[Turn {self.turn_count}]")
            print(f"  CSR  : {question}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        answer_text = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": answer_text})

        if self.verbose:
            print(f"  USER : {answer_text}")

        return answer_text

    def get_history(self) -> list:
        return self.history
