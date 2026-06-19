"""
pub_review 요약 스크립트 (Clova Studio HCX-DASH-001)

- books_with_category.csv를 읽어 전체 레코드의 pub_review를 4~6문장으로 요약
- 비동기 처리 + 세마포어로 동시 요청 수 제한
- CHECKPOINT_EVERY 건마다 중간 저장 → 중단 후 재시작 가능

실행:
    cd /home/jjeong3150/work/peekabook
    python research/src/preprocess/pub_review_summarizer.py
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[3]  # peekabook/
load_dotenv(_ROOT / ".env")

# ── 설정 ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = _ROOT / "research/data/processed/books_with_category.csv"
OUTPUT_CSV     = _ROOT / "research/data/processed/books_with_summarized_review_2.csv"
CHECKPOINT_CSV = _ROOT / "research/data/processed/books_with_summarized_review_checkpoint_2.csv"

MAX_CONCURRENT  = 3     # 동시 요청 수
CHECKPOINT_EVERY = 500  # N건마다 중간 저장
TIMEOUT         = 30    # 요청 타임아웃(초)

API_KEY  = os.getenv("CLOVASTUDIO_API_KEY", "")
API_URL  = "https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001"

SYSTEM_PROMPT = (
    "당신은 도서 큐레이터입니다.\n"
    "아래 [도서 소개]와 [출판사 서평]을 읽고, 출판사 서평을 핵심 내용만 남겨 요약하세요.\n\n"
    "[규칙]\n"
    "- 실제 출판사 서평이나 도서 소개에 나올 법한 문체와 어휘를 사용하세요.\n"
    "- 저자명, 책 제목은 만들지 마세요. 내용과 주제만 묘사하세요.\n"
    "- 책의 주제·특징을 중심으로 4~6문장으로 작성하세요.\n"
    "- 어떤 독자에게 맞는 책인지(대상 독자층, 독서 목적)가 드러나도록 하세요.\n"
    "- 책의 난이도·분량 성격(가볍게 읽을 수 있는지, 학술적인지 등)을 한 문장 이내로 포함하세요.\n"
    "- 이 책만의 차별점이나 추천 근거가 드러나도록 하세요.\n"
    "- [출판사 서평]이 빈약하면 [도서 소개]의 내용을 보완적으로 활용하세요. 단, 두 원문 어디에도 없는 내용은 절대 만들어내지 마세요.\n"
    "- [출판사 서평]이 짧거나 내용이 부족하면 있는 내용만 요약하고, 문장 수를 억지로 채우지 마세요.\n"
    "- [출판사 서평]이 비어 있거나 의미 있는 정보가 없으면 빈 문자열만 반환하세요."
)


# ── API 호출 ─────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS = 4000  # HCX-DASH-001 입력 한도 대비 안전 마진
MIN_INPUT_CHARS = 50    # 이보다 짧으면 요약 불필요 (원문 그대로 사용)

def _build_user_message(book_intro: str, pub_review: str) -> str:
    intro_part  = book_intro[:1000] if book_intro else "(없음)"
    review_part = pub_review[:3000] if pub_review else "(없음)"
    return f"[도서 소개]\n{intro_part}\n\n[출판사 서평]\n{review_part}"


async def summarize(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                    book_intro: str, pub_review: str) -> str:
    async with sem:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(book_intro, pub_review)},
            ],
            "maxTokens":     300,
            "temperature":   0.3,
            "topK":          0,
            "topP":          0.8,
            "repeatPenalty": 5.0,
            "includeAiFilters": False,
        }
        resp = await client.post(API_URL, headers=headers, json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        text = resp.json()["result"]["message"]["content"].strip()
        # 모델이 자체적으로 붙이는 "출판사 서평 :", "요약 :" 등 prefix 제거
        for prefix in ("출판사 서평 :", "출판사서평:", "요약 :", "요약:"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        return text


# ── 메인 ─────────────────────────────────────────────────────────────────────
async def main():
    df = pd.read_csv(INPUT_CSV)
    rows = [row for _, row in df.iterrows()]
    print(f"전체: {len(rows):,}건 (전체 요약)")

    all_results = []
    sem    = asyncio.Semaphore(MAX_CONCURRENT)
    chunks = [rows[i:i + CHECKPOINT_EVERY] for i in range(0, len(rows), CHECKPOINT_EVERY)]
    pbar   = tqdm(total=len(rows), desc="진행")

    async def process(client, row):
        intro  = str(row["book_intro"])  if pd.notna(row["book_intro"])  else ""
        review = str(row["pub_review"])  if pd.notna(row["pub_review"])  else ""
        if len(review) >= MIN_INPUT_CHARS:
            summary = await summarize(client, sem, intro, review)
        elif len(intro) >= MIN_INPUT_CHARS:
            summary = await summarize(client, sem, intro, review)  # review는 짧거나 없음
        else:
            summary = "[서평 없음]"
        pbar.update(1)
        return {**row.to_dict(), "pub_review_summarized": summary}

    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            chunk_results = await asyncio.gather(*[process(client, row) for row in chunk])
            all_results.extend(chunk_results)
            pd.DataFrame(all_results).to_csv(CHECKPOINT_CSV, index=False)

    pbar.close()
    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n완료: {len(all_results):,}건 → {OUTPUT_CSV}")


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("CLOVASTUDIO_API_KEY가 .env에 없습니다.")
    asyncio.run(main())
