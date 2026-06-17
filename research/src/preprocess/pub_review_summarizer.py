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
OUTPUT_CSV     = _ROOT / "research/data/processed/books_with_summarized_review.csv"
CHECKPOINT_CSV = _ROOT / "research/data/processed/books_with_summarized_review_checkpoint.csv"

MAX_CONCURRENT  = 3     # 동시 요청 수
CHECKPOINT_EVERY = 500  # N건마다 중간 저장
TIMEOUT         = 30    # 요청 타임아웃(초)

API_KEY  = os.getenv("CLOVASTUDIO_API_KEY", "")
API_URL  = "https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001"

SYSTEM_PROMPT = (
    "당신은 도서 서평 요약 전문가입니다. "
    "아래 출판사 서평을 핵심 내용만 남겨 4~6문장으로 요약하세요. "
    "불필요한 마케팅 문구나 반복 내용은 제거하고, "
    "책의 주제·특징·대상 독자를 중심으로 간결하게 작성하세요."
)


# ── API 호출 ─────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS = 4000  # HCX-DASH-001 입력 한도 대비 안전 마진

async def summarize(client: httpx.AsyncClient, sem: asyncio.Semaphore, text: str) -> str:
    text = text[:MAX_INPUT_CHARS]  # 입력 길이 제한
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
                {"role": "user",   "content": text},
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
        return resp.json()["result"]["message"]["content"].strip()


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
        review = str(row["pub_review"]) if pd.notna(row["pub_review"]) else ""
        summary = await summarize(client, sem, review) if review else ""
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
