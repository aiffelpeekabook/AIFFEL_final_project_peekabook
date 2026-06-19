"""
pub_review 요약 샘플 테스트 (5건)

길이 다양한 샘플로 프롬프트 결과 빠르게 확인용.

실행:
    cd /home/jjeong3150/work/peekabook
    python research/src/preprocess/pub_review_summarizer_sample.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import sys

import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from pub_review_summarizer import summarize, MIN_INPUT_CHARS  # noqa: E402

INPUT_CSV  = _ROOT / "research/data/processed/books_with_category.csv"
OUTPUT_CSV = _ROOT / "research/data/processed/pub_review_summarized_sample.csv"

# 길이 다양한 샘플 인덱스 (짧음<50 / 짧음50~300 / 중간×2 / 긴 2000+)
SAMPLE_INDICES = [2569, 0, 3, 4, 2]


async def main():
    import httpx

    df = pd.read_csv(INPUT_CSV)
    samples = df.iloc[SAMPLE_INDICES].copy()
    samples["review_len"] = samples["pub_review"].fillna("").str.len()

    results = []
    sem = asyncio.Semaphore(3)

    async with httpx.AsyncClient() as client:
        for _, row in samples.iterrows():
            intro  = str(row["book_intro"])  if pd.notna(row["book_intro"])  else ""
            review = str(row["pub_review"])  if pd.notna(row["pub_review"])  else ""

            if len(review) >= MIN_INPUT_CHARS:
                summary = await summarize(client, sem, intro, review)
            elif len(intro) >= MIN_INPUT_CHARS:
                summary = await summarize(client, sem, intro, review)
            else:
                summary = "[서평 없음]"

            results.append({
                "isbn":                 row["isbn"],
                "title":                row["title"],
                "review_len":           len(review),
                "book_intro":           intro,
                "pub_review_original":  review,
                "pub_review_summarized": summary,
            })

            print(f"\n{'='*60}")
            print(f"[{row['title']}]  (원문 {len(review)}자)")
            print(f"  원문: {review[:100]}...")
            print(f"  요약: {summary}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\n저장 완료 → {OUTPUT_CSV}")


if __name__ == "__main__":
    api_key = os.getenv("CLOVASTUDIO_API_KEY", "")
    if not api_key:
        raise ValueError("CLOVASTUDIO_API_KEY가 .env에 없습니다.")
    asyncio.run(main())
