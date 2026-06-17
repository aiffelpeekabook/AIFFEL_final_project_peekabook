"""
페르소나 JSON 파일들을 dict로 합쳐서 로드함.

실제 최종 평가용 페르소나 JSON은 backend/data/personas/ 아래 *.json 형태.
하위 폴더(dummy/)는 자동으로 제외됨.

사용 예:
    from app.simulation.persona_loader import load_persona_bank
    bank = load_persona_bank("backend/data/personas/")
    # bank["A_최재원"], bank["B_한미영"], ...
"""

from __future__ import annotations

import json
from pathlib import Path


def load_persona_bank(persona_dir: str = "backend/data/personas") -> dict:
    bank: dict = {}
    pdir = Path(persona_dir)

    if not pdir.exists():
        raise FileNotFoundError(f"페르소나 디렉토리 없음: {persona_dir}")

    # 변경: 모든 .json 파일 로드 (더미와 정식 페르소나 모두 지원)
    files = sorted(pdir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{persona_dir}에 .json 파일 없음")

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            bank.update(data)

    return bank


def load_dummy_bank(dummy_dir: str = "backend/data/personas/dummy") -> dict:
    """테스트용 더미 페르소나 풀 로드. dummy_personas.json 단일 파일."""
    pdir = Path(dummy_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"더미 디렉토리 없음: {dummy_dir}")

    bank: dict = {}
    for path in sorted(pdir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            bank.update(json.load(f))
    return bank
