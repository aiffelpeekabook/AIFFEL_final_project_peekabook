"""
페르소나 JSON 파일들을 dict로 합쳐서 로드함.

인계 문서의 load_persona_bank() 패턴.
실제 페르소나 JSON은 backend/data/personas/*_sessions_manual.json 형태.

사용 예:
    from app.simulation.persona_loader import load_persona_bank
    bank = load_persona_bank("backend/data/personas/")
    # bank["A_최재원"], bank["B_한미영"], ...
"""

from __future__ import annotations

import json
from pathlib import Path


def load_persona_bank(persona_dir: str = "backend/data/personas") -> dict:
    """
    persona_dir의 모든 *_sessions_manual.json 파일을 읽어 하나의 dict로 합침.

    Args:
        persona_dir: 페르소나 JSON 파일들이 있는 디렉토리

    Returns:
        {"A_최재원": {...}, "B_한미영": {...}, ...} 형태의 통합 dict
    """
    bank: dict = {}
    pdir = Path(persona_dir)

    if not pdir.exists():
        raise FileNotFoundError(f"페르소나 디렉토리 없음: {persona_dir}")

    files = sorted(pdir.glob("*_sessions_manual.json"))
    if not files:
        raise FileNotFoundError(
            f"{persona_dir}에 *_sessions_manual.json 파일 없음"
        )

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
