"""
main.py — PeekaBook 데모 웹 백엔드.

HF Spaces에서 실행되는 FastAPI 서버.
프론트엔드(Vercel + Lovable)와 CORS로 통신하며,
LangGraph CRS를 한 턴씩 호출하는 얇은 래퍼 역할만 수행.

상태 관리:
- 백엔드는 stateless. 세션 간 누적은 클라이언트 localStorage가 담당.
- 같은 브라우저 세션 내 멀티턴은 LangGraph MemorySaver의 thread_id로 관리.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.graph_demo import (
    run_one_turn,
    healthcheck,
)


# ────────────────────────────────────────────────────────────────────────────
# FastAPI 앱
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PeekaBook Demo API",
    description="LangGraph 기반 대화형 도서 추천 데모 백엔드",
    version="0.1.0",
)

# CORS — Vercel 배포 도메인이 확정되면 origins를 좁히는 것을 권장
# 데모 단계에서는 와일드카드로 시작해도 무방
_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "*",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# 스키마
# ────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., description="유저가 입력한 메시지")
    session_id: Optional[str] = Field(
        None, description="현재 세션 ID. 없으면 서버가 새로 발급."
    )
    thread_id: Optional[str] = Field(
        None,
        description=(
            "LangGraph thread ID. 같은 세션 내 멀티턴에서는 첫 응답의 thread_id를 "
            "그대로 다음 요청에 포함시켜야 함."
        ),
    )
    prior_profile_payload: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "이전 세션에서 누적된 프로파일 (localStorage에 저장돼 있던 것). "
            "새 사용자거나 데이터를 비우고 싶으면 None."
        ),
    )


class ChatResponse(BaseModel):
    ai_response: str
    recommendations: list[Any] = Field(default_factory=list)
    phase: str = ""
    profile_payload: dict[str, Any] = Field(default_factory=dict)
    session_id: str
    thread_id: str
    session_done: bool = False


# ────────────────────────────────────────────────────────────────────────────
# 엔드포인트
# ────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """루트 — 사람이 URL을 직접 열었을 때 보일 안내."""
    return {
        "service": "PeekaBook Demo API",
        "status": "running",
        "endpoints": ["/chat", "/health"],
    }


@app.get("/health")
async def health():
    """GitHub Actions가 6시간마다 핑할 엔드포인트. HF Spaces 슬립 방지용. 시간은 UTC 기준이므로 주의.

    무거운 로직 없이 그래프가 컴파일됐는지만 가볍게 확인.
    """
    return {
        "status": "ok",
        **healthcheck(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """한 턴의 대화를 처리한다.

    같은 세션의 후속 요청에서는 응답에 받은 thread_id를 그대로 다시 보내고,
    세션이 끝나면(session_done=True) profile_payload를 localStorage에 저장한다.
    다음 세션 시작 시 그 payload를 prior_profile_payload로 보내면
    이전 세션의 프로파일이 누적된 상태로 대화가 시작된다.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = await run_one_turn(
            user_message=req.message,
            session_id=session_id,
            prior_profile_payload=req.prior_profile_payload,
            thread_id=req.thread_id,
        )
    except Exception as e:
        # 데모 단계에서는 디버깅을 위해 에러 메시지를 그대로 노출.
        # 운영 단계로 가면 로깅으로 빼고 generic 메시지로 교체할 것.
        raise HTTPException(status_code=500, detail=f"graph error: {type(e).__name__}: {e}")

    return ChatResponse(
        ai_response=result["ai_response"],
        recommendations=result["recommendations"],
        phase=result["phase"],
        profile_payload=result["profile_payload"],
        session_id=session_id,
        thread_id=result["thread_id"],
        session_done=result["session_done"],
    )
