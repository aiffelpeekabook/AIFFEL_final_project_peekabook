# PeekaBook

멀티-에이전트 기반 대화형 도서 추천 시스템

## 개요

공공도서관의 키워드 검색 방식은 목적이 명확하지 않거나 필요 자체가 언어화되지 않은 이용자에게는 출발점조차 제공하지 못한다. 또한, 전통적인 추천 시스템은 과거 행동 데이터에 의존하기 때문에 신규 사용자(Cold-Start), 실시간 선호 변화 반영에 한계가 있다.

PeekaBook은 이러한 한계를 극복하기 위한 **대화형 도서 추천 시스템(CRS)** 이다. 자연어 대화를 통해 사용자의 독서 목적·상황·선호를 점진적으로 수집하고, RAG 기반 검색과 도서관 대출 정보 조회를 결합하여 신뢰도 높고 실용적인 도서 큐레이션을 제공한다.

### 주요 기능

- **사용자 프로파일링**: LLM이 맥락에 맞는 질문을 생성하고 5가지 슬롯(독서 목적, 선호 장르 등) + 도서 경험을 수집
- **Reflection**: 유사 프로파일 검색을 통해 단일 세션에서 드러나지 않은 잠재적 선호를 패턴화
- **RAG 기반 추천**: ClaBi 수집 10만 건 도서 서지 데이터 VectorDB, Query Transformation + Weighted RRF + Reranking
- **도서관 연동**: 추천 도서의 주변 도서관 대출 가능 여부를 함께 제공
- **사용자 시뮬레이션**: 페르소나 기반 합성 사용자 에이전트로 CRS 평가

---

## 디렉토리 구조

```
peekabook/
├── .env                          # API 키 및 설정값 (직접 채워야 함)
├── .gitignore
├── backend/
│   ├── run_simulation.py         # CRS × UserSim 오케스트레이션 진입점
│   ├── requirements.txt
│   └── app/
│       ├── config.py             # 환경변수 중앙 관리
│       ├── state/
│       │   └── state.py          # GraphState, UserProfile, Phase, SlotStatus 등 타입 정의
│       ├── db/
│       │   └── qdrant.py         # QdrantDB 클래스 (insert / search / filter)
│       ├── embedding/
│       │   └── embedder.py       # LocalEmbedder (BAAI/bge-m3), APIEmbedder
│       ├── reranking/
│       │   └── reranker.py       # LocalReranker (BAAI/bge-reranker-v2-m3)
│       ├── tools/
│       │   ├── tools.py          # LangChain @tool: 정보나루·네이버 API 6종
│       │   └── api_tools.py      # ReAct agent executor, api_tool_calling_node
│       ├── profiling/
│       │   └── profiler.py       # 슬롯 질문 생성·응답 처리·유사 프로파일 검색·Reflection 노드
│       ├── rag/
│       │   └── query_transform.py # Step-back / Rewrite / Decompose → RRF → Reranking 노드
│       ├── pipeline/
│       │   └── graph.py          # LangGraph 그래프 조립 및 컴파일 (app, initial_state, config export)
│       └── simulation/
│           └── user_sim.py       # UserSimAgent, PERSONA_TEMPLATES (3종 페르소나)
├── research/                     # Colab 실험용 원본 노트북 (참고용)
│   └── src/
│       ├── main.ipynb
│       ├── state/, db/, embedding/, reranking/
│       ├── tools/, profiling/, rag/, simulation/
│       └── data/
├── frontend/                     # (미구현, Next.js 예정)
├── infra/
├── scripts/
└── docs/
```

---

## 그래프 흐름

```
[슬롯 질문 생성] → [슬롯 응답 처리] → [유사 프로파일 검색]
                                      ↓
                              [매칭 확인] → [도서 경험 수집]
                                                ↓
                                         [요약 생성] → [Reflection]
                                                            ↓
                                                    [장르 추출] → [Query Transform RAG]
                                                                        ↓
                                                                  [RAG LLM] → [API Tool Calling] → END
```

`interrupt_before`: `process_slot_answer`, `process_match_confirm`, `process_book_experience` (사용자 입력 대기 지점)

---

## 설치 및 실행

### 1. 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

### 2. 환경변수 설정

루트에 `.env` 파일을 생성하고 아래 값을 채운다.

```env
# ── LLM ─────────────────────────────────────────────
OPENAI_API_KEY=          # 필수 | platform.openai.com

# ── Qdrant Vector DB ─────────────────────────────────
QDRANT_URL=              # 필수 | Qdrant Cloud 클러스터 URL
QDRANT_API_KEY=          # 필수 | Qdrant Cloud API 키
QDRANT_COLLECTION_NAME=  # 필수 | 사용할 컬렉션 이름 (예: books_intro_48k)

# ── 정보나루 Open API (도서관 검색 · 대출 조회) ──────
# https://www.data4library.kr 에서 발급
LIBRARY_API_KEY=         # 필수

# ── 네이버 검색 API (도서 ISBN · 표지 이미지) ────────
# https://developers.naver.com 에서 애플리케이션 등록 후 발급
NAVER_CLIENT_ID=
NAVER_CLIENT_SECRET=

# ── 알라딘 API (선택) ────────────────────────────────
ALADIN_API_KEY=
ALADIN_API_KEYS=         # 복수 키 사용 시 쉼표로 구분

# ── 앱 설정 (기본값 그대로 사용 가능) ────────────────
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
CHROMA_DB_PATH=./chroma_db
```

### 3. 시뮬레이션 실행

```bash
cd backend
python run_simulation.py
```

페르소나(`직장인_SF팬` / `대학생_문학팬` / `중년_역사_비문학`)를 `run_simulation.py` 내 `main()` 함수에서 선택하여 실행한다.
