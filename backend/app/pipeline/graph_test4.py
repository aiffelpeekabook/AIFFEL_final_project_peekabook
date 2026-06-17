"""
graph_test3 기반 (v4).

test3 대비 변경:
- extract_genre_node → extract_genre_node_v2 (대분류 2개, 중분류 3개 필수 선택)
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from app.profiling.profiler import (
    MemoryStore,
    create_nodes,
    route_after_slot_processing,
    route_after_similar_search,
    route_after_match_confirm,
    route_after_book_experience,
)
from app.state.state_v3 import GraphState, Phase, UserProfile

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

initial_state = {
    "messages": [],
    "session_id": "",
    "phase": Phase.SLOT_FILLING,
    "turn_count": 0,
    "user_profile": UserProfile(),
    "current_slot": "reading_goal",
    "similar_profiles": None,
    "matched_profile_id": None,
    "book_experiences": [],
    "asked_book_experience": False,
    "summary": "",
    "reflection": "",
    "links": [],
    "ai_response": "",
    "retrieved_books": [],
    "recommendations": [],
    "genre_filter": [],
    "genre_level": "none",
    "availability_results": None,
}


def create_app(chroma_db_path: str,
               use_genre_filter: bool = True,
               rag_module=None):
    if rag_module is None:
        import app.rag.query_transform_v4 as rag_module

    memory_store = MemoryStore(persist_directory=chroma_db_path)
    nodes        = create_nodes(llm, memory_store)

    graph = StateGraph(GraphState)

    # 프로파일링 노드
    graph.add_node("generate_slot_question",  nodes["generate_slot_question"])
    graph.add_node("process_slot_answer",     nodes["process_slot_answer"])
    graph.add_node("search_similar_profiles", nodes["search_similar_profiles"])
    graph.add_node("process_match_confirm",   nodes["process_match_confirm"])
    graph.add_node("ask_book_experience",     nodes["ask_book_experience"])
    graph.add_node("process_book_experience", nodes["process_book_experience"])
    graph.add_node("generate_summary",        nodes["generate_summary"])
    graph.add_node("perform_reflection",      nodes["perform_reflection"])

    # RAG 노드
    graph.add_node("query_transform_rag", rag_module.query_transform_rag_node)

    # 프로파일링 엣지
    graph.set_entry_point("generate_slot_question")
    graph.add_edge("generate_slot_question", "process_slot_answer")
    graph.add_conditional_edges("process_slot_answer", route_after_slot_processing, {
        "generate_slot_question": "generate_slot_question",
        "search_similar_profiles": "search_similar_profiles",
        "ask_book_experience": "ask_book_experience",
    })
    graph.add_conditional_edges("search_similar_profiles", route_after_similar_search, {
        "process_match_confirm": "process_match_confirm",
        "generate_slot_question": "generate_slot_question",
    })
    graph.add_conditional_edges("process_match_confirm", route_after_match_confirm, {
        "ask_book_experience": "ask_book_experience",
        "generate_slot_question": "generate_slot_question",
    })
    graph.add_edge("ask_book_experience", "process_book_experience")
    graph.add_conditional_edges("process_book_experience", route_after_book_experience, {
        "ask_book_experience": "ask_book_experience",
        "generate_summary": "generate_summary",
    })
    graph.add_edge("generate_summary", "perform_reflection")

    # profiling → (장르 필터) → RAG → END
    if use_genre_filter:
        graph.add_node("extract_genre", rag_module.extract_genre_node_v2)
        graph.add_edge("perform_reflection",  "extract_genre")
        graph.add_edge("extract_genre",       "query_transform_rag")
    else:
        graph.add_edge("perform_reflection", "query_transform_rag")

    graph.add_edge("query_transform_rag", END)

    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=[
            "process_slot_answer",
            "process_match_confirm",
            "process_book_experience",
        ],
    )
