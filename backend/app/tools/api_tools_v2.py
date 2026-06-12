import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.state.state import GraphState
from app.tools.tools import (
    search_library,
    check_book_availability,
)

load_dotenv()

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

tools = [
    search_library,
    check_book_availability,
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """당신은 도서관 책 추천 큐레이터입니다.

[출력 형식 - 반드시 지킬 것]
추천 도서마다 아래 형식으로 작성하세요.

---
📚 {책 제목} | {저자}
![책 제목](표지URL)

✏️ 추천 이유
(전달받은 추천 이유를 그대로 작성)

📍 대출 가능 여부
- (도구 호출로 얻은 실제 도서관 이름): (대출 가능) or (대출 불가) or (미소장)
---

[주의사항]
- 추천 도서 목록의 3권을 반드시 모두 위 형식으로 출력하세요. 예외는 없습니다.
- 대출 가능 여부와 관계없이 3권 전부 출력하세요.
- 표지 이미지는 [추천 도서] 목록에 제공된 cover_url을 그대로 사용하세요. 도구로 가져오지 않습니다.
- cover_url이 비어 있으면 이미지 라인은 생략하세요.
- 항목에는 search_library 도구로 조회한 실제 도서관 이름을 그대로 사용하세요.
- 도구 호출에 실패하거나 정보를 확인할 수 없는 경우 "확인 불가"로 표시하고 출력 형식은 반드시 유지하세요.
- 책을 제거하거나 다른 책으로 대체하는 것은 절대 금지입니다.
- 대출 불가를 이유로 추가 멘트, 대안 제시, 도서관 방문 권유 등 불필요한 내용을 덧붙이지 마세요.
- 대출 정보는 반드시 도구 호출 결과만 사용하세요. 절대 지어내지 마세요.
"""

agent_executor = create_react_agent(llm, tools, prompt=system_prompt)


def api_tool_calling_node(state: GraphState) -> dict:
    recommendations = state.get("recommendations", [])
    summary = state.get("summary", "")
    district = "강남구"

    if not recommendations:
        msg = "검색된 도서가 없어 추천을 제공할 수 없습니다."
        return {"messages": [AIMessage(content=msg)]}

    if isinstance(recommendations, str):
        rec_text = recommendations
    else:
        rec_text = "\n".join([
            f"- 제목: {r['title']}, 저자: {r['author']}, ISBN: {r['isbn']}, cover_url: {r.get('cover_url', '')}, 추천 이유: {r['reason']}"
            for r in recommendations
        ])

    query = f"""
아래 추천 도서 3권의 {district} 도서관 대출 가능 여부를 확인해서 최종 추천 답변을 만들어줘.
표지 이미지는 각 도서의 cover_url을 그대로 사용해.

[추천 도서]
{rec_text}

[사용자 프로파일]
{summary}
"""

    result = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    return {"messages": [result["messages"][-1]]}
