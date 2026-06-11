import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. 연결
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 2. 컬렉션 생성
client.create_collection(
    collection_name="books",
    vectors_config=VectorParams(
        size=1024,        # bge-m3 차원 수
        distance=Distance.COSINE
    )
)

# 3. 샘플 도서 데이터
sample_books = [
    {
        "id": 1,
        "title": "채식주의자",
        "author": "한강",
        "genre": "소설",
        "description": "한 여성이 채식주의자가 되면서 벌어지는 이야기"
    },
    {
        "id": 2,
        "title": "82년생 김지영",
        "author": "조남주",
        "genre": "소설",
        "description": "평범한 한국 여성의 삶을 그린 이야기"
    },
    {
        "id": 3,
        "title": "코스모스",
        "author": "칼 세이건",
        "genre": "과학",
        "description": "우주의 기원과 인류의 역사를 탐구하는 책"
    }
]

# 4. 임베딩 모델 로드
model = SentenceTransformer("BAAI/bge-m3")

# 5. 임베딩 생성 & 삽입
points = []
for book in sample_books:
    embedding = model.encode(book["description"]).tolist()
    points.append(
        PointStruct(
            id=book["id"],
            vector=embedding,
            payload={
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "description": book["description"]
            }
        )
    )

client.upsert(
    collection_name="books",
    points=points
)
print("✅ 데이터 삽입 완료!")

# 6. 검색 테스트
query = "여성의 삶을 다룬 소설"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="books",
    query=query_vector,
    limit=2
).points

for r in results:
    print(f"📚 {r.payload['title']} - {r.payload['author']} (score: {r.score:.3f})")