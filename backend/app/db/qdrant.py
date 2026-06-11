# src/db/qdrant.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from dotenv import load_dotenv
import os
import uuid


class QdrantDB:
    def __init__(self, vector_size=1024, default_limit=10, default_threshold=0.7, timeout=300):
        """
        Parameters
        ----------
        vector_size        : 임베딩 모델 차원 수 (기본 bge-m3 = 1024)
        default_limit      : 기본 검색 결과 수
        default_threshold  : 기본 유사도 임계값 (0~1, 높을수록 엄격)
        timeout            : Qdrant 요청 타임아웃(초), 기본 300초
        """
        load_dotenv()
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=timeout
        )
        self.vector_size = vector_size
        self.default_limit = default_limit
        self.default_threshold = default_threshold

    # ─────────────────────────────────────────
    # 컬렉션 관련
    # ─────────────────────────────────────────

    def create_collection(self, collection_name, vector_size=None):
        """
        컬렉션 생성
        - vector_size를 따로 지정하면 컬렉션마다 다른 차원 수 사용 가능
        - 예) db.create_collection("books_en", vector_size=1536)
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size or self.vector_size,
                distance=Distance.COSINE    # 벡터 방향이 얼마나 비슷한지(그 외 EUCLID, DOT 방식 있음)
            )
        )
        print(f"✅ 컬렉션 생성 완료: {collection_name}")

    def delete_collection(self, collection_name):
        """컬렉션 삭제"""
        self.client.delete_collection(collection_name)
        print(f"🗑️ 컬렉션 삭제 완료: {collection_name}")

    def get_collections(self):
        """컬렉션 목록 조회"""
        return self.client.get_collections()

    def create_payload_index(self, collection_name, field_name, field_schema):
        """
        필터 검색에 사용할 payload 인덱스 생성

        Parameters
        ----------
        field_name   : 인덱스를 생성할 필드명
        field_schema : 필드 타입 ("keyword", "integer", "float", "bool", "text")
            예)
            db.create_payload_index("books", "genre", "keyword")
            db.create_payload_index("books", "publish_year", "integer")
        """
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"✅ 인덱스 생성 완료: {field_name} ({field_schema})")

    # ─────────────────────────────────────────
    # 데이터 적재
    # ─────────────────────────────────────────

    def insert(self, collection_name, points, id_field=None):
        """
        데이터 적재 (upsert 방식 - 있으면 업데이트, 없으면 삽입)

        Parameters
        ----------
        points   : PointStruct 리스트
        id_field : payload에서 UUID 생성에 사용할 필드명 (예: "isbn")
                   지정하면 해당 필드값으로 결정적 UUID를 자동 생성하므로
                   같은 값을 다시 넣으면 upsert가 update로 동작함
            예)
            db.insert("books", points, id_field="isbn")

            points = [
                PointStruct(
                    id=None,  # id_field 지정 시 자동 생성되므로 생략 가능
                    vector=[0.12, 0.34, ...],
                    payload={
                        "title": "채식주의자",
                        "author": "한강",
                        "genre": "소설",
                        "isbn": "9788936434267",
                        "publish_year": 2007,
                        "loan_count": 348,
                        "is_available": True,
                        "description": "줄거리..."
                    }
                )
            ]
        """
        if id_field is not None:
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(p.payload[id_field]))),
                    vector=p.vector,
                    payload=p.payload
                )
                for p in points
            ]
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"✅ 데이터 적재 완료: {len(points)}건 → {collection_name}")

    def insert_batch(self, collection_name, points, id_field=None, batch_size=200):
        """
        배치 단위로 나눠 적재 (대용량 / 타임아웃 방지용)

        Parameters
        ----------
        batch_size : 한 번에 upsert할 포인트 수 (기본 200)
        """
        from tqdm import tqdm

        if id_field is not None:
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(p.payload[id_field]))),
                    vector=p.vector,
                    payload=p.payload
                )
                for p in points
            ]

        for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading → {collection_name}"):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i + batch_size]
            )

        print(f"✅ 데이터 적재 완료: {len(points)}건 → {collection_name}")

    # ─────────────────────────────────────────
    # 검색
    # ─────────────────────────────────────────

    def search(self, collection_name, query_vector, limit=None, threshold=None):
        """
        벡터 검색 (필터 없음)

        Parameters
        ----------
        query_vector : 쿼리 임베딩 벡터
        limit        : 반환할 최대 결과 수 (None이면 default_limit 사용)
        threshold    : 유사도 임계값 (None이면 default_threshold 사용)

        Returns
        -------
        list of ScoredPoint
            r.payload  : 메타데이터 (title, author 등)
            r.score    : 유사도 점수 (0~1)
        """
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit if limit is not None else self.default_limit,
            score_threshold=threshold if threshold is not None else self.default_threshold
        ).points

    def search_with_filter(self, collection_name, query_vector, query_filter, limit=None, threshold=None):
        """
        메타데이터 필터 검색

        Parameters
        ----------
        query_filter : qdrant_client.models.Filter
            예)
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

            # 장르 일치
            Filter(must=[FieldCondition(key="genre", match=MatchValue(value="문학 > 한국문학 > 소설"))])

            # 2020년 이후
            Filter(must=[FieldCondition(key="publish_year", range=Range(gte=2020))])

            # 장르 + 연도 복합
            Filter(must=[
                FieldCondition(key="genre", match=MatchValue(value="문학 > 한국문학 > 소설")),
                FieldCondition(key="publish_year", range=Range(gte=2020)),
            ])

        Returns
        -------
        list of ScoredPoint
        """
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit if limit is not None else self.default_limit,
            score_threshold=threshold if threshold is not None else self.default_threshold
        ).points