import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LocalReranker:
    """
    BAAI/bge-reranker-v2-m3 기반 로컬 리랭커.
    transformers 직접 방식으로 구현 (FlagReranker 호환성 문제 회피).

    Usage:
        reranker = LocalReranker("BAAI/bge-reranker-v2-m3")
        ranked_books = reranker.rerank(query=summary, books=merged_payloads, top_n=10)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def compute_scores(self, pairs: list) -> list:
        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)

        scores = torch.sigmoid(logits).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def rerank(self, query: str, books: list, top_n: int = 10) -> list:
        """
        query 기준으로 후보 도서를 재정렬.

        Args:
            query  : 리랭킹 기준 쿼리 (원본 summary 권장)
            books  : payload 딕셔너리 리스트 {"title", "book_intro", "isbn", ...}
            top_n  : 반환할 상위 도서 수
        """
        if not books:
            return []

        pairs = [
            [query, b.get("book_intro") or b.get("title", "")]
            for b in books
        ]
        scores = self.compute_scores(pairs)

        ranked = sorted(zip(scores, books), key=lambda x: x[0], reverse=True)

        print("\n[Reranking 결과]")
        for score, b in ranked[:top_n]:
            print(f"  score: {score:.4f} | {b.get('title')}")

        return [b for _, b in ranked[:top_n]]
