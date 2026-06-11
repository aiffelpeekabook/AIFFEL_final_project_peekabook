# src/preprocess/sample_collector.py

import requests
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

LOAN_ITEM_SRCH_URL = "https://data4library.kr/api/loanItemSrch"
ALADIN_SEARCH_URL = "http://www.aladin.co.kr/ttb/api/ItemSearch.aspx"
ALADIN_NS = "http://www.aladin.co.kr/ttb/apiguide.aspx"


class SampleCollector:
    """
    도서관정보나루 API를 통해 인기 대출 도서 샘플 데이터를 수집하고 저장하는 클래스.

    사용 예)
    collector = SampleCollector()
    books = collector.collect(total=1000)
    collector.save(books)
    """

    def __init__(self):
        self.auth_key = os.getenv("LIBRARY_API_KEY")
        if not self.auth_key:
            raise ValueError("LIBRARY_API_KEY가 .env에 설정되지 않았습니다.")

    def _fetch_page(self, page_no, page_size, start_dt, end_dt):
        params = {
            "authKey": self.auth_key,
            "startDt": start_dt,
            "endDt": end_dt,
            "pageNo": page_no,
            "pageSize": page_size,
            "format": "json",
        }
        response = requests.get(LOAN_ITEM_SRCH_URL, params=params)
        response.raise_for_status()
        return response.json()

    def collect(self, total=100, start_dt=None, end_dt=None):
        """
        인기 대출 도서 수집

        Parameters
        ----------
        total    : 수집할 총 도서 수
        start_dt : 조회 시작일 (YYYY-MM-DD, 기본: 30일 전)
        end_dt   : 조회 종료일 (YYYY-MM-DD, 기본: 오늘)

        Returns
        -------
        list of dict
        """
        if end_dt is None:
            end_dt = datetime.today().strftime("%Y-%m-%d")
        if start_dt is None:
            start_dt = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

        books = []
        page_size = min(total, 100)
        page_no = 1

        while len(books) < total:
            data = self._fetch_page(page_no, page_size, start_dt, end_dt)
            docs = data.get("response", {}).get("docs", [])

            if not docs:
                break

            for item in docs:
                doc = item.get("doc", {})
                books.append({
                    "title": doc.get("bookname", ""),
                    "author": doc.get("authors", ""),
                    "publisher": doc.get("publisher", ""),
                    "publish_year": doc.get("publication_year", ""),
                    "isbn": doc.get("isbn13", ""),
                    "genre": doc.get("class_nm", ""),
                    "loan_count": int(doc.get("loan_count", 0)),
                    "image_url": doc.get("bookImageURL", ""),
                })

            if len(docs) < page_size:
                break

            page_no += 1

        print(f"✅ {len(books[:total])}건 수집 완료 ({start_dt} ~ {end_dt})")
        return books[:total]

    def save(self, books, path="data/raw/sample_books.json"):
        """
        수집한 데이터를 JSON 파일로 저장

        Parameters
        ----------
        books : collect() 또는 AladinEnricher.enrich()의 반환값
        path  : 저장 경로
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(books, f, ensure_ascii=False, indent=2)
        print(f"✅ 저장 완료 → {path}")


class AladinEnricher:
    """
    알라딘 API로 제목 검색 후 저자 매칭을 통해 description을 보강하는 클래스.

    사용 예)
    enricher = AladinEnricher()
    enriched_books = enricher.enrich(books)
    """

    def __init__(self, delay=0.5):
        self.api_key = os.getenv("ALADIN_API_KEY")
        if not self.api_key:
            raise ValueError("ALADIN_API_KEY가 .env에 설정되지 않았습니다.")
        self.delay = delay

    def _clean_title(self, title):
        return re.sub(r"\s*:\s*", " ", title).strip()

    def _extract_author_name(self, author_str):
        if ":" in author_str:
            name_part = author_str.split(":")[1]
        else:
            name_part = author_str
        return re.split(r"[,;]", name_part)[0].strip()

    def _search(self, title):
        params = {
            "ttbkey": self.api_key,
            "Query": self._clean_title(title),
            "QueryType": "Title",
            "MaxResults": 10,
            "start": 1,
            "SearchTarget": "Book",
            "output": "xml",
            "Version": "20131101",
        }
        response = requests.get(ALADIN_SEARCH_URL, params=params)
        response.raise_for_status()
        return response.text

    def _find_description(self, xml_text, author):
        author_name = self._extract_author_name(author)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""

        ns = {"a": ALADIN_NS}
        items = root.findall("a:item", ns)
        first_description = ""

        for item in items:
            desc = item.findtext("a:description", "", ns)
            if not first_description:
                first_description = desc
            item_author = item.findtext("a:author", "", ns)
            if author_name in item_author:
                return desc

        return first_description

    def enrich(self, books):
        """
        books 리스트에 description 필드 추가

        Parameters
        ----------
        books : SampleCollector.collect()의 반환값

        Returns
        -------
        description 필드가 추가된 books 리스트
        """
        enriched = []
        total = len(books)

        for i, book in enumerate(books, 1):
            try:
                xml_text = self._search(book["title"])
                description = self._find_description(xml_text, book["author"])
            except Exception as e:
                print(f"  ⚠️ [{i}/{total}] {book['title'][:20]} 실패: {e}")
                description = ""

            enriched.append({**book, "description": description})

            if i % 50 == 0 or i == total:
                found = sum(1 for b in enriched if b.get("description"))
                print(f"진행: {i}/{total} | description 확보: {found}건")

            time.sleep(self.delay)

        return enriched


if __name__ == "__main__":
    collector = SampleCollector()
    books = collector.collect(total=1000)
    enricher = AladinEnricher()
    enriched = enricher.enrich(books)
    collector.save(enriched)
