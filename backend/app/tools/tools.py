import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DATA4LIBRARY_KEY = os.getenv("LIBRARY_API_KEY", "")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")


@tool
def search_library(district_name: str) -> str:
    """
    사용자가 도서관 위치를 물어볼 때 호출합니다.
    주의: 반드시 서울의 '구' 단위 이름(예: 강남구, 서초구, 마포구)을 입력해야 합니다.
    만약 사용자가 '삼성동'처럼 '동' 단위를 물어보면, 당신(AI)이 해당 동이 속한 '구'를 유추해서 입력하세요. (예: 삼성동 -> 강남구)
    """
    region_codes = {
        "종로구": "11010", "중구": "11020", "용산구": "11030", "성동구": "11040",
        "광진구": "11050", "동대문구": "11060", "중랑구": "11070", "성북구": "11080",
        "강북구": "11090", "도봉구": "11100", "노원구": "11110", "은평구": "11120",
        "서대문구": "11130", "마포구": "11140", "양천구": "11150", "강서구": "11160",
        "구로구": "11170", "금천구": "11180", "영등포구": "11190", "동작구": "11200",
        "관악구": "11210", "서초구": "11220", "강남구": "11230", "송파구": "11240",
        "강동구": "11250"
    }

    dtl_region = region_codes.get(district_name)
    if not dtl_region:
        return f"'{district_name}'에 대한 지역 코드를 찾을 수 없습니다. 서울의 '구' 단위로 정확히 입력되었는지 확인하세요."

    url = "http://data4library.kr/api/libSrch"
    params = {
        "authKey": DATA4LIBRARY_KEY,
        "region": "11",
        "dtl_region": dtl_region,
        "pageSize": 3,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        print("\n" + "▼"*50)
        print("📡 [도서관 목록 조회 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()

        libs = data.get('response', {}).get('libs', [])
        if not libs:
            return f"{district_name} 근처의 도서관 정보를 찾을 수 없습니다."

        result = []
        for item in libs:
            lib = item['lib']
            name = lib.get('libName', '이름 없음')
            lib_code = lib.get('libCode', '코드 없음')
            address = lib.get('address', '주소 없음')
            operating_time = lib.get('operatingTime', '운영시간 정보 없음')
            result.append(f"- {name} (도서관 코드: {lib_code})\n  주소: {address}\n  운영시간: {operating_time}")

        return "\n\n".join(result)

    except Exception as e:
        return f"API 호출 중 오류 발생: {str(e)}"


@tool
def check_book_availability(lib_code: str, isbn13: str) -> str:
    """
    사용자가 특정 도서관에서 특정 책을 '지금 빌릴 수 있는지', '대출 가능한지' 물어볼 때 호출합니다.
    주의: 이 도구를 사용하려면 먼저 대상 도서관의 코드(lib_code)와 책의 ISBN(isbn13)을 알고 있어야 합니다.
    """
    url = "http://data4library.kr/api/bookExist"
    params = {
        "authKey": DATA4LIBRARY_KEY,
        "libCode": lib_code,
        "isbn13": isbn13,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        print("\n" + "▼"*50)
        print("📡 [대출 가능 여부 조회 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()

        result_data = data.get('response', {}).get('result', {})
        has_book = result_data.get('hasBook')
        loan_available = result_data.get('loanAvailable')

        if has_book == "N":
            return "해당 도서관에는 이 책이 소장되어 있지 않습니다."
        elif has_book == "Y" and loan_available == "Y":
            return "현재 대출 가능합니다! (전일 기준 데이터이므로 방문 전 확인 권장)"
        elif has_book == "Y" and loan_available == "N":
            return "소장 중이나 현재 누군가 대출 중입니다. 예약이 필요할 수 있습니다."
        else:
            return "상태 정보를 확인할 수 없습니다."

    except Exception as e:
        return f"대출 가능 여부 확인 중 오류 발생: {str(e)}"


@tool
def check_book_availability_in_district(district_name: str, isbn13: str) -> str:
    """
    특정 구(예: 강남구)의 도서관들에서 특정 책의 대출 가능 여부를 한 번에 확인합니다.
    district_name은 서울의 '구' 단위(예: 강남구, 마포구), isbn13은 책의 13자리 ISBN입니다.
    """
    region_codes = {
        "종로구": "11010", "중구": "11020", "용산구": "11030", "성동구": "11040",
        "광진구": "11050", "동대문구": "11060", "중랑구": "11070", "성북구": "11080",
        "강북구": "11090", "도봉구": "11100", "노원구": "11110", "은평구": "11120",
        "서대문구": "11130", "마포구": "11140", "양천구": "11150", "강서구": "11160",
        "구로구": "11170", "금천구": "11180", "영등포구": "11190", "동작구": "11200",
        "관악구": "11210", "서초구": "11220", "강남구": "11230", "송파구": "11240",
        "강동구": "11250"
    }

    dtl_region = region_codes.get(district_name)
    if not dtl_region:
        return f"'{district_name}'에 대한 지역 코드를 찾을 수 없습니다."

    # ── 1. 도서관 목록 조회 ──────────────────────────────────
    try:
        lib_resp = requests.get("http://data4library.kr/api/libSrch", params={
            "authKey": DATA4LIBRARY_KEY,
            "region": "11",
            "dtl_region": dtl_region,
            "pageSize": 3,
            "format": "json",
        })
        print("\n" + "▼"*50)
        print("📡 [도서관 목록 조회 API]")
        print(f"👉 URL: {lib_resp.url}")
        print("▲"*50 + "\n")
        libs = lib_resp.json().get("response", {}).get("libs", [])
    except Exception as e:
        return f"도서관 목록 조회 중 오류: {e}"

    if not libs:
        return f"{district_name} 근처의 도서관 정보를 찾을 수 없습니다."

    # ── 2. 각 도서관별 대출 가능 여부 확인 ──────────────────
    results = []
    for item in libs:
        lib = item["lib"]
        lib_name = lib.get("libName", "이름 없음")
        lib_code = lib.get("libCode", "")

        try:
            avail_resp = requests.get("http://data4library.kr/api/bookExist", params={
                "authKey": DATA4LIBRARY_KEY,
                "libCode": lib_code,
                "isbn13": isbn13,
                "format": "json",
            })
            print("\n" + "▼"*50)
            print(f"📡 [대출 가능 여부 조회 API] {lib_name}")
            print(f"👉 URL: {avail_resp.url}")
            print("▲"*50 + "\n")
            result_data = avail_resp.json().get("response", {}).get("result", {})
            has_book = result_data.get("hasBook")
            loan_available = result_data.get("loanAvailable")

            if has_book == "N":
                status = "미소장"
            elif has_book == "Y" and loan_available == "Y":
                status = "대출 가능"
            elif has_book == "Y" and loan_available == "N":
                status = "대출 중 (예약 필요)"
            else:
                status = "확인 불가"
        except Exception as e:
            status = f"오류: {e}"

        results.append(f"- {lib_name}: {status}")

    return "\n".join(results)


@tool
def get_popular_books(age: str = None, gender: str = None) -> str:
    """
    사용자가 '요즘 인기 있는 책', '베스트셀러', '많이 읽는 책' 등을 추천해달라고 할 때 호출합니다.
    질문에서 연령대나 성별이 파악된다면 해당 코드를 인자(파라미터)로 반드시 넘겨주세요.
    - age: "0"(영유아), "6"(유아), "8"(초등), "14"(청소년), "20"(20대), "30"(30대), "40"(40대), "50"(50대), "60"(60세 이상)
    - gender: "0"(남성), "1"(여성)
    """
    url = "http://data4library.kr/api/loanItemSrch"
    params = {
        "authKey": DATA4LIBRARY_KEY,
        "format": "json",
        "pageSize": 5
    }
    if age:
        params["age"] = age
    if gender:
        params["gender"] = gender

    try:
        response = requests.get(url, params=params)
        print("\n" + "▼"*50)
        print("📡 [인기 대출 도서 조회 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()

        docs = data.get('response', {}).get('docs', [])
        if not docs:
            return "조건에 맞는 인기 도서 데이터를 찾을 수 없습니다."

        result = []
        for item in docs:
            doc = item['doc']
            ranking = doc.get('ranking', '순위 없음')
            bookname = doc.get('bookname', '제목 없음')
            authors = doc.get('authors', '저자 미상')
            publisher = doc.get('publisher', '출판사 미상')
            loan_count = doc.get('loan_count', '0')
            result.append(f"{ranking}위. {bookname}\n  - 저자: {authors}\n  - 출판사: {publisher}\n  - 누적 대출건수: {loan_count}건")

        condition_text = "전체 이용자"
        if age or gender:
            condition_text = f"선택된 조건(연령:{age if age else '전체'}, 성별:{gender if gender else '전체'})의"

        return f"[{condition_text} 인기 대출 도서 상위 5권]\n\n" + "\n\n".join(result)

    except Exception as e:
        return f"인기 도서 API 호출 중 오류 발생: {str(e)}"


@tool
def get_trending_books(search_date: str = None) -> str:
    """
    사용자가 '요즘 갑자기 뜨는 책', '역주행 베스트셀러', '대출 급상승 도서', '최근 트렌드'를 물어볼 때 호출합니다.
    - search_date: 'YYYY-MM-DD' 형식의 날짜.
    만약 사용자가 특정 날짜를 언급하지 않았다면 이 파라미터를 비워두세요(None). 코드가 알아서 최근 날짜로 검색합니다.
    """
    if not search_date:
        yesterday = datetime.now() - timedelta(days=1)
        search_date = yesterday.strftime("%Y-%m-%d")

    url = "http://data4library.kr/api/hotTrend"
    params = {
        "authKey": DATA4LIBRARY_KEY,
        "searchDt": search_date,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        print("\n" + "▼"*50)
        print("📡 [급상승 도서 조회 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()

        results = data.get('response', {}).get('results', [])
        if not results:
            return f"{search_date} 기준 급상승 도서 데이터를 찾을 수 없습니다."

        target_date_data = results[0]['result']
        actual_date = target_date_data.get('date', search_date)
        docs = target_date_data.get('docs', [])

        if not docs:
            return f"{actual_date} 기준 급상승 도서가 없습니다."

        formatted_result = []
        for item in docs:
            doc = item['doc']
            no = doc.get('no', '0')
            bookname = doc.get('bookname', '제목 없음')
            authors = doc.get('authors', '저자 미상')
            difference = doc.get('difference', '0')
            formatted_result.append(f"{no}위. {bookname} (저자: {authors}) \n   🔥 무려 {difference}계단 대출 순위 상승!")

        return f"[{actual_date} 기준 대출 급상승 도서 TOP 5]\n\n" + "\n\n".join(formatted_result)

    except Exception as e:
        return f"급상승 도서 API 호출 중 오류 발생: {str(e)}"


@tool
def get_book_recommendations(isbn13: str) -> str:
    """
    사용자가 특정 책(예: 채식주의자, 소년이 온다 등)을 언급하며 '비슷한 책', '같이 읽기 좋은 책', '이 책을 읽은 사람들이 좋아하는 책'을 추천해달라고 할 때 호출합니다.
    주의: 이 도구를 사용하려면 반드시 해당 책의 13자리 ISBN 번호가 필요합니다.
    만약 ISBN을 모른다면 책 검색 도구를 먼저 사용해서 ISBN을 알아낸 뒤 이 도구를 호출하세요.
    """
    url = "http://data4library.kr/api/usageAnalysisList"
    params = {
        "authKey": DATA4LIBRARY_KEY,
        "isbn13": isbn13,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        print("\n" + "▼"*50)
        print("📡 [연관 도서 추천 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()

        response_data = data.get('response', {})
        co_loan_books = response_data.get('coLoanBooks', [])
        mania_rec_books = response_data.get('maniaRecBooks', [])

        if not co_loan_books and not mania_rec_books:
            return "해당 도서와 연관된 추천 도서 데이터를 찾을 수 없습니다."

        result_text = []
        if co_loan_books:
            result_text.append("📖 [이 책을 빌린 사람들이 함께 빌린 책]")
            for i, item in enumerate(co_loan_books[:3]):
                book = item['book']
                result_text.append(f"  {i+1}. {book.get('bookname', '제목 없음')} (저자: {book.get('authors', '저자 미상')})")

        if mania_rec_books:
            result_text.append("\n💡 [이 책의 마니아들을 위한 맞춤 추천]")
            for i, item in enumerate(mania_rec_books[:3]):
                book = item['book']
                result_text.append(f"  {i+1}. {book.get('bookname', '제목 없음')} (저자: {book.get('authors', '저자 미상')})")

        return "\n".join(result_text)

    except Exception as e:
        return f"연관 도서 추천 API 호출 중 오류 발생: {str(e)}"


@tool
def get_book_isbn(book_title: str) -> str:
    """
    사용자가 언급한 책 제목으로 13자리 ISBN 번호와 표지 이미지 링크를 검색할 때 호출합니다.
    네이버 책 검색 API를 사용하여 융통성 있고 정확하게 데이터를 찾아옵니다.
    """
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": book_title, "display": 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        print("\n" + "▼"*50)
        print("📡 [ISBN + 표지 URL 검색 API 통신 로그]")
        print(f"👉 실제 호출된 URL: {response.url}")
        print("▲"*50 + "\n")
        data = response.json()
        items = data.get('items', [])

        if not items:
            return f"'{book_title}'에 대한 책 정보를 찾을 수 없습니다."

        isbn_raw = items[0].get('isbn', '')
        isbn13 = isbn_raw.split()[-1] if isbn_raw else ""
        full_title = items[0].get('title', book_title)
        author = items[0].get('author', '저자 미상')
        image_url = items[0].get('image', '')

        return f"책 '{full_title}' (저자: {author})의 정확한 ISBN13 번호는 '{isbn13}' 이며, 표지 이미지 링크는 '{image_url}' 입니다. 메시지 출력 시 이 이미지를 반드시 마크다운으로 보여주세요."

    except Exception as e:
        return f"네이버 ISBN 검색 API 호출 중 오류 발생: {str(e)}"
