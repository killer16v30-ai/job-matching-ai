"""
공공기관 취업 매칭 AI (공공데이터 API 연동)

재정경제부「공공기관 채용정보 조회서비스」API에서 채용공고를 불러와
취준생 프로필과 비교한 뒤 OpenAI로 TOP 3를 추천합니다.

실행 예:
  uv run streamlit run 11.Hackerton/code2/app.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 경로 · 환경 변수
# ---------------------------------------------------------------------------
# 이 파일 위치: .../AI-Education/11.Hackerton/code2/app.py
# 프로젝트 루트 .env (다른 예제와 동일하게 상위 2단계)
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DATA_GO_KR_API_KEY = os.getenv("DATA_GO_KR_API_KEY", "").strip()

# 공공데이터포털 API 기본 주소 (프롬프트에 명시된 End Point)
API_BASE_URL = "https://apis.data.go.kr/1051000/recruitment"
API_LIST_PATH = "/list"

# OpenAI에 넘길 최대 공고 수 (토큰·비용 절약)
MAX_JOBS_FOR_LLM = 40
MODEL_NAME = "gpt-4o-mini"

# API numOfRows 선택지 (selectbox)
NUM_OF_ROWS_OPTIONS = [10, 20, 30, 50, 100]

# LLM·표시에 쓰는 표준 열 이름 (기존 code/app.py와 동일)
STANDARD_COLUMNS = ["기관명", "직무", "근무지", "우대사항", "필요역량", "채용설명"]

# API 원본 필드 → 표준 열 매핑 (한글 열로 통일해 기존 추천 로직 재사용)
API_FIELD_MAP = {
    "기관명": "instNm",
    "직무": "ncsCdNmLst",
    "근무지": "workRgnNmLst",
    "우대사항": "prefCondCn",
    "필요역량": "aplyQlfcCn",
    "채용설명": "recrutPbancTtl",
}

# 표에 보여줄 추가 열 (사용자가 목록에서 한눈에 보도록)
TABLE_EXTRA_COLUMNS = {
    "채용구분": "recrutSeNm",
    "고용형태": "hireTypeNmLst",
    "채용인원": "recrutNope",
    "접수시작": "pbancBgngYmd",
    "접수마감": "pbancEndYmd",
    "공고URL": "srcUrl",
}


# ---------------------------------------------------------------------------
# 공공데이터 API 호출
# ---------------------------------------------------------------------------
class PublicDataApiError(Exception):
    """공공데이터 API 호출·응답 처리 중 발생한 오류 (화면에 친절한 메시지로 표시)."""


def calc_total_pages(total_count: int, num_of_rows: int) -> int:
    """totalCount와 numOfRows로 전체 페이지 수를 계산합니다."""
    if total_count <= 0 or num_of_rows <= 0:
        return 1
    return max(1, math.ceil(total_count / num_of_rows))


def init_pagination_state() -> None:
    """페이지네이션용 session_state 초기값."""
    if "api_page_no" not in st.session_state:
        st.session_state.api_page_no = 1
    if "api_num_of_rows" not in st.session_state:
        st.session_state.api_num_of_rows = 30


def store_fetched_jobs(
    raw_df: pd.DataFrame,
    total_count: int,
    page_no: int,
    num_of_rows: int,
) -> None:
    """API 조회 결과를 session_state에 저장 (표·추천·페이지 정보)."""
    st.session_state["jobs_raw_df"] = raw_df
    st.session_state["jobs_standard_df"] = api_raw_to_standard_df(raw_df)
    st.session_state["jobs_total_count"] = total_count
    st.session_state["jobs_page_no"] = page_no
    st.session_state["jobs_num_of_rows"] = num_of_rows
    st.session_state.api_page_no = page_no
    st.session_state.api_num_of_rows = num_of_rows


def fetch_and_store_jobs(page_no: int, num_of_rows: int) -> None:
    """API 호출 후 결과 저장. 실패 시 PublicDataApiError."""
    raw_df, total = fetch_recruitment_list(
        DATA_GO_KR_API_KEY,
        page_no=page_no,
        num_of_rows=num_of_rows,
    )
    store_fetched_jobs(raw_df, total, page_no, num_of_rows)


def fetch_recruitment_list(
    service_key: str,
    page_no: int = 1,
    num_of_rows: int = 20,
) -> tuple[pd.DataFrame, int]:
    """
    requests로 채용공고 목록 API를 호출하고 DataFrame으로 변환합니다.

    Returns:
        (공고 DataFrame, 전체 건수 totalCount)
    """
    if not service_key:
        raise PublicDataApiError(
            f"`{ENV_PATH}` 또는 이 폴더의 `.env`에 `DATA_GO_KR_API_KEY`를 설정해 주세요. "
            "`.env.example`을 참고할 수 있습니다."
        )

    # 공공데이터포털 REST API 공통 파라미터
    params = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "resultType": "json",  # JSON 응답 우선 사용
    }

    url = f"{API_BASE_URL}{API_LIST_PATH}"

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.Timeout:
        raise PublicDataApiError(
            "API 응답 시간이 초과되었습니다. 잠시 후 다시 시도하거나 "
            "한 번에 불러올 건수(numOfRows)를 줄여 보세요."
        ) from None
    except requests.exceptions.ConnectionError:
        raise PublicDataApiError(
            "네트워크 연결에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요."
        ) from None
    except requests.exceptions.RequestException as e:
        raise PublicDataApiError(f"API 요청 중 오류가 발생했습니다: {e}") from e

    # HTTP 상태 코드별 안내
    if response.status_code == 401:
        raise PublicDataApiError(
            "API 인증에 실패했습니다(401 Unauthorized). "
            "`DATA_GO_KR_API_KEY`가 올바른지, 공공데이터포털에서 해당 API 활용신청이 "
            "승인되었는지 확인해 주세요."
        )
    if response.status_code == 403:
        raise PublicDataApiError(
            "API 접근이 거부되었습니다(403). 일일 트래픽 한도를 초과했을 수 있습니다."
        )
    if response.status_code == 404:
        raise PublicDataApiError(
            "요청한 API 주소를 찾을 수 없습니다(404). "
            f"주소가 `{API_BASE_URL}{API_LIST_PATH}` 인지 확인해 주세요."
        )
    if response.status_code >= 500:
        raise PublicDataApiError(
            f"공공데이터 서버 오류입니다(HTTP {response.status_code}). "
            "잠시 후 다시 시도해 주세요."
        )
    if response.status_code != 200:
        raise PublicDataApiError(
            f"예상치 못한 응답입니다(HTTP {response.status_code}). "
            f"응답 내용: {response.text[:200]}"
        )

    # 본문이 JSON인지 확인
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise PublicDataApiError(
            "API 응답을 JSON으로 읽을 수 없습니다. "
            "serviceKey 인코딩 문제이거나 서버 점검 중일 수 있습니다."
        ) from None

    # API 자체 결과 코드 (200이면 성공)
    result_code = data.get("resultCode")
    if result_code != 200:
        msg = data.get("resultMsg", "알 수 없는 오류")
        raise PublicDataApiError(f"API 처리 결과 오류 (코드 {result_code}): {msg}")

    items = data.get("result")
    if not items:
        raise PublicDataApiError("조회된 채용공고가 없습니다. 페이지 번호를 바꿔 보세요.")

    total_count = int(data.get("totalCount", len(items)))

    # pandas DataFrame으로 변환 (원본 필드명 유지)
    raw_df = pd.DataFrame(items)
    return raw_df, total_count


def api_raw_to_standard_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    API 원본 DataFrame을 추천·표시용 표준 한글 열 DataFrame으로 바꿉니다.
    채용설명은 제목 + 접수기간 + 전형방법을 합쳐 풍부하게 만듭니다.
    """
    rows: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        # 접수기간 문자열 (YYYYMMDD → 보기 좋게)
        bgng = str(row.get("pbancBgngYmd", "") or "")
        end = str(row.get("pbancEndYmd", "") or "")
        period = ""
        if bgng or end:
            period = f"접수기간: {bgng} ~ {end}"

        title = str(row.get("recrutPbancTtl", "") or "")
        scrn = str(row.get("scrnprcdrMthdExpln", "") or "")
        description_parts = [p for p in [title, period, scrn] if p]
        description = "\n".join(description_parts)

        pref = row.get("prefCondCn") or row.get("prefCn") or ""

        rows.append(
            {
                "기관명": row.get("instNm", ""),
                "직무": row.get("ncsCdNmLst", ""),
                "근무지": row.get("workRgnNmLst", ""),
                "우대사항": pref,
                "필요역량": row.get("aplyQlfcCn", ""),
                "채용설명": description,
            }
        )
    return pd.DataFrame(rows)


def build_display_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit 표에 보여줄 한글 열 목록 DataFrame."""
    display_cols: dict[str, Any] = {}
    for label, api_key in {**API_FIELD_MAP, **TABLE_EXTRA_COLUMNS}.items():
        if api_key in raw_df.columns:
            display_cols[label] = raw_df[api_key]
    return pd.DataFrame(display_cols)


# ---------------------------------------------------------------------------
# OpenAI 추천 (기존 code/app.py와 동일한 흐름)
# ---------------------------------------------------------------------------
def jobs_to_text(df: pd.DataFrame, limit: int = MAX_JOBS_FOR_LLM) -> str:
    """공고 목록을 LLM이 읽기 쉬운 텍스트로 변환."""
    subset = df.head(limit)
    blocks: list[str] = []
    for idx, row in subset.iterrows():
        blocks.append(
            f"[공고 #{idx + 1}]\n"
            f"기관명: {row['기관명']}\n"
            f"직무: {row['직무']}\n"
            f"근무지: {row['근무지']}\n"
            f"우대사항: {row['우대사항']}\n"
            f"필요역량: {row['필요역량']}\n"
            f"채용설명: {row['채용설명']}\n"
        )
    extra = ""
    if len(df) > limit:
        extra = f"\n(참고: 전체 {len(df)}건 중 상위 {limit}건만 비교에 사용했습니다.)"
    return "\n".join(blocks) + extra


def profile_to_text(profile: dict[str, str]) -> str:
    """취준생 입력 폼 값을 하나의 텍스트로 묶음."""
    return (
        f"이름: {profile['name']}\n"
        f"희망 직무: {profile['desired_job']}\n"
        f"희망 근무지: {profile['desired_location']}\n"
        f"보유 역량: {profile['skills']}\n"
        f"경력/경험: {profile['experience']}\n"
        f"관심 분야: {profile['interests']}\n"
    )


SYSTEM_PROMPT = """당신은 공공기관 취업을 준비하는 취준생을 돕는 채용 매칭 전문가입니다.

규칙:
1. 제공된 채용공고 목록 안에서만 추천하세요. 목록에 없는 기관·직무를 지어내지 마세요.
2. 반드시 정확히 3개의 추천만 JSON으로 반환하세요 (rank 1, 2, 3).
3. 각 추천의 organization, job_title은 공고 원문의 기관명·직무와 일치해야 합니다.
4. 추천 이유·잘 맞는 점·부족한 점·준비하면 좋은 것은 구체적이고 실용적으로 한국어로 작성하세요.
5. 응답은 아래 JSON 스키마만 출력하고, 다른 설명 문장은 넣지 마세요.

JSON 스키마:
{
  "recommendations": [
    {
      "rank": 1,
      "organization": "기관명",
      "job_title": "직무",
      "reason": "추천 이유",
      "good_match": "잘 맞는 점",
      "gaps": "부족한 점",
      "preparation": "준비하면 좋은 것"
    }
  ]
}"""


def fetch_recommendations(
    client: OpenAI,
    profile: dict[str, str],
    jobs_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """OpenAI API로 TOP 3 추천 JSON을 받아 파싱."""
    user_content = (
        "## 취준생 정보\n"
        f"{profile_to_text(profile)}\n\n"
        "## 채용공고 목록 (공공데이터 API)\n"
        f"{jobs_to_text(jobs_df)}\n\n"
        "위 정보를 바탕으로 이 취준생에게 가장 적합한 공고 TOP 3를 추천하세요."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    recs = data.get("recommendations", [])
    if not isinstance(recs, list):
        raise ValueError("API 응답 형식이 올바르지 않습니다 (recommendations 목록 없음).")
    return recs[:3]


def render_recommendation_card(rec: dict[str, Any]) -> None:
    """순위별 추천 카드 출력."""
    rank = rec.get("rank", "-")
    org = rec.get("organization", "(기관명 없음)")
    job = rec.get("job_title", "(직무 없음)")
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    try:
        rank_num = int(rank)
    except (TypeError, ValueError):
        rank_num = 0
    medal = medals.get(rank_num, "📌")

    st.subheader(f"{medal} {rank}위 — {org} · {job}")
    st.markdown(f"**추천 이유**  \n{rec.get('reason', '')}")
    st.markdown(f"**잘 맞는 점**  \n{rec.get('good_match', '')}")
    st.markdown(f"**부족한 점**  \n{rec.get('gaps', '')}")
    st.markdown(f"**준비하면 좋은 것**  \n{rec.get('preparation', '')}")
    st.divider()


# ---------------------------------------------------------------------------
# Streamlit 메인 화면
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="공공기관 취업 매칭 (공공데이터)",
        page_icon="🏛️",
        layout="wide",
    )

    st.title("🏛️ 공공기관 취업 매칭 AI")
    st.caption(
        "재정경제부「공공기관 채용정보 조회서비스」API에서 공고를 불러와 "
        "프로필과 비교한 뒤 TOP 3를 추천합니다."
    )

    # 필수 키 안내
    if not OPENAI_API_KEY:
        st.error(
            f"`{ENV_PATH}`에 `OPENAI_API_KEY`를 설정해 주세요. "
            "`.env.example`을 참고할 수 있습니다."
        )
        st.stop()
    if not DATA_GO_KR_API_KEY:
        st.error(
            f"`{ENV_PATH}`에 `DATA_GO_KR_API_KEY`를 설정해 주세요. "
            "공공데이터포털에서 발급한 인증키를 넣어야 합니다."
        )
        st.stop()

    with st.sidebar:
        st.header("API 정보")
        st.caption(f"End Point: `{API_BASE_URL}{API_LIST_PATH}`")

    init_pagination_state()

    # --- 1. 취준생 정보 ---
    st.header("1. 취준생 정보")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("이름", placeholder="홍길동")
        desired_job = st.text_input("희망 직무", placeholder="데이터 분석, 경영·회계 등")
        desired_location = st.text_input("희망 근무지", placeholder="서울, 대전, 부산 등")
    with col2:
        skills = st.text_area("보유 역량", placeholder="Python, SQL, 통계 분석 ...", height=100)
        experience = st.text_area("경력/경험", placeholder="인턴, 프로젝트, 자격증 ...", height=100)
        interests = st.text_area("관심 분야", placeholder="공공데이터, 환경 정책 ...", height=100)

    profile = {
        "name": name.strip(),
        "desired_job": desired_job.strip(),
        "desired_location": desired_location.strip(),
        "skills": skills.strip(),
        "experience": experience.strip(),
        "interests": interests.strip(),
    }

    # --- 2. 공공데이터 API에서 채용공고 불러오기 ---
    st.header("2. 채용공고 (공공데이터 API)")

    # 저장된 totalCount (이전 조회 결과; 없으면 0)
    stored_total = int(st.session_state.get("jobs_total_count") or 0)
    default_rows = int(st.session_state.get("api_num_of_rows", 30))
    row_index = (
        NUM_OF_ROWS_OPTIONS.index(default_rows)
        if default_rows in NUM_OF_ROWS_OPTIONS
        else NUM_OF_ROWS_OPTIONS.index(30)
    )

    st.subheader("페이지 설정")

    # 2) numOfRows — selectbox로 선택
    num_of_rows = st.selectbox(
        "한 페이지당 데이터 수 (numOfRows)",
        options=NUM_OF_ROWS_OPTIONS,
        index=row_index,
        help="API 요청 시 numOfRows 파라미터로 전달됩니다.",
    )
    st.session_state.api_num_of_rows = int(num_of_rows)

    # totalCount가 있으면 전체 페이지 수 계산 (다음 버튼 활성화에 사용)
    total_pages = calc_total_pages(stored_total, int(num_of_rows))
    current_page_before_nav = int(st.session_state.api_page_no)

    # 4·5) totalCount · 현재 페이지 안내
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        if stored_total > 0:
            st.metric("전체 데이터 수 (totalCount)", f"{stored_total:,}건")
        else:
            st.caption("전체 데이터 수: 조회 후 표시됩니다.")
    with info_col2:
        if stored_total > 0:
            st.metric(
                "현재 보고 있는 페이지",
                f"{current_page_before_nav} / {total_pages}",
            )
        else:
            st.metric("현재 보고 있는 페이지", f"{current_page_before_nav}페이지")

    # 6) 이전 / 1) pageNo / 다음
    nav_prev, nav_page, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        prev_clicked = st.button(
            "◀ 이전 페이지",
            use_container_width=True,
            disabled=current_page_before_nav <= 1,
            help="pageNo를 1 감소시키고 API를 다시 호출합니다.",
        )
    with nav_page:
        # 1) pageNo — number_input
        page_no = st.number_input(
            "페이지 번호 (pageNo)",
            min_value=1,
            max_value=total_pages if stored_total > 0 else None,
            value=current_page_before_nav,
            step=1,
            help="이동할 페이지 번호를 입력한 뒤 「불러오기」를 누르세요.",
        )
    with nav_next:
        on_last_page = stored_total > 0 and current_page_before_nav >= total_pages
        next_clicked = st.button(
            "다음 페이지 ▶",
            use_container_width=True,
            disabled=on_last_page,
            help="pageNo를 1 증가시키고 API를 다시 호출합니다.",
        )

    load_clicked = st.button("채용공고 불러오기", type="secondary", use_container_width=True)

    # 네비게이션·불러오기 시 사용할 pageNo 결정
    fetch_page_no = int(page_no)
    if prev_clicked:
        fetch_page_no = max(1, current_page_before_nav - 1)
        st.session_state.api_page_no = fetch_page_no
    elif next_clicked:
        fetch_page_no = min(total_pages, current_page_before_nav + 1) if stored_total > 0 else current_page_before_nav + 1
        st.session_state.api_page_no = fetch_page_no
    elif load_clicked:
        st.session_state.api_page_no = fetch_page_no

    should_fetch = load_clicked or prev_clicked or next_clicked

    if should_fetch:
        with st.spinner(
            f"공공데이터 API 호출 중… (pageNo={fetch_page_no}, numOfRows={num_of_rows})"
        ):
            try:
                fetch_and_store_jobs(fetch_page_no, int(num_of_rows))
            except PublicDataApiError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"데이터 처리 중 예기치 않은 오류: {e}")
                st.stop()

    raw_df = st.session_state.get("jobs_raw_df")
    jobs_df = st.session_state.get("jobs_standard_df")
    total_count = int(st.session_state.get("jobs_total_count") or 0)
    viewed_page = int(st.session_state.get("jobs_page_no") or st.session_state.api_page_no)
    viewed_rows = int(st.session_state.get("jobs_num_of_rows") or num_of_rows)
    viewed_total_pages = calc_total_pages(total_count, viewed_rows)

    if raw_df is not None and jobs_df is not None:
        st.success(
            f"이번 조회: **{len(jobs_df)}건** · "
            f"**{viewed_page}페이지** / {viewed_total_pages} · "
            f"전체 **{total_count:,}건** (totalCount)"
        )
        st.dataframe(build_display_table(raw_df), use_container_width=True, hide_index=True)
        with st.expander("AI 추천에 사용하는 표준 데이터 미리보기"):
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)

    # --- 3. AI 추천 ---
    st.header("3. AI 추천")
    recommend_clicked = st.button("추천받기", type="primary", use_container_width=True)

    if recommend_clicked:
        if not profile["name"] or not profile["desired_job"]:
            st.warning("이름과 희망 직무는 필수입니다.")
            st.stop()
        if jobs_df is None or jobs_df.empty:
            st.warning("먼저 「채용공고 불러오기」로 공고를 불러와 주세요.")
            st.stop()

        with st.spinner("AI가 공고를 비교하고 있습니다..."):
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                recommendations = fetch_recommendations(client, profile, jobs_df)
            except json.JSONDecodeError:
                st.error("AI 응답을 해석하지 못했습니다. 잠시 후 다시 시도해 주세요.")
                st.stop()
            except Exception as e:
                st.error(f"추천 생성 중 오류가 발생했습니다: {e}")
                st.stop()

        st.session_state["last_recommendations"] = recommendations
        st.session_state["last_profile_name"] = profile["name"]

    # --- 4. TOP 3 결과 ---
    if st.session_state.get("last_recommendations"):
        st.header("4. TOP 3 추천 결과")
        who = st.session_state.get("last_profile_name", "")
        if who:
            st.info(f"**{who}** 님을 위한 맞춤 추천입니다.")
        for rec in st.session_state["last_recommendations"]:
            render_recommendation_card(rec)


if __name__ == "__main__":
    main()
