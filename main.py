import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균 기온 데이터를 이용해 선형회귀로 기온을 예측합니다.")


# -----------------------------------
# 데이터 불러오기
# -----------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 첫 번째 열에 BOM이 붙어 있는 경우를 대비
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 데이터만 사용
    df = df.dropna(subset=["날짜", "평균기온"])

    # 2025년까지의 데이터만 사용
    df = df[df["날짜"].dt.year <= 2025]

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# -----------------------------------
# 연도별 평균기온 계산
# -----------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------------
# 회귀용 변수
# 1908년부터 지난 연수
# -----------------------------------
yearly["지난연수"] = yearly["연도"] - 1908


# -----------------------------------
# 선형회귀
# -----------------------------------
x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값
yearly["회귀기온"] = slope * yearly["지난연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# -----------------------------------
# 회귀선 정보
# -----------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


st.subheader("📊 연평균 기온과 회귀선")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("회귀에 사용한 해의 개수", f"{data_count}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")


# -----------------------------------
# 산점도 + 회귀선
# -----------------------------------
fig = go.Figure()

# 실제 연평균 기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        text=yearly["관측일수"],
        customdata=yearly["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균 기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀기온"],
        mode="lines",
        name="회귀선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    )
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------------
# 상관계수
# -----------------------------------
st.subheader("📈 상관계수")

st.write(
    f"연도와 연평균 기온의 상관계수는 **{correlation:.3f}**입니다."
)

st.caption(
    "상관계수는 -1에 가까울수록 음의 관계, "
    "1에 가까울수록 양의 관계를 나타냅니다."
)


# -----------------------------------
# 회귀식
# -----------------------------------
st.subheader("📐 회귀식")

st.write(
    f"연평균 기온 = {slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.4f}"
)


# -----------------------------------
# 연도 슬라이더
# -----------------------------------
st.subheader("🔮 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 회귀 예측값
selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept


# -----------------------------------
# 예상 기온 크게 표시
# -----------------------------------
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        background-color: #f5f5f5;
        margin-top: 10px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 22px;">
            {selected_year}년 예상 연평균 기온
        </div>
        <div style="
            font-size: 55px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# 선택한 연도를 그래프에 표시
# -----------------------------------
prediction_years = np.array([1900, 2100])
prediction_x = prediction_years - 1908
prediction_temps = slope * prediction_x + intercept

prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균 기온"
    )
)

prediction_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀선"
    )
)

prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        marker=dict(size=14),
        name=f"{selected_year}년 예측"
    )
)

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=10
    ),
    hovermode="closest"
)

st.plotly_chart(prediction_fig, use_container_width=True)


# -----------------------------------
# 데이터 확인
# -----------------------------------
with st.expander("사용된 연도별 데이터 보기"):
    st.dataframe(
        yearly[
            ["연도", "관측일수", "연평균기온", "지난연수", "회귀기온"]
        ].round(2),
        use_container_width=True
    )


