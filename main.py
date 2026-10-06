import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 예측")
st.write(
    "과거의 연평균 기온으로 회귀모델을 학습하고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# -----------------------------------
# 데이터 불러오기
# -----------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"])

    # 2025년까지의 데이터만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()

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

# 관측일수가 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------------
# 학습 / 테스트 데이터
# -----------------------------------
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# -----------------------------------
# 회귀모델 함수
# -----------------------------------
def make_model(train_data):
    X_train = train_data[["연도"]]
    y_train = train_data["연평균기온"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# 모델 학습
model_50 = make_model(train_50)
model_100 = make_model(train_100)


# -----------------------------------
# 테스트 데이터 예측
# -----------------------------------
X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# -----------------------------------
# 성능 평가
# -----------------------------------
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# -----------------------------------
# 회귀선 기울기
# -----------------------------------
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]


# -----------------------------------
# 데이터 개수 확인
# -----------------------------------
st.subheader("📊 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}년"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}년"
    )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}년"
    )


st.write(
    f"**학습 데이터:** 1956~2005년 / 1906~2005년"
)

st.write(
    f"**테스트 데이터:** 2006~2025년"
)


# -----------------------------------
# 회귀선 기울기 비교
# -----------------------------------
st.subheader("📈 회귀선 기울기 비교")

slope_df = pd.DataFrame({
    "모델": ["최근 50년", "최근 100년"],
    "기울기 (℃/년)": [slope_50, slope_100]
})

st.dataframe(
    slope_df.style.format({
        "기울기 (℃/년)": "{:.5f}"
    }),
    use_container_width=True
)

st.write(
    f"최근 50년 모델의 기울기: **{slope_50:.5f} ℃/년**"
)

st.write(
    f"최근 100년 모델의 기울기: **{slope_100:.5f} ℃/년**"
)


# -----------------------------------
# 실제 데이터 + 두 회귀선 그래프
# -----------------------------------
st.subheader("📉 실제 기온과 두 회귀선 비교")

# 그래프를 그릴 전체 연도
graph_years = pd.DataFrame({
    "연도": range(
        int(yearly["연도"].min()),
        2026
    )
})

graph_50 = model_50.predict(graph_years[["연도"]])
graph_100 = model_100.predict(graph_years[["연도"]])

fig = go.Figure()


# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "실제 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 50년 회귀선
fig.add_trace(
    go.Scatter(
        x=graph_years["연도"],
        y=graph_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        hovertemplate=(
            "%{x}년<br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 100년 회귀선
fig.add_trace(
    go.Scatter(
        x=graph_years["연도"],
        y=graph_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        hovertemplate=(
            "%{x}년<br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 테스트 기간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="테스트 기간",
    annotation_position="top left"
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

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------------
# 예측 성능 비교
# -----------------------------------
st.subheader("🎯 최근 20년 예측 성능 비교")

result_df = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result_df.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)


# -----------------------------------
# 성능 해석
# -----------------------------------
st.subheader("🔎 결과 비교")

if mae_50 < mae_100:
    mae_result = "최근 50년 모델의 평균적인 예측 오차가 더 작았습니다."
else:
    mae_result = "최근 100년 모델의 평균적인 예측 오차가 더 작았습니다."

if mse_50 < mse_100:
    mse_result = "최근 50년 모델의 큰 오차도 상대적으로 적었습니다."
else:
    mse_result = "최근 100년 모델의 큰 오차도 상대적으로 적었습니다."

if r2_50 > r2_100:
    r2_result = "최근 50년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."
else:
    r2_result = "최근 100년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."

st.write(f"• {mae_result}")
st.write(f"• {mse_result}")
st.write(f"• {r2_result}")


# -----------------------------------
# 테스트 데이터의 실제값과 예측값
# -----------------------------------
st.subheader("📋 최근 20년 실제 기온과 예측값")

comparison = test[
    ["연도", "연평균기온"]
].copy()

comparison["50년 모델 예측"] = pred_50
comparison["100년 모델 예측"] = pred_100

st.dataframe(
    comparison.style.format({
        "연평균기온": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}"
    }),
    use_container_width=True
)
