import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


font_path = "assets/font/malgun.ttf"

if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
else:
    st.warning("⚠️ 한글 폰트 파일을 찾을 수 없습니다.")


#================================================================

st.set_page_config(layout="wide")

st.title("K-pop Demon Hunters 팬덤 분석")
st.markdown(
    "<h3>C321087 홍석현</h3>",
    unsafe_allow_html=True
)

# 데이터 로드
df = pd.read_csv("./data/naver_news.csv")

df['pubDate'] = pd.to_datetime(df['pubDate'])

st.subheader("데이터 미리보기")
st.dataframe(df.head())

#================================================================

st.sidebar.header("분석 옵션")

# 날짜 범위 선택
date_range = st.sidebar.date_input(
    "분석 기간 선택",
    [df['pubDate'].min(), df['pubDate'].max()]
)

# 키워드 선택
keyword_options = ['케이팝', 'K팝', '넷플릭스', '애니메이션', '영화', '글로벌', '데몬']
selected_keywords = st.sidebar.multiselect(
    "분석 키워드 선택",
    keyword_options,
    default=keyword_options
)

# 네트워크 엣지 기준
min_edge_weight = st.sidebar.slider(
    "네트워크 최소 연결 빈도",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)

# 워드클라우드 단어 수
max_words = st.sidebar.slider(
    "워드클라우드 단어 수",
    min_value=20,
    max_value=100,
    value=50,
    step=10
)

# 상위 키워드 개수
top_n = st.sidebar.slider(
    "상위 키워드 개수",
    min_value=5,
    max_value=20,
    value=10
)

# 시간 추이 분석 키워드 선택
keywords_of_interest = st.sidebar.multiselect(
    "시간 추이 분석 키워드",
    ["넷플릭스", "케이팝", "애니메이션", "영화", "글로벌"],
    default=["넷플릭스", "케이팝", "애니메이션"]
)

# 설명 표시 여부
show_text = st.sidebar.checkbox("해석 설명 표시", value=True)

# 위젯 값 적용 
df_filtered = df[
    (df['pubDate'].dt.date >= date_range[0]) &
    (df['pubDate'].dt.date <= date_range[1])
]

#================================================================

import altair as alt

st.divider()
st.subheader("📈 검색 관심도 변화 (Altair)")

# 날짜 컬럼을 날짜 단위로 변환 (필터된 데이터 기준)
df_filtered["date"] = pd.to_datetime(df_filtered["pubDate"]).dt.date

# 날짜별 기사 수 집계
trend_df = (
    df_filtered.groupby("date")
               .size()
               .reset_index(name="count")
)

# Altair 시계열 그래프
line_chart = (
    alt.Chart(trend_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("count:Q", title="기사 수"),
        tooltip=["date:T", "count:Q"]
    )
    .properties(height=300)
)

st.altair_chart(line_chart, use_container_width=True)

st.markdown("""
**설명&해석**  
*K-pop Demon Hunters*와 관련된 기사 노출 빈도가 시간에 따라 어떻게 변화했는지를 확인하기 위해 제작하였다.  
이를 통해 특정 시점에 관심이 급증한 계기가 있었는지, 또 관심이 일시적인 이슈인지 지속적인 팬덤 형성으로 이어졌는지를 파악하고자 하였다.
그래프를 살펴보면 검색 기사 수는 특정 시점을 기준으로 급격히 증가하였다가 다시 감소하는 양상을 보인다.  
이는 *K-pop Demon Hunters*가 특정 이벤트를 계기로 단기간 강한 미디어 주목을 받았음을 의미하며, 팬덤 형성의 초기 확산 국면을 확인할 수 있다.
""")

#================================================================

import seaborn as sns
from collections import Counter

st.divider()
st.subheader("📊 주요 키워드 빈도 비교 (Seaborn)")

# df_filtered 사용
text_series = (
    df_filtered["title"].astype(str) + " " +
    df_filtered["description"].astype(str)
)

# 강의 범위: 단순 토큰화 + 불용어 제거
stopwords = ["관련", "통해", "대한", "기자", "이번", "있다", "한다"]
words = " ".join(text_series).split()
words = [w for w in words if len(w) > 1 and w not in stopwords]

word_counts = Counter(words)
top_words = word_counts.most_common(top_n)

keyword_df = pd.DataFrame(top_words, columns=["keyword", "count"])

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=keyword_df,
    x="count",
    y="keyword",
    ax=ax
)

ax.set_xlabel("빈도")
ax.set_ylabel("키워드")

for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)

for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)

st.pyplot(fig)


st.markdown("""
**설명&해석**
이 그래프는 기사 제목과 본문에서 자주 등장하는 핵심 키워드를 추출하여, *K-pop Demon Hunters*가 어떤 요소들과 함께 언급되고 있는지를 파악하기 위해 제작하였다.    
상위 키워드 분석 결과, ‘데몬’, ‘케이팝’, ‘애니메이션’, ‘넷플릭스’와 같은 단어가 다른 키워드들에 비해 높은 빈도로 등장하였다.  
이는 *K-pop Demon Hunters*가 단순한 음악 콘텐츠를 넘어, 애니메이션·OTT 플랫폼가 결합된 복합 콘텐츠로 인식되고 있음을 보여준다.
""")

#================================================================

import plotly.express as px

st.divider()
st.subheader("📈 주요 키워드의 시간별 언급 추이 (Plotly)")

df_filtered["date"] = pd.to_datetime(df_filtered["pubDate"]).dt.date

plotly_df = []

for kw in keywords_of_interest:
    temp = df_filtered[
        df_filtered["title"].str.contains(kw, na=False) |
        df_filtered["description"].str.contains(kw, na=False)
    ]
    count_df = temp.groupby("date").size().reset_index(name="count")
    count_df["keyword"] = kw
    plotly_df.append(count_df)

if plotly_df:
    plotly_df = pd.concat(plotly_df)

    fig = px.line(
        plotly_df,
        x="date",
        y="count",
        color="keyword",
        markers=True
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("선택된 키워드에 해당하는 데이터가 없습니다.")


st.markdown("""
**설명&해석**  
위 그래프는 주요 키워드별로 기사 언급 빈도가 시간에 따라 어떻게 변화하는지를 비교하기 위해 제작하였다. 
시간별 키워드 언급 추이를 살펴보면, ‘케이팝’, ‘넷플릭스’, ‘애니메이션’ 키워드가 동일한 시점에 동시에 증가하는 구간이 관찰된다.  
이는 콘텐츠 공개 및 관련 이슈를 계기로 음악(K-pop), 플랫폼(Netflix), 영상 콘텐츠(애니메이션)가 결합된 형태로 관심이 확산되었음을 의미한다.  
""")

#================================================================

from collections import Counter
from wordcloud import WordCloud
import re

st.divider()
st.subheader("☁️ 팬덤 담론 WordCloud (불용어 기반)")

stopwords = [
    "기자", "뉴스", "보도", "관련", "이번", "통해",
    "대한", "이날", "등", "수", "것", "있다", "없다", "하다"
]

# title + description 결합
text_series = df["title"].astype(str) + " " + df["description"].astype(str)

# 텍스트 정제
clean_text = []
for text in text_series:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^가-힣\s]", "", text)
    clean_text.append(text)

# 단어 분리
words = " ".join(clean_text).split()

# 불용어 제거 + 길이 필터
words = [w for w in words if w not in stopwords and len(w) > 1]

word_freq = Counter(words)

wc = WordCloud(
    font_path="assets/font/malgun.ttf",
    background_color="white",
    width=500,
    height=250,
    max_words=100
).generate_from_frequencies(word_freq)

fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)

st.markdown("""
**설명&해석**
WordCloud는 기사 텍스트 전반에서 반복적으로 등장하는 단어를 시각적으로 확인하기 위해 사용하였다.  
형태소 분석과 불용어 제거를 통해 의미 없는 단어를 제외하고, 팬덤 담론에서 실제로 중요한 키워드가 무엇인지 직관적으로 파악하고자 하였다.  
WordCloud 결과, ‘케이팝’, ‘데몬’, ‘헌터스’, ‘애니메이션’, ‘넷플릭스’와 같은 키워드가 상대적으로 크게 나타난다.  
이는 *K-pop Demon Hunters*가 K-pop 음악 요소와 애니메이션 형식, 그리고 넷플릭스라는 플랫폼을 중심으로 주목받고 있음을 의미한다.  
""")

#================================================================

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import itertools

st.subheader("키워드 네트워크 분석")

# 위젯 연동
keywords = selected_keywords

# 문서별 키워드 추출 (필터된 데이터 기준)
docs = []
for text in df_filtered['title'] + df_filtered['description']:
    found = [kw for kw in keywords if kw in text]
    if len(found) >= 2:
        docs.append(found)

# 키워드 쌍 빈도 계산
counter = Counter()
for d in docs:
    counter.update(itertools.combinations(sorted(d), 2))

# 엣지 필터링 (slider 연동)
filtered_edges = {
    edge: w for edge, w in counter.items()
    if w >= min_edge_weight
}

# 상위 15개 엣지 제한
filtered_edges = dict(
    sorted(filtered_edges.items(), key=lambda x: x[1], reverse=True)[:15]
)

if len(filtered_edges) == 0:
    st.info("선택한 조건에서 네트워크를 구성할 수 있는 데이터가 없습니다.")
else:
    G = nx.Graph()
    G.add_weighted_edges_from(
        [(a, b, w) for (a, b), w in filtered_edges.items()]
    )

    fig = plt.figure(figsize=(4, 4))

    pos = nx.spring_layout(G, k=1.3, iterations=50, seed=42)

    node_sizes = [G.degree(n) * 250 for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 0.03 for u, v in G.edges()]

    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_size=node_sizes,
        width=edge_widths,
        font_family=font_prop.get_name(),
        font_size=8,
        node_color='skyblue',
        edge_color='gray',
        alpha=0.5
    )

    plt.title("키워드 네트워크", fontsize=10)
    plt.axis('off')

    st.pyplot(fig, use_container_width=False)
    plt.close()


st.markdown("""
**설명&해석**
노드의 크기는 키워드의 연결 정도를, 엣지의 두께는 함께 언급된 빈도를 의미하며, 이를 통해 팬덤 담론에서 중심적인 키워드와 구조를 파악하고자 한다.  
네트워크 그래프를 보면 ‘케이팝’과 ‘데몬’이 중심 노드로 위치하며, ‘넷플릭스’, ‘애니메이션’, ‘영화’가 이들과 강하게 연결되어 있다.
K-pop Demon Hunters는 K팝 IP를 기반으로 넷플릭스와 영상 콘텐츠가 결합된 구조를 통해 글로벌 팬덤을 형성하고 있다는 것을 알 수 있다. 
""")

