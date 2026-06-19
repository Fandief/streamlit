from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


def data_path(file_name: str) -> Path:
    gz_path = BASE_DIR / f"{file_name}.gz"
    if gz_path.exists():
        return gz_path
    return BASE_DIR / file_name


def read_csv_file(path: Path, **kwargs) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, compression=compression, **kwargs)


MAIN_CSV = data_path("all_platforms_classified_final.csv")
DETAIL_FILES = {
    "playstore": data_path("playstore_classified_final.csv"),
    "youtube": data_path("youtube_classified_final.csv"),
    "reddit": data_path("reddit_classified_final.csv"),
}

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Neutral": "#2563eb",
    "Negative": "#dc2626",
}
PLATFORM_COLORS = {
    "playstore": "#14b8a6",
    "youtube": "#ef4444",
    "reddit": "#f97316",
}
CHART_TEXT_COLOR = "#111827"
CHART_MUTED_COLOR = "#475569"
CHART_GRID_COLOR = "#e5e7eb"
CHART_BACKGROUND = "#ffffff"
QUARTER_LABELS = {
    1: "Q1 (Jan-Mar)",
    2: "Q2 (Apr-Jun)",
    3: "Q3 (Jul-Sep)",
    4: "Q4 (Okt-Des)",
}

STOPWORDS_ID = {
    "ada",
    "agar",
    "akan",
    "aku",
    "anda",
    "apa",
    "atau",
    "bagaimana",
    "bagi",
    "bahwa",
    "banyak",
    "baru",
    "begitu",
    "belum",
    "bisa",
    "buat",
    "cara",
    "chatgpt",
    "dan",
    "dari",
    "dengan",
    "di",
    "dia",
    "ini",
    "itu",
    "jadi",
    "juga",
    "kalau",
    "kami",
    "karena",
    "ke",
    "lebih",
    "maka",
    "masih",
    "mereka",
    "nya",
    "orang",
    "pada",
    "paling",
    "saya",
    "sebagai",
    "sudah",
    "supaya",
    "tapi",
    "telah",
    "tentang",
    "tidak",
    "untuk",
    "yang",
}


st.set_page_config(
    page_title="Dashboard Sentimen ChatGPT",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --surface: var(--secondary-background-color);
        --line: rgba(128, 128, 128, 0.22);
        --text-soft: var(--text-color);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-soft);
        opacity: 0.82;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    .small-note {
        color: var(--text-soft);
        opacity: 0.82;
        font-size: 0.92rem;
        margin-top: -0.4rem;
    }
    .chart-help {
        color: var(--text-color);
        opacity: 0.86;
        font-size: 0.93rem;
        line-height: 1.45;
        margin: 0.15rem 0 0.55rem 0;
    }
    .chart-help strong {
        color: var(--text-color);
        opacity: 1;
    }
    .chart-help span {
        color: var(--text-color);
        opacity: 0.78;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", ".")


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1f}%"


def sentiment_category(series: pd.Series) -> pd.Series:
    return pd.Categorical(series, categories=SENTIMENT_ORDER, ordered=True)


@st.cache_data(show_spinner="Memuat data gabungan...")
def load_main_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path.name}")

    usecols = ["text", "date", "month_period", "platform", "predicted_sentiment", "confidence_score"]
    df = read_csv_file(path, usecols=usecols)
    df["platform"] = df["platform"].astype("string").str.lower().str.strip()
    df["predicted_sentiment"] = df["predicted_sentiment"].astype("string").str.title().str.strip()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["month"] = pd.to_datetime(df["month_period"].astype("string") + "-01", errors="coerce")
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["quarter_label"] = df["quarter"].map(QUARTER_LABELS)
    df["text"] = df["text"].fillna("").astype("string")
    df = df.dropna(subset=["platform", "predicted_sentiment", "date", "month"])
    df["year"] = df["year"].astype(int)
    df = df[df["predicted_sentiment"].isin(SENTIMENT_ORDER)]
    return df


@st.cache_data(show_spinner="Memuat data detail platform...")
def load_detail_data(platform: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    available_columns = read_csv_file(path, nrows=0).columns.tolist()
    wanted_columns = {
        "playstore": [
            "rating",
            "review_datetime",
            "month_period",
            "predicted_sentiment",
            "confidence_score",
            "platform",
        ],
        "youtube": [
            "viewCount",
            "likeCount",
            "commentCount",
            "commentLikeCount",
            "totalReplyCount",
            "videoTitle",
            "month_period",
            "predicted_sentiment",
            "confidence_score",
            "platform",
        ],
        "reddit": [
            "up_votes",
            "up_vote_ratio",
            "number_of_comments",
            "number_ofreplies",
            "community_name",
            "data_type",
            "month_period",
            "predicted_sentiment",
            "confidence_score",
            "platform",
        ],
    }.get(platform, [])
    usecols = [column for column in wanted_columns if column in available_columns]
    if not usecols:
        return pd.DataFrame()

    df = read_csv_file(path, usecols=usecols)
    if "platform" not in df.columns:
        df["platform"] = platform
    df["platform"] = df["platform"].astype("string").str.lower().str.strip()
    if "predicted_sentiment" in df.columns:
        df["predicted_sentiment"] = df["predicted_sentiment"].astype("string").str.title().str.strip()
    if "confidence_score" in df.columns:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    if "month_period" in df.columns:
        df["month"] = pd.to_datetime(df["month_period"].astype("string") + "-01", errors="coerce")
    for column in [
        "rating",
        "viewCount",
        "likeCount",
        "commentCount",
        "commentLikeCount",
        "totalReplyCount",
        "up_votes",
        "up_vote_ratio",
        "number_of_comments",
        "number_ofreplies",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def apply_filters(
    df: pd.DataFrame,
    platforms: list[str],
    sentiments: list[str],
    years: list[int],
    quarters: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp] | list[pd.Timestamp] | None,
    confidence_range: tuple[float, float],
    search_text: str,
) -> pd.DataFrame:
    filtered = df[
        df["platform"].isin(platforms)
        & df["predicted_sentiment"].isin(sentiments)
        & df["year"].isin(years)
        & df["quarter_label"].isin(quarters)
        & df["confidence_score"].between(confidence_range[0], confidence_range[1], inclusive="both")
    ].copy()

    if date_range is not None and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0], tz="UTC")
        end_date = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered["date"].between(start_date, end_date, inclusive="both")]

    if search_text.strip():
        filtered = filtered[filtered["text"].str.contains(search_text.strip(), case=False, na=False, regex=False)]

    return filtered


def build_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["platform", "predicted_sentiment"], observed=False)
        .size()
        .reset_index(name="total")
    )
    totals = summary.groupby("platform")["total"].transform("sum")
    summary["percentage"] = (summary["total"] / totals * 100).fillna(0)
    summary["predicted_sentiment"] = sentiment_category(summary["predicted_sentiment"])
    return summary.sort_values(["platform", "predicted_sentiment"])


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby(["month", "platform", "predicted_sentiment"], observed=False)
        .size()
        .reset_index(name="total")
        .sort_values("month")
    )
    totals = trend.groupby(["month", "platform"])["total"].transform("sum")
    trend["percentage"] = (trend["total"] / totals * 100).fillna(0)
    return trend


def extract_top_words(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    words: list[str] = []
    for text in df["text"].dropna().astype(str):
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        words.extend(token for token in tokens if token not in STOPWORDS_ID)

    counter = Counter(words).most_common(limit)
    return pd.DataFrame(counter, columns=["word", "count"])


def empty_state(message: str) -> None:
    st.info(message)


def explain_chart(title: str, description: str, reading_hint: str | None = None) -> None:
    hint = f"<br><span>{reading_hint}</span>" if reading_hint else ""
    st.markdown(
        f'<div class="chart-help"><strong>{title}</strong> - {description}{hint}</div>',
        unsafe_allow_html=True,
    )


def apply_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font=dict(color=CHART_TEXT_COLOR),
        legend=dict(font=dict(color=CHART_TEXT_COLOR), title_font=dict(color=CHART_TEXT_COLOR)),
        margin=dict(l=20, r=20),
    )
    fig.update_xaxes(
        color=CHART_TEXT_COLOR,
        title_font=dict(color=CHART_TEXT_COLOR),
        tickfont=dict(color=CHART_TEXT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    fig.update_yaxes(
        color=CHART_TEXT_COLOR,
        title_font=dict(color=CHART_TEXT_COLOR),
        tickfont=dict(color=CHART_TEXT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(apply_chart_style(fig), width="stretch", key=key)


def plot_bar_sentiment(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary,
        x="platform",
        y="total",
        color="predicted_sentiment",
        text="total",
        barmode="group",
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={
            "platform": "Platform",
            "total": "Jumlah data",
            "predicted_sentiment": "Sentimen",
        },
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=430, legend_title_text="", margin=dict(t=30, b=20))
    return fig


def plot_stacked_percentage(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary,
        x="platform",
        y="percentage",
        color="predicted_sentiment",
        text=summary["percentage"].map(lambda value: f"{value:.1f}%"),
        barmode="stack",
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={
            "platform": "Platform",
            "percentage": "Persentase",
            "predicted_sentiment": "Sentimen",
        },
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_traces(textposition="inside", textfont_color="#ffffff")
    fig.update_layout(height=430, legend_title_text="", margin=dict(t=30, b=20))
    return fig


def plot_donut(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentiment")
        .reset_index(name="total")
    )
    fig = px.pie(
        counts,
        names="sentiment",
        values="total",
        hole=0.55,
        color="sentiment",
        category_orders={"sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextfont_color="#ffffff")
    fig.update_layout(height=430, showlegend=False, margin=dict(t=25, b=15))
    return fig


def plot_monthly_trend(trend: pd.DataFrame, mode: str) -> go.Figure:
    y_column = "percentage" if mode == "Persentase" else "total"
    fig = px.line(
        trend,
        x="month",
        y=y_column,
        color="predicted_sentiment",
        line_dash="platform",
        markers=True,
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={
            "month": "Bulan",
            y_column: "Persentase" if y_column == "percentage" else "Jumlah data",
            "predicted_sentiment": "Sentimen",
            "platform": "Platform",
        },
    )
    if y_column == "percentage":
        fig.update_yaxes(ticksuffix="%")
    fig.update_layout(height=520, legend_title_text="", margin=dict(t=20, b=20))
    return fig


def plot_heatmap(summary: pd.DataFrame, value_column: str) -> go.Figure:
    pivot = summary.pivot_table(
        index="platform",
        columns="predicted_sentiment",
        values=value_column,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reindex(columns=SENTIMENT_ORDER)
    text = pivot.map(lambda value: f"{value:.1f}%" if value_column == "percentage" else format_number(value))
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            text=text.values,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#f8fafc"],
                [0.35, "#dbeafe"],
                [0.7, "#bfdbfe"],
                [1.0, "#93c5fd"],
            ],
            textfont=dict(color=CHART_TEXT_COLOR),
            hovertemplate="Platform: %{y}<br>Sentimen: %{x}<br>Nilai: %{text}<extra></extra>",
        )
    )
    fig.update_layout(height=360, margin=dict(t=20, b=20))
    return fig


def plot_confidence_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="predicted_sentiment",
        y="confidence_score",
        color="predicted_sentiment",
        points=False,
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={"predicted_sentiment": "Sentimen", "confidence_score": "Confidence score"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=400, showlegend=False, margin=dict(t=25, b=15))
    return fig


def plot_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="confidence_score",
        color="predicted_sentiment",
        nbins=30,
        barmode="overlay",
        opacity=0.74,
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={"confidence_score": "Confidence score", "count": "Jumlah data"},
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(height=400, legend_title_text="", margin=dict(t=25, b=15))
    return fig


def plot_top_words(words_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        words_df.sort_values("count"),
        x="count",
        y="word",
        orientation="h",
        color="count",
        color_continuous_scale=["#d1fae5", "#14b8a6", "#0f766e"],
        labels={"count": "Frekuensi", "word": "Kata"},
    )
    fig.update_layout(height=620, coloraxis_showscale=False, margin=dict(t=20, b=20, l=10))
    return fig


def render_sidebar(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[int], list[str], tuple | None, tuple[float, float], str, int]:
    st.sidebar.header("Filter")

    platforms = sorted(df["platform"].dropna().unique().tolist())
    selected_platforms = st.sidebar.multiselect("Platform", platforms, default=platforms)

    sentiments = [sentiment for sentiment in SENTIMENT_ORDER if sentiment in df["predicted_sentiment"].unique()]
    selected_sentiments = st.sidebar.multiselect("Sentimen", sentiments, default=sentiments)

    year_options = sorted(df["year"].dropna().astype(int).unique().tolist())
    selected_years = st.sidebar.multiselect("Tahun", year_options, default=year_options)

    quarter_source = df[df["year"].isin(selected_years)] if selected_years else df.iloc[0:0]
    quarter_options = [
        label
        for _, label in sorted(QUARTER_LABELS.items())
        if label in set(quarter_source["quarter_label"].dropna())
    ]
    selected_quarters = st.sidebar.multiselect("Kuartal", quarter_options, default=quarter_options)

    selected_dates = None
    use_date_range = st.sidebar.checkbox("Gunakan rentang tanggal detail", value=False)
    if use_date_range:
        date_source = quarter_source[quarter_source["quarter_label"].isin(selected_quarters)]
        if date_source.empty:
            date_source = df
        min_date = date_source["date"].min().date()
        max_date = date_source["date"].max().date()
        selected_dates = st.sidebar.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
        )

    confidence_min = float(df["confidence_score"].min())
    confidence_max = float(df["confidence_score"].max())
    selected_confidence = st.sidebar.slider(
        "Confidence score",
        min_value=0.0,
        max_value=1.0,
        value=(max(0.0, round(confidence_min, 2)), min(1.0, round(confidence_max, 2))),
        step=0.01,
    )

    search_text = st.sidebar.text_input("Cari teks", placeholder="contoh: lambat, akurat, error")
    top_n = st.sidebar.slider("Jumlah kata teratas", min_value=10, max_value=50, value=25, step=5)

    st.sidebar.divider()
    st.sidebar.caption("Data utama: all_platforms_classified_final.csv")
    return selected_platforms, selected_sentiments, selected_years, selected_quarters, selected_dates, selected_confidence, search_text, top_n


def render_kpis(df: pd.DataFrame) -> None:
    total_rows = len(df)
    sentiment_counts = df["predicted_sentiment"].value_counts()
    positive_pct = sentiment_counts.get("Positive", 0) / total_rows * 100 if total_rows else 0
    negative_pct = sentiment_counts.get("Negative", 0) / total_rows * 100 if total_rows else 0
    neutral_pct = sentiment_counts.get("Neutral", 0) / total_rows * 100 if total_rows else 0
    avg_confidence = df["confidence_score"].mean()
    dominant_platform = df["platform"].value_counts().idxmax() if total_rows else "-"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total data", format_number(total_rows))
    col2.metric("Positive", pct(positive_pct))
    col3.metric("Neutral", pct(neutral_pct))
    col4.metric("Negative", pct(negative_pct))
    col5.metric("Rata-rata confidence", f"{avg_confidence:.3f}" if pd.notna(avg_confidence) else "-")
    col6.metric("Platform dominan", str(dominant_platform).title())


def render_overview_tab(df: pd.DataFrame) -> None:
    summary = build_sentiment_summary(df)
    trend = build_monthly_trend(df)

    st.subheader("Ringkasan Sentimen")
    col1, col2 = st.columns([1.35, 1])
    with col1:
        explain_chart(
            "Jumlah sentimen per platform",
            "grafik batang ini membandingkan jumlah komentar/review Positive, Neutral, dan Negative pada setiap platform.",
            "Semakin tinggi batangnya, semakin banyak data dengan sentimen tersebut pada platform itu.",
        )
        show_chart(plot_bar_sentiment(summary), "overview_sentiment_bar")
    with col2:
        explain_chart(
            "Proporsi sentimen keseluruhan",
            "diagram donat ini menunjukkan pembagian sentimen dari seluruh data yang sedang aktif setelah filter.",
            "Bagian terbesar menunjukkan sentimen yang paling dominan.",
        )
        show_chart(plot_donut(df), "overview_sentiment_donut")

    col3, col4 = st.columns([1.2, 1])
    with col3:
        st.subheader("Komposisi Persentase per Platform")
        explain_chart(
            "Persentase sentimen per platform",
            "grafik bertumpuk ini membuat setiap platform menjadi 100%, lalu membagi porsinya ke Positive, Neutral, dan Negative.",
            "Gunakan grafik ini untuk membandingkan komposisi sentimen, bukan jumlah data mentah.",
        )
        show_chart(plot_stacked_percentage(summary), "overview_platform_percentage")
    with col4:
        st.subheader("Heatmap Sentimen")
        heatmap_metric = st.radio("Nilai heatmap", ["percentage", "total"], horizontal=True, label_visibility="collapsed")
        explain_chart(
            "Peta intensitas sentimen",
            "heatmap ini memberi warna lebih kuat pada kombinasi platform dan sentimen yang nilainya lebih besar.",
            "Mode percentage cocok untuk proporsi; mode total cocok untuk volume data.",
        )
        show_chart(plot_heatmap(summary, heatmap_metric), f"overview_heatmap_{heatmap_metric}")

    st.subheader("Tren Bulanan")
    trend_mode = st.radio("Mode tren", ["Jumlah", "Persentase"], horizontal=True)
    explain_chart(
        "Perubahan sentimen dari waktu ke waktu",
        "grafik garis ini menunjukkan naik-turunnya sentimen tiap bulan untuk masing-masing platform.",
        "Jika garis Negative naik pada bulan tertentu, berarti keluhan atau komentar negatif meningkat pada periode itu.",
    )
    show_chart(plot_monthly_trend(trend, trend_mode), f"overview_monthly_trend_{trend_mode}")


def render_platform_tab(df: pd.DataFrame) -> None:
    st.subheader("Perbandingan Platform")
    summary = build_sentiment_summary(df)
    platform_totals = df["platform"].value_counts().rename_axis("platform").reset_index(name="total")

    col1, col2 = st.columns([1, 1])
    with col1:
        explain_chart(
            "Jumlah data per platform",
            "grafik ini menunjukkan berapa banyak data yang tersedia dari Play Store, YouTube, dan Reddit.",
            "Platform dengan data paling banyak akan lebih dominan dalam analisis gabungan.",
        )
        fig_total = px.bar(
            platform_totals.sort_values("total"),
            x="total",
            y="platform",
            orientation="h",
            color="platform",
            color_discrete_map=PLATFORM_COLORS,
            labels={"total": "Jumlah data", "platform": "Platform"},
        )
        fig_total.update_layout(height=360, showlegend=False, margin=dict(t=20, b=20))
        show_chart(fig_total, "platform_total_bar")
    with col2:
        explain_chart(
            "Komposisi sentimen antar platform",
            "heatmap ini membantu melihat platform mana yang cenderung memiliki porsi Positive, Neutral, atau Negative lebih tinggi.",
            "Warna yang lebih kuat berarti persentasenya lebih besar.",
        )
        show_chart(plot_heatmap(summary, "percentage"), "platform_percentage_heatmap")

    st.subheader("Distribusi Confidence per Platform")
    explain_chart(
        "Sebaran confidence model per platform",
        "violin plot ini menunjukkan seberapa yakin model saat mengklasifikasikan data di setiap platform.",
        "Bentuk yang melebar berarti banyak data berada di rentang confidence tersebut; kotak di dalamnya menunjukkan median dan sebaran utama.",
    )
    fig = px.violin(
        df,
        x="platform",
        y="confidence_score",
        color="platform",
        box=True,
        points=False,
        color_discrete_map=PLATFORM_COLORS,
        labels={"platform": "Platform", "confidence_score": "Confidence score"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=430, showlegend=False, margin=dict(t=25, b=15))
    show_chart(fig, "platform_confidence_violin")


def render_confidence_tab(df: pd.DataFrame) -> None:
    st.subheader("Kualitas Prediksi Model")
    col1, col2 = st.columns(2)
    with col1:
        explain_chart(
            "Boxplot confidence per sentimen",
            "boxplot ini merangkum nilai confidence model untuk setiap kelas sentimen.",
            "Garis tengah menunjukkan median; rentang kotak dan titik ekstrem membantu melihat kestabilan prediksi.",
        )
        show_chart(plot_confidence_box(df), "confidence_box")
    with col2:
        explain_chart(
            "Distribusi confidence",
            "histogram ini menunjukkan seberapa sering model menghasilkan confidence pada rentang tertentu.",
            "Jika banyak data mendekati 1.0, model umumnya sangat yakin terhadap prediksinya.",
        )
        show_chart(plot_confidence_histogram(df), "confidence_histogram")

    low_confidence_threshold = st.slider("Ambang confidence rendah", 0.0, 1.0, 0.80, 0.01)
    low_confidence = df[df["confidence_score"] < low_confidence_threshold].copy()
    st.caption(f"{format_number(len(low_confidence))} data berada di bawah ambang confidence {low_confidence_threshold:.2f}.")
    if not low_confidence.empty:
        columns = ["platform", "date", "predicted_sentiment", "confidence_score", "text"]
        st.dataframe(
            low_confidence[columns].sort_values("confidence_score").head(200),
            width="stretch",
            hide_index=True,
        )


def render_text_tab(df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Kata Paling Sering Muncul")
    explain_chart(
        "Frekuensi kata dominan",
        "grafik ini menampilkan kata yang paling sering muncul pada teks sesuai filter platform dan sentimen.",
        "Kata dengan frekuensi tinggi dapat menjadi petunjuk topik yang sering dibahas pengguna.",
    )

    col1, col2 = st.columns([1, 1])
    selected_platform = col1.selectbox("Platform untuk kata", ["Semua"] + sorted(df["platform"].unique().tolist()))
    selected_sentiment = col2.selectbox("Sentimen untuk kata", ["Semua"] + SENTIMENT_ORDER)

    text_df = df.copy()
    if selected_platform != "Semua":
        text_df = text_df[text_df["platform"] == selected_platform]
    if selected_sentiment != "Semua":
        text_df = text_df[text_df["predicted_sentiment"] == selected_sentiment]

    words_df = extract_top_words(text_df, top_n)
    if words_df.empty:
        empty_state("Tidak ada kata yang cukup untuk filter ini.")
        return

    show_chart(plot_top_words(words_df), "text_top_words")

    st.subheader("Contoh Teks")
    explain_chart(
        "Contoh data teks",
        "tabel ini menampilkan contoh komentar atau review terbaru yang sesuai filter aktif.",
        "Gunakan tabel ini untuk membaca konteks asli dari pola yang muncul di grafik.",
    )
    sample_columns = ["platform", "date", "predicted_sentiment", "confidence_score", "text"]
    st.dataframe(
        text_df[sample_columns].sort_values("date", ascending=False).head(300),
        width="stretch",
        hide_index=True,
    )


def render_detail_tab(filtered_main: pd.DataFrame) -> None:
    st.subheader("Insight Tambahan per Platform")
    explain_chart(
        "Analisis tambahan dari file detail",
        "bagian ini memakai CSV per-platform untuk melihat variabel yang tidak ada di file gabungan utama.",
        "Contohnya rating Play Store, engagement YouTube, dan komunitas Reddit.",
    )
    active_platforms = set(filtered_main["platform"].unique())

    detail_tabs = st.tabs(["Play Store", "YouTube", "Reddit"])

    with detail_tabs[0]:
        if "playstore" not in active_platforms:
            empty_state("Play Store tidak termasuk dalam filter saat ini.")
        else:
            playstore = load_detail_data("playstore", DETAIL_FILES["playstore"])
            playstore = filter_detail_by_main(playstore, filtered_main)
            render_playstore_detail(playstore)

    with detail_tabs[1]:
        if "youtube" not in active_platforms:
            empty_state("YouTube tidak termasuk dalam filter saat ini.")
        else:
            youtube = load_detail_data("youtube", DETAIL_FILES["youtube"])
            youtube = filter_detail_by_main(youtube, filtered_main)
            render_youtube_detail(youtube)

    with detail_tabs[2]:
        if "reddit" not in active_platforms:
            empty_state("Reddit tidak termasuk dalam filter saat ini.")
        else:
            reddit = load_detail_data("reddit", DETAIL_FILES["reddit"])
            reddit = filter_detail_by_main(reddit, filtered_main)
            render_reddit_detail(reddit)


def filter_detail_by_main(detail_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty or "month" not in detail_df.columns:
        return detail_df

    months = main_df["month"].dropna().unique()
    sentiments = main_df["predicted_sentiment"].dropna().unique()
    detail_df = detail_df[detail_df["month"].isin(months)]
    if "predicted_sentiment" in detail_df.columns:
        detail_df = detail_df[detail_df["predicted_sentiment"].isin(sentiments)]
    return detail_df


def render_playstore_detail(df: pd.DataFrame) -> None:
    if df.empty or "rating" not in df.columns:
        empty_state("Data detail Play Store belum tersedia untuk filter ini.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        explain_chart(
            "Rating Play Store per sentimen",
            "grafik ini membandingkan jumlah review pada setiap rating untuk masing-masing sentimen.",
            "Rating rendah yang didominasi Negative menunjukkan keluhan pengguna yang konsisten dengan nilai rating.",
        )
        rating_summary = (
            df.dropna(subset=["rating"])
            .groupby(["rating", "predicted_sentiment"], observed=False)
            .size()
            .reset_index(name="total")
        )
        fig = px.bar(
            rating_summary,
            x="rating",
            y="total",
            color="predicted_sentiment",
            barmode="group",
            category_orders={"predicted_sentiment": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS,
            labels={"rating": "Rating", "total": "Jumlah review", "predicted_sentiment": "Sentimen"},
        )
        fig.update_layout(height=420, legend_title_text="", margin=dict(t=25, b=15))
        show_chart(fig, "playstore_rating_sentiment")
    with col2:
        explain_chart(
            "Rata-rata rating per sentimen",
            "grafik ini menunjukkan rata-rata rating Play Store untuk kelas Positive, Neutral, dan Negative.",
            "Jika Positive memiliki rating rata-rata lebih tinggi, hasil sentimen selaras dengan penilaian bintang.",
        )
        rating_avg = (
            df.dropna(subset=["rating"])
            .groupby("predicted_sentiment", observed=False)["rating"]
            .mean()
            .reindex(SENTIMENT_ORDER)
            .reset_index()
        )
        fig = px.bar(
            rating_avg,
            x="predicted_sentiment",
            y="rating",
            color="predicted_sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            labels={"predicted_sentiment": "Sentimen", "rating": "Rata-rata rating"},
        )
        fig.update_yaxes(range=[0, 5])
        fig.update_layout(height=420, showlegend=False, margin=dict(t=25, b=15))
        show_chart(fig, "playstore_rating_average")


def render_youtube_detail(df: pd.DataFrame) -> None:
    if df.empty:
        empty_state("Data detail YouTube belum tersedia untuk filter ini.")
        return

    numeric_columns = [column for column in ["viewCount", "likeCount", "commentCount", "commentLikeCount", "totalReplyCount"] if column in df.columns]
    if not numeric_columns:
        empty_state("Kolom engagement YouTube tidak ditemukan.")
        return

    engagement = (
        df.groupby("predicted_sentiment", observed=False)[numeric_columns]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .reset_index()
        .melt(id_vars="predicted_sentiment", var_name="metric", value_name="average")
    )
    explain_chart(
        "Rata-rata engagement YouTube per sentimen",
        "grafik ini membandingkan metrik seperti view, like, komentar, like komentar, dan reply berdasarkan sentimen.",
        "Nilai tinggi pada sentimen tertentu menunjukkan jenis opini itu muncul pada konten dengan engagement lebih besar.",
    )
    fig = px.bar(
        engagement,
        x="metric",
        y="average",
        color="predicted_sentiment",
        barmode="group",
        category_orders={"predicted_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        labels={"metric": "Metrik", "average": "Rata-rata", "predicted_sentiment": "Sentimen"},
    )
    fig.update_layout(height=430, legend_title_text="", margin=dict(t=25, b=15))
    show_chart(fig, "youtube_engagement")

    if "videoTitle" in df.columns and "commentCount" in df.columns:
        top_videos = (
            df.dropna(subset=["videoTitle"])
            .groupby("videoTitle", observed=False)
            .agg(total_komentar=("videoTitle", "size"), avg_confidence=("confidence_score", "mean"))
            .sort_values("total_komentar", ascending=False)
            .head(15)
            .reset_index()
        )
        explain_chart(
            "Video dengan komentar terbanyak",
            "tabel ini menampilkan judul video yang paling banyak menyumbang data komentar dalam filter aktif.",
            "Gunakan tabel ini untuk mengetahui sumber percakapan yang paling dominan.",
        )
        st.dataframe(top_videos, width="stretch", hide_index=True)


def render_reddit_detail(df: pd.DataFrame) -> None:
    if df.empty:
        empty_state("Data detail Reddit belum tersedia untuk filter ini.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        if "community_name" in df.columns:
            explain_chart(
                "Sentimen per komunitas Reddit",
                "grafik ini menunjukkan komunitas Reddit mana yang paling banyak memuat data dan bagaimana komposisi sentimennya.",
                "Komunitas dengan batang panjang adalah sumber diskusi yang paling aktif pada filter saat ini.",
            )
            community = (
                df.dropna(subset=["community_name"])
                .groupby(["community_name", "predicted_sentiment"], observed=False)
                .size()
                .reset_index(name="total")
            )
            top_communities = community.groupby("community_name")["total"].sum().nlargest(10).index
            community = community[community["community_name"].isin(top_communities)]
            fig = px.bar(
                community,
                x="total",
                y="community_name",
                color="predicted_sentiment",
                orientation="h",
                barmode="stack",
                category_orders={"predicted_sentiment": SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_COLORS,
                labels={"total": "Jumlah", "community_name": "Community", "predicted_sentiment": "Sentimen"},
            )
            fig.update_layout(height=430, legend_title_text="", margin=dict(t=25, b=15))
            show_chart(fig, "reddit_community_sentiment")
    with col2:
        numeric_columns = [column for column in ["up_votes", "up_vote_ratio", "number_of_comments", "number_ofreplies"] if column in df.columns]
        if numeric_columns:
            explain_chart(
                "Rata-rata metrik Reddit per sentimen",
                "grafik ini membandingkan upvote, rasio upvote, jumlah komentar, dan reply berdasarkan sentimen.",
                "Ini membantu melihat apakah sentimen tertentu lebih sering muncul pada diskusi yang mendapat respons lebih tinggi.",
            )
            reddit_metric = (
                df.groupby("predicted_sentiment", observed=False)[numeric_columns]
                .mean()
                .reindex(SENTIMENT_ORDER)
                .reset_index()
                .melt(id_vars="predicted_sentiment", var_name="metric", value_name="average")
            )
            fig = px.bar(
                reddit_metric,
                x="metric",
                y="average",
                color="predicted_sentiment",
                barmode="group",
                category_orders={"predicted_sentiment": SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_COLORS,
                labels={"metric": "Metrik", "average": "Rata-rata", "predicted_sentiment": "Sentimen"},
            )
            fig.update_layout(height=430, legend_title_text="", margin=dict(t=25, b=15))
            show_chart(fig, "reddit_metrics")


def render_download_tab(df: pd.DataFrame) -> None:
    st.subheader("Data Hasil Filter")
    explain_chart(
        "Tabel data setelah filter",
        "tabel ini berisi data yang sudah mengikuti semua filter di sidebar.",
        "Data ini dapat diunduh untuk pengecekan manual atau analisis lanjutan.",
    )
    st.caption(f"Menampilkan {format_number(len(df))} baris sesuai filter aktif.")

    columns = ["platform", "date", "month_period", "predicted_sentiment", "confidence_score", "text"]
    st.dataframe(df[columns].sort_values("date", ascending=False).head(1000), width="stretch", hide_index=True)

    csv = df[columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Unduh data filter sebagai CSV",
        data=csv,
        file_name="filtered_sentiment_chatgpt.csv",
        mime="text/csv",
        width="stretch",
    )


def main() -> None:
    st.title("Dashboard Analisis Sentimen ChatGPT")
    st.markdown(
        '<div class="small-note">Visualisasi sentimen dari Play Store, YouTube, dan Reddit.</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_main_data(MAIN_CSV)
    except Exception as exc:
        st.error(f"Gagal memuat data: {exc}")
        st.stop()

    (
        selected_platforms,
        selected_sentiments,
        selected_years,
        selected_quarters,
        selected_dates,
        selected_confidence,
        search_text,
        top_n,
    ) = render_sidebar(df)

    if not selected_platforms or not selected_sentiments or not selected_years or not selected_quarters:
        empty_state("Pilih minimal satu platform, satu sentimen, satu tahun, dan satu kuartal.")
        st.stop()

    filtered_df = apply_filters(
        df,
        selected_platforms,
        selected_sentiments,
        selected_years,
        selected_quarters,
        selected_dates,
        selected_confidence,
        search_text,
    )

    if filtered_df.empty:
        empty_state("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    render_kpis(filtered_df)
    st.divider()

    tab_overview, tab_platform, tab_confidence, tab_text, tab_detail, tab_data = st.tabs(
        [
            "Overview",
            "Platform",
            "Confidence",
            "Teks",
            "Detail Platform",
            "Data",
        ]
    )

    with tab_overview:
        render_overview_tab(filtered_df)
    with tab_platform:
        render_platform_tab(filtered_df)
    with tab_confidence:
        render_confidence_tab(filtered_df)
    with tab_text:
        render_text_tab(filtered_df, top_n)
    with tab_detail:
        render_detail_tab(filtered_df)
    with tab_data:
        render_download_tab(filtered_df)


if __name__ == "__main__":
    main()
