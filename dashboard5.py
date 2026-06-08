import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title="Cars Dashboard", layout="wide", page_icon="🚗")

PALETTE = [
    "#00d4ff", "#00b8e6", "#009dcc", "#0081b3",
    "#0066ff", "#1a6aff", "#3385ff", "#4d9fff",
    "#66b8ff", "#80d0ff", "#99e0ff", "#b3f0ff",
    "#cce8ff", "#d9f0ff", "#e6f8ff",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ─── Global Background ─── */
.stApp {
    background: #070b14;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0, 102, 255, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0, 212, 255, 0.06) 0%, transparent 60%);
}

.main .block-container {
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #070b14 100%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.1);
}

[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #0066ff, #00d4ff, transparent);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8899bb !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4ff !important;
}

/* ─── Page Title ─── */
.page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 60%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.page-subtitle {
    color: #4a5a7a;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* ─── Divider ─── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(0,102,255,0.2), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ─── KPI Cards ─── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.75rem;
}
.kpi-card {
    background: #0a0e1a;
    border: 1px solid rgba(0,212,255,0.09);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #0066ff, #00d4ff);
    border-radius: 12px 0 0 12px;
}
.kpi-icon-box {
    width: 38px; height: 38px;
    min-width: 38px;
    border-radius: 10px;
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.kpi-text { flex: 1; min-width: 0; }
.kpi-label {
    font-size: 0.6rem;
    color: #304560;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.2rem;
    white-space: nowrap;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #e8f4ff;
    line-height: 1;
    white-space: nowrap;
}
.kpi-value sup {
    font-size: 0.6rem;
    font-weight: 500;
    color: #00d4ff;
    margin-left: 2px;
    vertical-align: super;
}

/* ─── Section Heading ─── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}
.section-heading-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, rgba(0,102,255,0.2), rgba(0,212,255,0.1));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-heading-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #ddeeff;
}
.section-heading-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.15), transparent);
}

/* ─── Chart Container ─── */
.chart-card {
    background: rgba(13,17,32,0.7);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem 1rem 0.5rem 1rem;
    margin-bottom: 0.5rem;
}
.chart-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
    padding-left: 0.25rem;
}

/* ─── Predictor Page ─── */
.pred-section {
    background: rgba(13,17,32,0.7);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.pred-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00d4ff;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.pred-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.25), transparent);
}

/* ─── Price Result Card ─── */
.price-result-wrap {
    position: relative;
    margin-top: 1.5rem;
    border-radius: 20px;
    overflow: hidden;
}
.price-result-glow {
    position: absolute;
    inset: -1px;
    background: linear-gradient(135deg, #0066ff, #00d4ff, #0066ff);
    border-radius: 20px;
    opacity: 0.5;
    z-index: 0;
    animation: glowPulse 3s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { opacity: 0.35; }
    50% { opacity: 0.65; }
}
.price-result-inner {
    position: relative;
    z-index: 1;
    background: linear-gradient(135deg, #0d1425 0%, #07101f 100%);
    border-radius: 19px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.price-result-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #4a6880;
    margin-bottom: 0.75rem;
}
.price-result-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 1rem;
}
.price-result-meta {
    display: inline-flex;
    gap: 1rem;
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-size: 0.82rem;
    color: #6688aa;
}
.price-result-meta span { color: #99bbcc; font-weight: 500; }

/* ─── Predict Button ─── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0066ff, #00aaff) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.7rem 2rem !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(0, 102, 255, 0.35) !important;
    transition: all 0.25s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(0, 180, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ─── Inputs ─── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(13,17,32,0.8) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
    color: #ddeeff !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: rgba(0,212,255,0.4) !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px;
    overflow: hidden;
}

/* ─── Caption / Footer ─── */
.dash-footer {
    text-align: center;
    color: #2a3a5a;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 1.5rem 0 1rem;
}

/* ─── Multiselect ─── */
.stMultiSelect > div > div {
    background: rgba(13,17,32,0.8) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
}

/* ─── Similar cars section title ─── */
.similar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a6880;
    margin: 1.5rem 0 0.75rem;
}
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8899bb', family='DM Sans'),
    margin=dict(l=10, r=10, t=10, b=10),
)

# ── Load Data ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cars_updated_2.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("xgb_best_model.pkl")

df = load_data()
model = load_model()

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(135deg,#fff,#00d4ff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:0.2rem;">
            🚗 AutoInsight
        </div>
        <div style="color:#2a3a5a; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em;">
            Market Intelligence
        </div>
    </div>
    <hr style="border:none; height:1px; background:linear-gradient(90deg,rgba(0,212,255,0.2),transparent); margin-bottom:1.5rem;">
    """, unsafe_allow_html=True)

    page = st.radio("", ["📊 Dashboard", "🤖 Price Predictor"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    if page == "📊 Dashboard":
        st.markdown('<div style="color:#2a4060; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.5rem;">Filter by Brand</div>', unsafe_allow_html=True)
        all_brands = sorted(df['brand'].unique())
        selected_brands = st.multiselect("", all_brands, default=all_brands[:15], label_visibility="collapsed")
        dff = df[df['brand'].isin(selected_brands)] if selected_brands else df

# ══════════════════════════════════════════════════════
#  PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # Header
    st.markdown('<div class="page-title">Cars Market Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time market intelligence · Used cars analysis</div>', unsafe_allow_html=True)

    # ── KPI Row ─────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon-box">🚙</div>
            <div class="kpi-text">
                <div class="kpi-label">Total Listings</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon-box">🏷️</div>
            <div class="kpi-text">
                <div class="kpi-label">Active Brands</div>
                <div class="kpi-value">{df['brand'].nunique()}</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon-box">💵</div>
            <div class="kpi-text">
                <div class="kpi-label">Avg Market Price</div>
                <div class="kpi-value">${df['price'].mean():,.0f}</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon-box">⚡</div>
            <div class="kpi-text">
                <div class="kpi-label">Avg Horsepower</div>
                <div class="kpi-value">{df['horsepower'].mean():,.0f}<sup>HP</sup></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ══ ROW 1 ══
    st.markdown("""
    <div class="section-heading">
        <div class="section-heading-icon">📊</div>
        <div class="section-heading-text">Sales Volume & Mileage</div>
        <div class="section-heading-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="chart-label">Sales Trend per Brand · Line Chart</div>', unsafe_allow_html=True)
        yearly = dff.groupby(['brand', 'model_year']).size().reset_index(name='count')
        fig1 = px.line(
            yearly, x='model_year', y='count', color='brand',
            labels={'model_year': 'Year', 'count': 'Count', 'brand': 'Brand'},
            template='plotly_dark', height=400,
            color_discrete_sequence=PALETTE,
        )
        fig1.update_traces(line=dict(width=2), mode='lines+markers',
                           marker=dict(size=4))
        fig1.update_layout(**LAYOUT,
            legend=dict(orientation='v', x=1.01, y=1, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(0,212,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(0,212,255,0.1)'),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-label">Avg Mileage per Brand · Lollipop</div>', unsafe_allow_html=True)
        avg_mil = dff.groupby('brand')['milage'].mean().reset_index()
        avg_mil.columns = ['brand', 'avg_mileage']
        avg_mil = avg_mil.sort_values('avg_mileage', ascending=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=avg_mil['avg_mileage'], y=avg_mil['brand'],
            mode='markers',
            marker=dict(
                color=avg_mil['avg_mileage'],
                colorscale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
                size=11, line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            name='Avg Mileage'
        ))
        for _, row in avg_mil.iterrows():
            fig2.add_shape(
                type='line',
                x0=0, x1=row['avg_mileage'],
                y0=row['brand'], y1=row['brand'],
                line=dict(color='rgba(0, 212, 255, 0.18)', width=1.5)
            )
        fig2.update_layout(**LAYOUT,
            height=400,
            xaxis=dict(title='Avg Mileage (mi)', gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(0,212,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ══ ROW 2 ══
    st.markdown("""
    <div class="section-heading">
        <div class="section-heading-icon">🎨</div>
        <div class="section-heading-text">Color Popularity Analysis</div>
        <div class="section-heading-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown('<div class="chart-label">Exterior Colors · Treemap</div>', unsafe_allow_html=True)
        ext_counts = dff.groupby(['brand', 'ext_col']).size().reset_index(name='count')
        ext_top = ext_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig3 = px.treemap(
            ext_top, path=['brand', 'ext_col'], values='count',
            template='plotly_dark', height=400,
            color='count',
            color_continuous_scale=[[0, '#001a4d'], [0.5, '#004488'], [1, '#00d4ff']],
        )
        fig3.update_traces(
            textfont=dict(color='white', size=12),
            marker=dict(cornerradius=6),
        )
        fig3.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="chart-label">Interior Colors · Sunburst</div>', unsafe_allow_html=True)
        int_counts = dff.groupby(['brand', 'int_col']).size().reset_index(name='count')
        int_top = int_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig4 = px.sunburst(
            int_top, path=['brand', 'int_col'], values='count',
            template='plotly_dark', height=400,
            color='count',
            color_continuous_scale=[[0, '#001a4d'], [0.5, '#004488'], [1, '#00d4ff']],
        )
        fig4.update_traces(textfont=dict(color='white', size=11))
        fig4.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ══ ROW 3 ══
    st.markdown("""
    <div class="section-heading">
        <div class="section-heading-icon">⚡</div>
        <div class="section-heading-text">Performance & Pricing</div>
        <div class="section-heading-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2, gap="medium")

    with col5:
        st.markdown('<div class="chart-label">Avg Horsepower per Brand · Bar Chart</div>', unsafe_allow_html=True)
        avg_hp = dff.groupby('brand')['horsepower'].mean().reset_index()
        avg_hp.columns = ['brand', 'avg_hp']
        avg_hp = avg_hp.sort_values('avg_hp', ascending=False)
        fig5 = px.bar(
            avg_hp, x='brand', y='avg_hp',
            labels={'avg_hp': 'Avg HP', 'brand': 'Brand'},
            template='plotly_dark', color='avg_hp',
            color_continuous_scale=[[0, '#001a4d'], [0.5, '#0055cc'], [1, '#00d4ff']],
            height=400,
        )
        fig5.update_layout(**{**LAYOUT, 'margin': dict(l=10, r=10, t=5, b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown('<div class="chart-label">Avg Price per Brand · Bar Chart</div>', unsafe_allow_html=True)
        avg_price = dff.groupby('brand')['price'].mean().reset_index()
        avg_price.columns = ['brand', 'avg_price']
        avg_price = avg_price.sort_values('avg_price', ascending=False)
        fig6 = px.bar(
            avg_price, x='brand', y='avg_price',
            labels={'avg_price': 'Avg Price (USD)', 'brand': 'Brand'},
            template='plotly_dark', color='avg_price',
            color_continuous_scale=[[0, '#001a4d'], [0.5, '#0055cc'], [1, '#00d4ff']],
            height=400,
        )
        fig6.update_layout(**{**LAYOUT, 'margin': dict(l=10, r=10, t=5, b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        )
        fig6.update_yaxes(tickprefix='$', tickformat=',.0f')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="dash-footer">AutoInsight · Cars Market Dashboard · Built with Streamlit & Plotly</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 2 – PRICE PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "🤖 Price Predictor":

    st.markdown('<div class="page-title">Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">XGBoost Model · Market Price Estimation</div>', unsafe_allow_html=True)

    brands        = sorted(df['brand'].unique())
    fuel_types    = sorted(df['fuel_type'].dropna().unique())
    transmissions = sorted(df['transmission'].dropna().unique())
    ext_colors    = sorted(df['ext_col'].dropna().unique())
    int_colors    = sorted(df['int_col'].dropna().unique())
    cylinders     = ['cyl_3','cyl_4','cyl_5','cyl_6','cyl_8','cyl_10','cyl_12']

    # Section 1
    st.markdown("""
    <div class="pred-section">
        <div class="pred-section-title">🏷️ &nbsp; Brand & Specifications</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            brand = st.selectbox("Brand", brands)
            fuel_type = st.selectbox("Fuel Type", fuel_types)
        with c2:
            transmission = st.selectbox("Transmission", transmissions)
            num_cylinders = st.selectbox("Cylinders", cylinders, index=3)
        with c3:
            ext_col = st.selectbox("Exterior Color", ext_colors)
            int_col = st.selectbox("Interior Color", int_colors)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2
    st.markdown("""
    <div class="pred-section">
        <div class="pred-section-title">📐 &nbsp; Numeric Specs</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        n1, n2, n3, n4 = st.columns(4, gap="medium")
        with n1:
            model_year = st.number_input("Model Year",
                min_value=int(df['model_year'].min()),
                max_value=int(df['model_year'].max()),
                value=2020, step=1)
        with n2:
            horsepower = st.number_input("Horsepower (HP)",
                min_value=50, max_value=2000,
                value=int(df['horsepower'].median()), step=10)
        with n3:
            engine_size = st.number_input("Engine Size (L)",
                min_value=0.5, max_value=10.0,
                value=float(round(df['engine_size_liters'].median(), 1)),
                step=0.1, format="%.1f")
        with n4:
            milage = st.number_input("Mileage (miles)",
                min_value=0, max_value=500000,
                value=int(df['milage'].median()), step=1000)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮  Predict Price", use_container_width=True, type="primary")

    if predict_clicked:
        try:
            input_data = pd.DataFrame([{
                'brand': brand,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'ext_col': ext_col,
                'int_col': int_col,
                'num_cylinders': num_cylinders,
                'model_year': model_year,
                'horsepower': horsepower,
                'engine_size_liters': engine_size,
                'milage': milage,
            }])

            predicted_price = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="price-result-wrap">
                <div class="price-result-glow"></div>
                <div class="price-result-inner">
                    <div class="price-result-label">Estimated Market Value</div>
                    <div class="price-result-value">${predicted_price:,.0f}</div>
                    <div class="price-result-meta">
                        <span>{brand}</span> · <span>{model_year}</span> · <span>{horsepower} HP</span> · <span>{milage:,} mi</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            similar = df[
                (df['brand'] == brand) &
                (df['model_year'].between(model_year - 2, model_year + 2)) &
                (df['horsepower'].between(horsepower * 0.85, horsepower * 1.15))
            ][['brand', 'model_year', 'horsepower', 'milage', 'price']].head(8)

            if not similar.empty:
                st.markdown('<div class="similar-title">Similar Cars in Dataset</div>', unsafe_allow_html=True)
                similar = similar.reset_index(drop=True)
                similar_display = similar.copy()
                similar_display['price'] = similar_display['price'].apply(lambda x: f"${x:,.0f}")
                similar_display['milage'] = similar_display['milage'].apply(lambda x: f"{x:,}")
                similar_display.columns = ['Brand', 'Year', 'HP', 'Mileage', 'Price']
                st.dataframe(similar_display, use_container_width=True, hide_index=True)

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=similar['model_year'].astype(str) + ' #' + (similar.index + 1).astype(str),
                    y=similar['price'],
                    marker=dict(
                        color=similar['price'],
                        colorscale=[[0, '#001a4d'], [0.5, '#0055cc'], [1, '#00d4ff']],
                        line=dict(color='rgba(0,212,255,0.15)', width=1),
                    ),
                    name='Similar Cars'
                ))
                fig_comp.add_hline(
                    y=predicted_price,
                    line_dash='dot', line_color='#00d4ff', line_width=2,
                    annotation_text=f"  Your Prediction: ${predicted_price:,.0f}",
                    annotation_font_color='#00d4ff',
                    annotation_font_size=12,
                )
                fig_comp.update_layout(
                    **{**LAYOUT, 'margin': dict(l=10, r=10, t=30, b=10)},
                    template='plotly_dark',
                    height=340,
                    xaxis=dict(title='Similar Cars', gridcolor='rgba(255,255,255,0.04)'),
                    yaxis=dict(title='Price (USD)', gridcolor='rgba(255,255,255,0.04)', tickprefix='$', tickformat=',.0f'),
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No closely matching cars found in the dataset.")

        except Exception as e:
            st.error("Prediction failed. Please check inputs or model compatibility.")
            st.write("Debug info:", str(e))

    st.markdown('<div class="dash-footer">AutoInsight · Price Predictor · XGBoost Model</div>', unsafe_allow_html=True)
