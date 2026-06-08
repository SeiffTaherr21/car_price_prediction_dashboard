import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title="Cars Market Analytics", layout="wide", page_icon="🚗")

PALETTE = [
    "#00d4ff", "#00b8e6", "#009dcc", "#0081b3",
    "#0066ff", "#1a6aff", "#3385ff", "#4d9fff",
    "#66b8ff", "#80d0ff", "#99e0ff", "#b3f0ff",
    "#cce8ff", "#d9f0ff", "#e6f8ff",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #080b14; }
.block-container { padding-top: 2rem; padding-left: 2.5rem; padding-right: 2.5rem; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #080b14 100%);
    border-right: 1px solid rgba(0,212,255,0.12);
}
div[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #00d4ff;
    letter-spacing: -0.5px;
    padding: 0.75rem 0 1.25rem 0;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    margin-bottom: 1.5rem;
}
.sidebar-logo span { color: #ffffff; }

.page-header {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 0.25rem;
}
.page-header span { color: #00d4ff; }
.page-sub { color: #5a6a8a; font-size: 0.9rem; margin-bottom: 1.75rem; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1120 0%, #101525 100%);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0066ff, #00d4ff);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 0.5rem; display: block; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-label {
    color: #4a5a7a;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1.25rem 0;
}
.section-header-line {
    width: 3px; height: 22px;
    background: linear-gradient(180deg, #00d4ff, #0066ff);
    border-radius: 2px; flex-shrink: 0;
}
.section-header-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e0e8ff;
    letter-spacing: -0.2px;
}

.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #8899bb;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
    margin: 1.5rem 0;
}

.pred-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1.25rem;
}

.price-result {
    background: linear-gradient(135deg, rgba(0,102,255,0.12), rgba(0,212,255,0.08));
    border: 1px solid rgba(0,212,255,0.35);
    border-radius: 18px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.price-result::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0066ff, #00d4ff, #0066ff);
}
.price-tag {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0.5rem 0;
}
.price-tag span { color: #00d4ff; }
.price-label {
    color: #5a6a8a;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.price-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: #00d4ff;
    margin: 0.2rem;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0066ff, #00b8e6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
}

div[data-testid="stMetric"] { display: none; }

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stMultiSelect"] label {
    color: #8899bb !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

.footer {
    text-align: center;
    color: #2a3a5a;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    plot_bgcolor='#0d1120',
    paper_bgcolor='#0d1120',
    font=dict(color='#8899bb', family='DM Sans'),
    margin=dict(l=10, r=10, t=10, b=10),
)

@st.cache_data
def load_data():
    return pd.read_csv("cars_updated_2.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_best_model.pkl")

df = load_data()
model = load_model()

st.sidebar.markdown('<div class="sidebar-logo">🚗 Cars<span>Analytics</span></div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["📊 Dashboard", "🤖 Price Predictor"], label_visibility="collapsed")
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.markdown("""
    <div class="page-header">Market <span>Dashboard</span></div>
    <div class="page-sub">Used Cars — Real-Time Analytics & Insights</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="kpi-icon">🚗</span>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-label">Total Listings</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">🏷️</span>
            <div class="kpi-value">{df['brand'].nunique()}</div>
            <div class="kpi-label">Brands</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">💰</span>
            <div class="kpi-value">${df['price'].mean():,.0f}</div>
            <div class="kpi-label">Avg Price</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">⚡</span>
            <div class="kpi-value">{df['horsepower'].mean():,.0f} <small style="font-size:1rem;color:#4a5a7a">HP</small></div>
            <div class="kpi-label">Avg Horsepower</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    all_brands = sorted(df['brand'].unique())
    st.sidebar.markdown("**Filter by Brand**")
    selected_brands = st.sidebar.multiselect("", all_brands, default=all_brands[:15], label_visibility="collapsed")
    dff = df[df['brand'].isin(selected_brands)] if selected_brands else df

    # ROW 1
    st.markdown('<div class="section-header"><div class="section-header-line"></div><div class="section-header-text">Sales Volume & Mileage Analysis</div></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-title">📈 Sales Trend per Brand</div>', unsafe_allow_html=True)
        yearly = dff.groupby(['brand', 'model_year']).size().reset_index(name='count')
        fig1 = px.line(yearly, x='model_year', y='count', color='brand',
            labels={'model_year': 'Year', 'count': 'Sales Count', 'brand': 'Brand'},
            template='plotly_dark', height=400, color_discrete_sequence=PALETTE)
        fig1.update_traces(line=dict(width=2))
        fig1.update_layout(**LAYOUT,
            legend=dict(orientation='v', x=1.01, y=1, bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown('<div class="chart-title">🛣️ Avg Mileage per Brand</div>', unsafe_allow_html=True)
        avg_mil = dff.groupby('brand')['milage'].mean().reset_index()
        avg_mil.columns = ['brand', 'avg_mileage']
        avg_mil = avg_mil.sort_values('avg_mileage', ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=avg_mil['avg_mileage'], y=avg_mil['brand'],
            mode='markers',
            marker=dict(color=avg_mil['avg_mileage'],
                colorscale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']],
                size=12, line=dict(color='white', width=1))))
        for _, row in avg_mil.iterrows():
            fig2.add_shape(type='line', x0=0, x1=row['avg_mileage'],
                y0=row['brand'], y1=row['brand'],
                line=dict(color='rgba(0,212,255,0.2)', width=1.5))
        fig2.update_layout(**LAYOUT, height=400,
            xaxis=dict(title='Avg Mileage (miles)', gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(title='', gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig2, use_container_width=True)

    # ROW 2
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-header-line"></div><div class="section-header-text">Color Popularity Analysis</div></div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-title">🚘 Exterior Colors Distribution</div>', unsafe_allow_html=True)
        ext_counts = dff.groupby(['brand', 'ext_col']).size().reset_index(name='count')
        ext_top = ext_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig3 = px.treemap(ext_top, path=['brand', 'ext_col'], values='count',
            template='plotly_dark', height=400, color='count',
            color_continuous_scale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']])
        fig3.update_traces(textfont=dict(color='white'), marker=dict(cornerradius=5))
        fig3.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="chart-title">🪑 Interior Colors Distribution</div>', unsafe_allow_html=True)
        int_counts = dff.groupby(['brand', 'int_col']).size().reset_index(name='count')
        int_top = int_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig4 = px.sunburst(int_top, path=['brand', 'int_col'], values='count',
            template='plotly_dark', height=400, color='count',
            color_continuous_scale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']])
        fig4.update_traces(textfont=dict(color='white'))
        fig4.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ROW 3
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-header-line"></div><div class="section-header-text">Performance & Pricing Analysis</div></div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="chart-title">⚡ Avg Horsepower per Brand</div>', unsafe_allow_html=True)
        avg_hp = dff.groupby('brand')['horsepower'].mean().reset_index()
        avg_hp.columns = ['brand', 'avg_hp']
        avg_hp = avg_hp.sort_values('avg_hp', ascending=False)
        fig5 = px.bar(avg_hp, x='brand', y='avg_hp',
            labels={'avg_hp': 'Avg HP', 'brand': 'Brand'},
            template='plotly_dark', color='avg_hp',
            color_continuous_scale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']], height=400)
        fig5.update_layout(**{**LAYOUT, 'margin': dict(l=10,r=10,t=10,b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'), bargap=0.3)
        fig5.update_traces(marker_line_width=0)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown('<div class="chart-title">💰 Avg Price per Brand</div>', unsafe_allow_html=True)
        avg_price = dff.groupby('brand')['price'].mean().reset_index()
        avg_price.columns = ['brand', 'avg_price']
        avg_price = avg_price.sort_values('avg_price', ascending=False)
        fig6 = px.bar(avg_price, x='brand', y='avg_price',
            labels={'avg_price': 'Avg Price (USD)', 'brand': 'Brand'},
            template='plotly_dark', color='avg_price',
            color_continuous_scale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']], height=400)
        fig6.update_layout(**{**LAYOUT, 'margin': dict(l=10,r=10,t=10,b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)'), bargap=0.3)
        fig6.update_traces(marker_line_width=0)
        fig6.update_yaxes(tickprefix='$', tickformat=',.0f')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="footer">Cars Market Analytics · Built with Streamlit & Plotly · XGBoost Model</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 2 – PRICE PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "🤖 Price Predictor":

    st.markdown("""
    <div class="page-header">Price <span>Predictor</span></div>
    <div class="page-sub">XGBoost Model — Estimate Market Value Instantly</div>
    """, unsafe_allow_html=True)

    brands        = sorted(df['brand'].unique())
    fuel_types    = sorted(df['fuel_type'].dropna().unique())
    transmissions = sorted(df['transmission'].dropna().unique())
    ext_colors    = sorted(df['ext_col'].dropna().unique())
    int_colors    = sorted(df['int_col'].dropna().unique())
    cylinders     = ['cyl_3','cyl_4','cyl_5','cyl_6','cyl_8','cyl_10','cyl_12']

    st.markdown('<div class="pred-section-title">🏷️ Brand & Specs</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", brands)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
    with c2:
        transmission = st.selectbox("Transmission", transmissions)
        num_cylinders = st.selectbox("Cylinders", cylinders, index=3)
    with c3:
        ext_col = st.selectbox("Exterior Color", ext_colors)
        int_col = st.selectbox("Interior Color", int_colors)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pred-section-title">📐 Numeric Specs</div>', unsafe_allow_html=True)

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        model_year = st.number_input("Model Year",
            min_value=int(df['model_year'].min()), max_value=int(df['model_year'].max()),
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

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮 Predict Price", use_container_width=True, type="primary")

    if predict_clicked:
        try:
            input_data = pd.DataFrame([{
                'brand': brand, 'fuel_type': fuel_type, 'transmission': transmission,
                'ext_col': ext_col, 'int_col': int_col, 'num_cylinders': num_cylinders,
                'model_year': model_year, 'horsepower': horsepower,
                'engine_size_liters': engine_size, 'milage': milage,
            }])

            predicted_price = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="price-result">
                <div class="price-label">Estimated Market Value</div>
                <div class="price-tag"><span>$</span>{predicted_price:,.0f}</div>
                <div style="margin-top:0.75rem">
                    <span class="price-badge">{brand}</span>
                    <span class="price-badge">{model_year}</span>
                    <span class="price-badge">{horsepower} HP</span>
                    <span class="price-badge">{milage:,} mi</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header"><div class="section-header-line"></div><div class="section-header-text">Similar Cars in Dataset</div></div>', unsafe_allow_html=True)

            similar = df[
                (df['brand'] == brand) &
                (df['model_year'].between(model_year - 2, model_year + 2)) &
                (df['horsepower'].between(horsepower * 0.85, horsepower * 1.15))
            ][['brand', 'model_year', 'horsepower', 'milage', 'price']].head(8)

            if not similar.empty:
                similar = similar.reset_index(drop=True)
                similar_display = similar.copy()
                similar_display['price'] = similar_display['price'].apply(lambda x: f"${x:,.0f}")
                similar_display['milage'] = similar_display['milage'].apply(lambda x: f"{x:,}")
                similar_display.columns = ['Brand', 'Year', 'HP', 'Mileage', 'Price']
                st.dataframe(similar_display, use_container_width=True, hide_index=True)

                st.markdown('<div class="chart-title" style="margin-top:1rem">📊 Price Comparison</div>', unsafe_allow_html=True)
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=similar['model_year'].astype(str) + ' #' + (similar.index + 1).astype(str),
                    y=similar['price'],
                    marker=dict(color=similar['price'],
                        colorscale=[[0,'#0066ff'],[0.5,'#009dcc'],[1,'#00d4ff']],
                        line=dict(width=0)),
                    name='Similar Cars'
                ))
                fig_comp.add_hline(y=predicted_price, line_dash='dash',
                    line_color='#00d4ff', line_width=2,
                    annotation_text=f"  Prediction: ${predicted_price:,.0f}",
                    annotation_font_color='#00d4ff', annotation_font_size=12)
                fig_comp.update_layout(
                    **{**LAYOUT, 'margin': dict(l=10,r=10,t=30,b=10)},
                    template='plotly_dark', height=330, bargap=0.35,
                    xaxis=dict(title='Similar Cars', gridcolor='rgba(255,255,255,0.04)'),
                    yaxis=dict(title='Price (USD)', gridcolor='rgba(255,255,255,0.04)',
                        tickprefix='$', tickformat=',.0f'))
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No closely matching cars found in the dataset.")

        except Exception as e:
            st.error("Prediction failed. Please check inputs or model compatibility.")
            st.write("Debug info:", str(e))

    st.markdown('<div class="footer">Price Predictor · XGBoost Model · Cars Market Analytics</div>', unsafe_allow_html=True)