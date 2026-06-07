import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title="🚗 Cars Dashboard", layout="wide", page_icon="🚗")

# ── Unified color palette: cyan → blue gradient
PALETTE = [
    "#00d4ff", "#00b8e6", "#009dcc", "#0081b3",
    "#0066ff", "#1a6aff", "#3385ff", "#4d9fff",
    "#66b8ff", "#80d0ff", "#99e0ff", "#b3f0ff",
    "#cce8ff", "#d9f0ff", "#e6f8ff",
]

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #00d4ff; font-size: 2.2rem; text-align: center; }
    .stMetric label { color: #aaa; }
    div[data-testid="stSidebar"] { background-color: #1a1d2e; }

    .pred-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
        border: 1px solid #00d4ff33;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1rem;
    }
    .price-result {
        background: linear-gradient(135deg, #00d4ff22 0%, #0066ff22 100%);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .price-tag {
        font-size: 3rem;
        font-weight: 800;
        color: #00d4ff;
        letter-spacing: -1px;
    }
    .price-label {
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .section-title {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 3px solid #00d4ff;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(
    plot_bgcolor='#1a1d2e',
    paper_bgcolor='#1a1d2e',
    font=dict(color='#cccccc'),
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

# ── Navigation ─────────────────────────────────────────
st.sidebar.markdown("## 🚗 Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "🤖 Price Predictor"], label_visibility="collapsed")

# ══════════════════════════════════════════════════════
#  PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("🚗 Cars Market Dashboard")
    st.markdown("---")

    # ── KPI Row ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cars", f"{len(df):,}")
    k2.metric("Brands", df['brand'].nunique())
    k3.metric("Avg Price", f"${df['price'].mean():,.0f}")
    k4.metric("Avg Horsepower", f"{df['horsepower'].mean():,.0f} HP")

    st.markdown("---")

    # ── Sidebar filter ───────────────────────────────────
    all_brands = sorted(df['brand'].unique())
    selected_brands = st.sidebar.multiselect("Filter by Brand", all_brands, default=all_brands[:15])
    dff = df[df['brand'].isin(selected_brands)] if selected_brands else df

    # ══ ROW 1: Sales Volume & Mileage ══
    st.subheader("📊 Sales Volume & Mileage Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Sales Trend per Brand")
        yearly = dff.groupby(['brand', 'model_year']).size().reset_index(name='count')
        fig1 = px.line(
            yearly, x='model_year', y='count', color='brand',
            labels={'model_year': 'Year', 'count': 'Sales Count', 'brand': 'Brand'},
            template='plotly_dark', height=430,
            color_discrete_sequence=PALETTE,
        )
        fig1.update_traces(line=dict(width=2))
        fig1.update_layout(**LAYOUT,
            legend=dict(orientation='v', x=1.01, y=1),
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 🛣️ Avg Mileage per Brand")
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
                size=12, line=dict(color='white', width=1)
            ),
            name='Avg Mileage'
        ))
        for _, row in avg_mil.iterrows():
            fig2.add_shape(
                type='line',
                x0=0, x1=row['avg_mileage'],
                y0=row['brand'], y1=row['brand'],
                line=dict(color='rgba(0, 212, 255, 0.27)', width=1.5)
            )
        fig2.update_layout(**LAYOUT,
            height=430,
            xaxis=dict(title='Avg Mileage (miles)', gridcolor='#2a2d3e'),
            yaxis=dict(title='Brand'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ══ ROW 2: Color Popularity ══
    st.markdown("---")
    st.subheader("🎨 Color Popularity Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🚘 Exterior Colors Distribution")
        ext_counts = dff.groupby(['brand', 'ext_col']).size().reset_index(name='count')
        ext_top = ext_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig3 = px.treemap(
            ext_top,
            path=['brand', 'ext_col'],
            values='count',
            template='plotly_dark',
            height=430,
            color='count',
            color_continuous_scale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
        )
        fig3.update_traces(
            textfont=dict(color='white'),
            marker=dict(cornerradius=4),
        )
        fig3.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 🪑 Interior Colors Distribution")
        int_counts = dff.groupby(['brand', 'int_col']).size().reset_index(name='count')
        int_top = int_counts.sort_values('count', ascending=False).groupby('brand').head(5).reset_index(drop=True)
        fig4 = px.sunburst(
            int_top,
            path=['brand', 'int_col'],
            values='count',
            template='plotly_dark',
            height=430,
            color='count',
            color_continuous_scale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
        )
        fig4.update_traces(textfont=dict(color='white'))
        fig4.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ══ ROW 3: Performance & Pricing ══
    st.markdown("---")
    st.subheader("⚡ Performance & Pricing Analysis")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### ⚡ Avg Horsepower per Brand")
        avg_hp = dff.groupby('brand')['horsepower'].mean().reset_index()
        avg_hp.columns = ['brand', 'avg_hp']
        avg_hp = avg_hp.sort_values('avg_hp', ascending=False)
        fig5 = px.bar(
            avg_hp, x='brand', y='avg_hp',
            labels={'avg_hp': 'Avg Horsepower (HP)', 'brand': 'Brand'},
            template='plotly_dark', color='avg_hp',
            color_continuous_scale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
            height=420,
        )
        fig5.update_layout(**{**LAYOUT, 'margin': dict(l=10, r=10, t=10, b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("#### 💰 Avg Price per Brand")
        avg_price = dff.groupby('brand')['price'].mean().reset_index()
        avg_price.columns = ['brand', 'avg_price']
        avg_price = avg_price.sort_values('avg_price', ascending=False)
        fig6 = px.bar(
            avg_price, x='brand', y='avg_price',
            labels={'avg_price': 'Avg Price (USD)', 'brand': 'Brand'},
            template='plotly_dark', color='avg_price',
            color_continuous_scale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
            height=420,
        )
        fig6.update_layout(**{**LAYOUT, 'margin': dict(l=10, r=10, t=10, b=80)},
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45),
        )
        fig6.update_yaxes(tickprefix='$', tickformat=',.0f')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.caption("Cars Market Analysis Dashboard · Built with Streamlit & Plotly")

# ══════════════════════════════════════════════════════
#  PAGE 2 – PRICE PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "🤖 Price Predictor":

    st.title("🤖 Car Price Predictor")
    st.markdown("---")

    brands        = sorted(df['brand'].unique())
    fuel_types    = sorted(df['fuel_type'].dropna().unique())
    transmissions = sorted(df['transmission'].dropna().unique())
    ext_colors    = sorted(df['ext_col'].dropna().unique())
    int_colors    = sorted(df['int_col'].dropna().unique())
    cylinders     = ['cyl_3','cyl_4','cyl_5','cyl_6','cyl_8','cyl_10','cyl_12']

    with st.container():
        st.markdown('<div class="section-title">🏷️ Brand & Specs</div>', unsafe_allow_html=True)
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

    st.markdown("---")

    with st.container():
        st.markdown('<div class="section-title">📐 Numeric Specs</div>', unsafe_allow_html=True)
        n1, n2, n3, n4 = st.columns(4)

        with n1:
            model_year = st.number_input(
                "Model Year",
                min_value=int(df['model_year'].min()),
                max_value=int(df['model_year'].max()),
                value=2020, step=1
            )
        with n2:
            horsepower = st.number_input(
                "Horsepower (HP)",
                min_value=50, max_value=2000,
                value=int(df['horsepower'].median()), step=10
            )
        with n3:
            engine_size = st.number_input(
                "Engine Size (L)",
                min_value=0.5, max_value=10.0,
                value=float(round(df['engine_size_liters'].median(), 1)),
                step=0.1, format="%.1f"
            )
        with n4:
            milage = st.number_input(
                "Mileage (miles)",
                min_value=0, max_value=500000,
                value=int(df['milage'].median()), step=1000
            )

    st.markdown("---")

    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮 Predict Price", use_container_width=True, type="primary")

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
            <div class="price-result">
                <div class="price-label">Estimated Market Price</div>
                <div class="price-tag">${predicted_price:,.0f}</div>
                <div style="color:#aaa; margin-top:0.5rem; font-size:0.85rem;">
                    {brand} · {model_year} · {horsepower} HP · {milage:,} miles
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Similar Cars in Dataset")
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

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=similar['model_year'].astype(str) + ' #' + (similar.index + 1).astype(str),
                    y=similar['price'],
                    marker=dict(
                        color=similar['price'],
                        colorscale=[[0, '#0066ff'], [0.5, '#009dcc'], [1, '#00d4ff']],
                    ),
                    name='Similar Cars'
                ))
                fig_comp.add_hline(
                    y=predicted_price,
                    line_dash='dash', line_color='#00d4ff', line_width=2,
                    annotation_text=f"Your Prediction: ${predicted_price:,.0f}",
                    annotation_font_color='#00d4ff'
                )
                fig_comp.update_layout(
                    **{**LAYOUT, 'margin': dict(l=10, r=10, t=40, b=10)},
                    template='plotly_dark',
                    title='Predicted Price vs Similar Cars',
                    height=350,
                    xaxis_title='Similar Cars',
                    yaxis_title='Price (USD)'
                )
                fig_comp.update_yaxes(tickprefix='$', tickformat=',.0f')
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No closely matching cars found in the dataset.")

        except Exception as e:
            st.error("Prediction failed. Please check inputs or model compatibility.")
            st.write("Debug info:", str(e))

    st.markdown("---")
    st.caption("Price Predictor · XGBoost Model · Cars Market Analysis Dashboard")
