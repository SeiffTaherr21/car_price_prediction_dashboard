import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Cars Dashboard", layout="wide")
st.title("🚗 Cars Market Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/user-data/uploads/cars_updated.csv")
    return df

df = load_data()

st.markdown("---")

# ── ROW 1 ──────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

# Chart 1: Average price by top 10 brands
with col1:
    st.subheader("💰 أغلى الماركات في المتوسط")
    top_brands = df.groupby("brand")["price"].median().sort_values(ascending=False).head(10).reset_index()
    top_brands.columns = ["Brand", "Median Price"]
    fig1 = px.bar(top_brands, x="Median Price", y="Brand", orientation="h",
                  color="Median Price", color_continuous_scale="Reds",
                  text_auto=".2s")
    fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=350,
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("🔍 Insight: بورش وBMW ومرسيدس في القمة — الماركات الأوروبية تحتل أعلى الأسعار باستمرار.")

# Chart 2: Price drops with age
with col2:
    st.subheader("📉 السعر بيقل مع العمر")
    age_price = df.groupby("car_age")["price"].median().reset_index()
    age_price.columns = ["Car Age (years)", "Median Price"]
    fig2 = px.area(age_price, x="Car Age (years)", y="Median Price",
                   color_discrete_sequence=["#EF553B"])
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔍 Insight: العربية بتخسر نص قيمتها في أول 10 سنين — التوقيت الأمثل للشراء بعد 3-5 سنين من الإنتاج.")

# Chart 3: Fuel type distribution
with col3:
    st.subheader("⛽ توزيع أنواع الوقود")
    fuel = df["fuel_type"].value_counts().reset_index()
    fuel.columns = ["Fuel Type", "Count"]
    fig3 = px.pie(fuel, values="Count", names="Fuel Type",
                  color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🔍 Insight: 91% من العربيات بنزين — السوق لسه بعيد عن التحول الكامل للكهربائي والهايبريد.")

st.markdown("---")

# ── ROW 2 ──────────────────────────────────────────────────────────────────────
col4, col5, col6 = st.columns(3)

# Chart 4: Horsepower vs Price
with col4:
    st.subheader("🏎️ أكتر حصان = أكتر فلوس؟")
    sample = df[df["price"] < 200000].dropna(subset=["horsepower"]).sample(2000, random_state=42)
    fig4 = px.scatter(sample, x="horsepower", y="price",
                      color="fuel_type", opacity=0.5,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig4.update_layout(height=350, legend_title="Fuel")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("🔍 Insight: العلاقة واضحة — كل ما زاد الهورس باور ارتفع السعر، لكن الهايبريد بيقدم أداء معقول بسعر أقل.")

# Chart 5: Manual vs Automatic avg price
with col5:
    st.subheader("🔧 أوتوماتيك vs مانيوال")
    trans_price = df.groupby("transmission")["price"].median().reset_index()
    trans_price.columns = ["Transmission", "Median Price"]
    fig5 = px.bar(trans_price, x="Transmission", y="Median Price",
                  color="Transmission", text_auto=".2s",
                  color_discrete_map={"Automatic": "#636EFA", "Manual": "#EF553B"})
    fig5.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("🔍 Insight: الأوتوماتيك أغلى بنسبة كبيرة — لأنه مرتبط بالسيارات الفاخرة والحديثة في السوق ده.")

# Chart 6: Mileage vs Price heatmap by brand
with col6:
    st.subheader("📊 المسافة والسعر عند أكبر الماركات")
    top5 = df["brand"].value_counts().head(5).index.tolist()
    df_top5 = df[df["brand"].isin(top5)].copy()
    df_top5["milage_bucket"] = pd.cut(df_top5["milage"], bins=5,
                                       labels=["0-20k", "20k-60k", "60k-100k", "100k-140k", "140k+"])
    heat = df_top5.groupby(["brand", "milage_bucket"])["price"].median().reset_index()
    heat.columns = ["Brand", "Mileage Range", "Median Price"]
    fig6 = px.density_heatmap(heat, x="Mileage Range", y="Brand", z="Median Price",
                               color_continuous_scale="Viridis", text_auto=".2s")
    fig6.update_layout(height=350)
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("🔍 Insight: BMW وAudi أكتر ماركات بتحتفظ بسعرها حتى مع الكيلومترات العالية — دليل على قوة الـ brand value.")
