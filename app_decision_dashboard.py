
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import re

# --- 1. Page Configuration & Styling ---
st.set_page_config(
    page_title="Sales Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #AAAAAA;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 5px;
        color: #FFFFFF;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #4CAF50;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Data Loading & Preprocessing ---
@st.cache_data
def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None

    # Date Handling
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
    
    # Weight Cleaning (Sync with EDA logic)
    def clean_weight(weight_str):
        if pd.isna(weight_str): return None
        weight_str = str(weight_str).lower().replace(' ', '')
        match = re.search(r'([\d\.]+)(kg|g)', weight_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'g': return value / 1000.0
            return value
        try: return float(weight_str)
        except: return None

    df['무게_수치'] = df['무게'].apply(clean_weight)
    
    # Clustering (On the fly)
    # Features
    req_cols = ['공급가', '결제금액(상품별)', '주문수량', '무게_수치', '등급', '옵션', '세트이벤트여부']
    # Ensure columns exist before using
    available_cols = [c for c in req_cols if c in df.columns]
    
    if len(available_cols) < len(req_cols):
        # Missing columns fallback
        df['Cluster'] = 0
        return df

    df_cluster = df[available_cols].copy()
    
    numeric_features = ['공급가', '결제금액(상품별)', '주문수량', '무게_수치']
    categorical_features = ['등급', '옵션', '세트이벤트여부']
    
    # Filter only available features
    numeric_features = [f for f in numeric_features if f in df_cluster.columns]
    categorical_features = [f for f in categorical_features if f in df_cluster.columns]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    try:
        X = preprocessor.fit_transform(df_cluster)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
    except Exception as e:
        df['Cluster'] = 0 # Fallback
        
    return df

data_path = 'data/data_classified_unified.csv'
df = load_and_process_data(data_path)

if df is None:
    st.stop()

# --- 3. Sidebar Filters ---
st.sidebar.title("🔍 검색 및 필터")

# Date Filter
if '주문일' in df.columns:
    min_date = df['주문일'].min().date()
    max_date = df['주문일'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "기간 선택",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )
else:
    st.sidebar.warning("주문일 데이터가 없습니다.")
    start_date, end_date = None, None

# Product Detail Filter (Replaces Grade)
if '상품명_상세' in df.columns:
    # Get unique vals, drop NA
    details = df['상품명_상세'].dropna().astype(str).unique().tolist()
    all_details = ['All'] + sorted(details)
    selected_detail = st.sidebar.selectbox("상품명 상세", all_details, key="detail_filter")
else:
    selected_detail = 'All'

# Filter Data
mask = pd.Series([True] * len(df))
if start_date and end_date:
    mask = mask & (df['주문일'].dt.date >= start_date) & (df['주문일'].dt.date <= end_date)

if selected_detail != 'All':
    mask = mask & (df['상품명_상세'] == selected_detail)

df_filtered = df[mask]

# --- 4. Main Dashboard ---
st.title("📊 매출 증대 전략 대시보드")
st.markdown("데이터 분석을 통한 **고객 세분화** 및 **매출 인사이트**를 제공합니다.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
total_sales = df_filtered['결제금액(상품별)'].sum()
total_orders = df_filtered.shape[0]
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
# Top product detail or Channel if detail is filtered
top_metric_label = "베스트 상품"
top_metric_val = "-"
if not df_filtered.empty:
    if '상품명_상세' in df_filtered.columns:
        top_metric_val = df_filtered['상품명_상세'].value_counts().idxmax()
        if len(str(top_metric_val)) > 15: top_metric_val = str(top_metric_val)[:15] + "..."
    else:
        top_metric_val = "N/A"

def metric_card(title, value, prefix="", suffix=""):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
    </div>
    """

with col1: st.markdown(metric_card("총 매출액", f"{total_sales:,.0f}", suffix="원"), unsafe_allow_html=True)
with col2: st.markdown(metric_card("총 주문건수", f"{total_orders:,.0f}", suffix="건"), unsafe_allow_html=True)
with col3: st.markdown(metric_card("평균 주문단가 (AOV)", f"{avg_order_value:,.0f}", suffix="원"), unsafe_allow_html=True)
with col4: st.markdown(metric_card(top_metric_label, top_metric_val), unsafe_allow_html=True)

st.write("") # Spacer

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 매출 개요", 
    "👥 고객 세분화", 
    "🛍️ 판매 채널/셀러", # New Tab
    "📦 상품 상세 분석",
    "🗓️ 기간/시간 분석",
    "👤 고객 분석"
])

with tab1:
    st.subheader("매출 트렌드 분석")
    if not df_filtered.empty and '주문일' in df_filtered.columns:
        date_range_days = (end_date - start_date).days
        if date_range_days > 60:
            freq = 'M' # Monthly
            date_col = df_filtered['주문일'].dt.to_period('M').astype(str)
            x_label = '월'
        else:
            freq = 'D' # Daily
            date_col = df_filtered['주문일'].dt.date
            x_label = '일'
            
        sales_trend = df_filtered.groupby(date_col)['결제금액(상품별)'].sum().reset_index()
        sales_trend.columns = ['Date', 'Sales']
        
        fig_trend = px.line(sales_trend, x='Date', y='Sales', title=f"기간별 매출 추이 ({x_label} 단위)",
                            markers=True, line_shape='spline')
        fig_trend.update_layout(xaxis_title="날짜", yaxis_title="매출액", template="plotly_dark",
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_trend.update_traces(line_color='#4CAF50', line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")
    else:
        st.info("데이터가 없습니다.")

with tab2:
    st.subheader("고객 클러스터링 기반 전략")
    if not df_filtered.empty:
        cluster_stats = df_filtered.groupby('Cluster').agg({
            '결제금액(상품별)': 'mean',
            '주문수량': 'mean',
            '무게_수치': 'mean',
            '등급': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        }).reset_index()
        cluster_stats.columns = ['Cluster', '평균주문금액', '평균주문수량', '평균무게(kg)', '주요등급']
        counts = df_filtered['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', '고객수']
        cluster_summary = pd.merge(cluster_stats, counts, on='Cluster')
        
        plot_data = df_filtered
        if len(plot_data) > 2000:
            plot_data = plot_data.sample(2000)
            
        fig_scatter = px.scatter_3d(plot_data, x='무게_수치', y='주문수량', z='결제금액(상품별)',
                                    color='Cluster', opacity=0.7,
                                    title="클러스터링 분포 (무게 vs 수량 vs 금액)",
                                    hover_data=['등급', '옵션'])
        fig_scatter.update_layout(scene = dict(
                        xaxis_title='무게(kg)',
                        yaxis_title='주문수량',
                        zaxis_title='금액'),
                        margin=dict(r=0, l=0, b=0, t=40),
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.plotly_chart(fig_scatter, use_container_width=True, key="cluster_scatter")
        
        with col_c2:
            st.write("#### 클러스터별 특성 및 전략")
            for _, row in cluster_summary.iterrows():
                cluster_id = int(row['Cluster'])
                avg_pay = row['평균주문금액']
                avg_qty = row['평균주문수량']
                
                strategy = "일반 고객 관리"
                if avg_pay > 100000:
                    strategy = "🥇 **VIP 관리**: 프리미엄 패키지 및 전용 혜택 제공"
                elif avg_qty > 5:
                    strategy = "📦 **대량 구매 유도**: 묶음 할인 및 B2B 제안"
                elif avg_pay < 30000:
                    strategy = "💸 **객단가 상승**: '1+1' 또는 '배송비 절약' 번들 제안"
                
                with st.expander(f"Cluster {cluster_id} (n={row['고객수']})", expanded=True):
                    st.write(f"- **특징**: 평균 {avg_pay:,.0f}원, 주력 '{row['주요등급']}'")
                    st.write(f"- **전략**: {strategy}")

with tab3: # New Tab: Channel & Seller
    st.subheader("판매 채널 및 셀러 분석")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        if '주문경로' in df_filtered.columns:
            channel_sales = df_filtered.groupby('주문경로')['결제금액(상품별)'].sum().sort_values(ascending=False).reset_index()
            fig_channel = px.pie(channel_sales, values='결제금액(상품별)', names='주문경로', title="채널별 매출 비중")
            fig_channel.update_layout(template="plotly_dark")
            st.plotly_chart(fig_channel, use_container_width=True, key="channel_pie")
            
    with col_s2:
        if '셀러명' in df_filtered.columns:
            seller_sales = df_filtered.groupby('셀러명')['결제금액(상품별)'].sum().nlargest(10).sort_values(ascending=True).reset_index()
            fig_seller = px.bar(seller_sales, x='결제금액(상품별)', y='셀러명', orientation='h', title="Top 10 셀러 매출")
            fig_seller.update_layout(template="plotly_dark")
            st.plotly_chart(fig_seller, use_container_width=True, key="seller_bar")
            
    # Detailed Data
    st.write("#### 채널/셀러 상세 성과")
    if '주문경로' in df_filtered.columns and '셀러명' in df_filtered.columns:
        pivot = df_filtered.pivot_table(index='셀러명', columns='주문경로', values='결제금액(상품별)', aggfunc='sum', fill_value=0)
        st.dataframe(pivot, use_container_width=True, key="seller_pivot")

with tab4:
    st.subheader("상품 상세 및 옵션 분석")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if '상품명_상세' in df_filtered.columns:
            detail_sales = df_filtered.groupby('상품명_상세')['결제금액(상품별)'].sum().nlargest(10).reset_index()
            fig_detail = px.bar(detail_sales, x='결제금액(상품별)', y='상품명_상세', orientation='h',
                             title="상품명 상세별 매출 Top 10")
            fig_detail.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_detail, use_container_width=True, key="detail_bar")
    
    with col_p2:
        if '옵션' in df_filtered.columns:
            option_sales = df_filtered.groupby('옵션')['결제금액(상품별)'].sum().nlargest(10).reset_index()
            fig_opt = px.bar(option_sales, x='결제금액(상품별)', y='옵션', orientation='h',
                             title="옵션별 매출 Top 10")
            fig_opt.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            fig_opt.update_traces(marker_color='#E91E63')
            st.plotly_chart(fig_opt, use_container_width=True, key="opt_bar")
        
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        if '등급' in df_filtered.columns:
            fig_grade = px.pie(df_filtered, values='결제금액(상품별)', names='등급', 
                               title="등급별 매출 점유율", hole=0.4)
            fig_grade.update_layout(template="plotly_dark")
            st.plotly_chart(fig_grade, use_container_width=True, key="grade_pie")
        
    with col_p4:
        st.write("#### 등급별 상세 데이터")
        grade_summary = df_filtered.groupby('등급')[['주문수량', '결제금액(상품별)']].agg(['sum', 'count'])
        st.dataframe(grade_summary, key="grade_df")

with tab5:
    st.subheader("기간 및 시간대 패턴 분석")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if '주문일' in df_filtered.columns:
            df_filtered['요일'] = df_filtered['주문일'].dt.day_name()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_sales = df_filtered.groupby('요일')['결제금액(상품별)'].sum().reindex(days_order).reset_index()
            fig_day = px.bar(day_sales, x='요일', y='결제금액(상품별)', title="요일별 매출")
            fig_day.update_layout(template="plotly_dark")
            st.plotly_chart(fig_day, use_container_width=True, key="day_bar")
        
    with col_t2:
        if '주문일' in df_filtered.columns:
            df_filtered['시간'] = df_filtered['주문일'].dt.hour
            hour_sales = df_filtered.groupby('시간')['주문수량'].sum().reset_index()
            fig_hour = px.line(hour_sales, x='시간', y='주문수량', title="시간대별 주문 패턴", markers=True)
            fig_hour.update_layout(template="plotly_dark", xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig_hour, use_container_width=True, key="hour_line")

with tab6:
    st.subheader("고객 분석 (VIP & RFM)")
    
    if '주문자연락처' in df_filtered.columns:
        cust_col = '주문자연락처'
    else:
        cust_col = '주문자명'
        
    if not df_filtered.empty:
        customer_stats = df_filtered.groupby(cust_col).agg({
            '주문일': lambda x: (df_filtered['주문일'].max() - x.max()).days,
            '주문번호': 'count',
            '결제금액(상품별)': 'sum'
        }).reset_index()
        customer_stats.columns = ['Customer', 'Recency(Days)', 'Frequency', 'Monetary']
        
        col_cust1, col_cust2 = st.columns([2, 1])
        
        with col_cust1:
            st.write("#### VIP 고객 Top 20 (매출 기준)")
            top_cust = customer_stats.sort_values(by='Monetary', ascending=False).head(20)
            st.dataframe(top_cust, use_container_width=True, key="vip_df")
            
        with col_cust2:
            st.write("#### 고객 분포")
            fig_hist = px.histogram(customer_stats[customer_stats['Monetary'] < 500000], x="Monetary", nbins=30, title="주문 금액 분포 (50만원 이하)")
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True, key="cust_hist")
