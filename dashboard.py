import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ========================
# 页面配置
# ========================
st.set_page_config(page_title="Data Analytics Dashboard", layout="wide")
st.title("📊 数据分析仪表板")
st.markdown("基于 ACCFIN 5254 论文的交互式展示")

# ========================
# 侧边栏：选择分析模块
# ========================
analysis = st.sidebar.radio(
    "选择分析问题",
    ["🔍 应付账款异常检测", "🌱 ESG与财务绩效", "📢 营销渠道效果"]
)

# ========================
# 1. 读取数据（适配你的文件名）
# ========================

@st.cache_data
def load_fraud_data():
    """读取应付账款数据"""
    try:
        df = pd.read_csv("accounts_payable_with_anomalies.csv")
        # 确保日期列存在
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # 如果没有日期列，创建模拟日期（仅用于演示）
            st.warning("数据中没有 'Date' 列，将使用索引生成模拟日期")
            df['Date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
        # 确保金额列为数值
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        else:
            st.error("数据中缺少 'Amount' 列，无法进行异常检测")
            return None
        return df
    except FileNotFoundError:
        st.error("未找到 accounts_payable_with_anomalies.csv，请将文件放在当前目录下。")
        return None

@st.cache_data
def load_esg_data():
    """读取 ESG 数据，自动识别列名"""
    try:
        df = pd.read_csv("company_esg_financial_dataset.csv")
        # 尝试找到 ESG 得分列
        esg_col = None
        for col in ['ESG_Score', 'Overall_ESG', 'ESG', 'esg_score', 'Overall ESG']:
            if col in df.columns:
                esg_col = col
                break
        # 尝试找到利润列
        profit_col = None
        for col in ['Profit_Margin', 'Profit Margin', 'profit_margin', 'Net Profit Margin']:
            if col in df.columns:
                profit_col = col
                break
        if esg_col is None or profit_col is None:
            st.warning("未找到 ESG 或利润列，将使用模拟数据演示。")
            # 模拟数据
            np.random.seed(42)
            df = pd.DataFrame({
                'ESG_Score': np.random.uniform(30, 90, 1000),
                'Profit_Margin': np.random.uniform(5, 30, 1000)
            })
            df['Profit_Margin'] = df['Profit_Margin'] + 0.2 * (df['ESG_Score'] - 50)
            esg_col = 'ESG_Score'
            profit_col = 'Profit_Margin'
        return df, esg_col, profit_col
    except FileNotFoundError:
        st.error("未找到 company_esg_financial_dataset.csv，请将文件放在当前目录下。")
        return None, None, None

@st.cache_data
def load_marketing_data():
    """读取营销数据，优先使用 SalesMind_Marketing_Campaigns_2026.csv"""
    try:
        df = pd.read_csv("SalesMind_Marketing_Campaigns_2026.csv")
        # 检查必要列
        required = ['Channel', 'Spend', 'Revenue', 'Clicks', 'Conversions']
        if all(col in df.columns for col in required):
            return df
        else:
            st.warning("营销数据缺少必要列，将使用模拟数据。")
            return None
    except FileNotFoundError:
        st.warning("未找到 SalesMind_Marketing_Campaigns_2026.csv，使用模拟数据。")
        return None

# ========================
# 2. 分析函数
# ========================

def run_anomaly_detection(df):
    """使用 Isolation Forest 识别异常"""
    # 特征工程
    df_temp = df.copy()
    if 'Date' in df_temp.columns and df_temp['Date'].notna().any():
        df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
        df_temp['Month'] = df_temp['Date'].dt.month
    else:
        df_temp['DayOfWeek'] = 0
        df_temp['Month'] = 0
    
    features = ['Amount', 'DayOfWeek', 'Month']
    X = df_temp[features].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df_temp['Anomaly'] = iso_forest.fit_predict(X)
    df_temp['Anomaly'] = df_temp['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df_temp

def run_esg_analysis(df, esg_col, profit_col):
    """计算相关性并回归"""
    # 相关性
    corr = df[[esg_col, profit_col]].corr().iloc[0,1]
    # 回归
    X = df[esg_col].values.reshape(-1,1)
    y = df[profit_col].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    return corr, slope, intercept, r2

def run_marketing_analysis(df):
    """计算渠道汇总指标"""
    if df is None:
        # 生成模拟数据（与论文一致）
        np.random.seed(42)
        channels = ['Email', 'Paid Search', 'Social Media']
        n_campaigns = 15
        data = []
        for i in range(n_campaigns):
            channel = np.random.choice(channels)
            if channel == 'Email':
                spend = np.random.uniform(500, 2000)
                revenue = spend * np.random.uniform(2.5, 4.0)
                clicks = np.random.randint(500, 3000)
                conversions = np.random.randint(80, 450)
            elif channel == 'Paid Search':
                spend = np.random.uniform(1000, 2500)
                revenue = spend * np.random.uniform(1.8, 2.8)
                clicks = np.random.randint(300, 2000)
                conversions = np.random.randint(50, 280)
            else:
                spend = np.random.uniform(1500, 3500)
                revenue = spend * np.random.uniform(1.2, 2.0)
                clicks = np.random.randint(1000, 6000)
                conversions = np.random.randint(40, 200)
            data.append({
                'Channel': channel,
                'Spend': spend,
                'Revenue': revenue,
                'Clicks': clicks,
                'Conversions': conversions,
                'Campaign': f"{channel}_{i+1}"
            })
        df = pd.DataFrame(data)
    # 计算衍生指标
    df['ROI'] = (df['Revenue'] - df['Spend']) / df['Spend'] * 100
    df['Conversion_Rate'] = df['Conversions'] / df['Clicks'] * 100
    # 如果数据中没有 CTR，则按渠道平均值估算
    if 'CTR' not in df.columns:
        ctr_map = {'Email': 4.3, 'Paid Search': 2.8, 'Social Media': 5.2}
        df['CTR'] = df['Channel'].map(ctr_map)
    # 汇总
    summary = df.groupby('Channel').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    summary['ROI'] = (summary['Revenue'] - summary['Spend']) / summary['Spend'] * 100
    summary['Conversion_Rate'] = summary['Conversions'] / summary['Clicks'] * 100
    summary['CPA'] = summary['Spend'] / summary['Conversions']
    # 添加 CTR 汇总
    ctr_mean = df.groupby('Channel')['CTR'].mean().reset_index()
    summary = summary.merge(ctr_mean, on='Channel')
    return summary, df

# ========================
# 3. 根据选择展示内容
# ========================

if analysis == "🔍 应付账款异常检测":
    st.header("🔍 应付账款异常检测")
    st.markdown("**方法：** 无监督学习（Isolation Forest），识别与常规交易模式不同的异常支付。")
    
    df_raw = load_fraud_data()
    if df_raw is not None:
        df = run_anomaly_detection(df_raw)
        anomalies = df[df['Anomaly'] == 'Anomaly']
        st.success(f"共检测到 {len(anomalies)} 笔异常交易（占总交易 {len(anomalies)/len(df)*100:.1f}%）")
        
        # 散点图：金额随时间分布
        fig1 = px.scatter(df, x='Date', y='Amount', color='Anomaly',
                          title='交易金额随时间分布（异常高亮）',
                          color_discrete_map={'Normal':'blue', 'Anomaly':'red'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # 箱线图
        fig2 = px.box(df, x='Anomaly', y='Amount', color='Anomaly',
                      title='异常交易 vs 正常交易金额分布')
        st.plotly_chart(fig2, use_container_width=True)
        
        # 供应商分布（如果有 Vendor 列）
        if 'Vendor' in df.columns:
            vendor_count = anomalies['Vendor'].value_counts().reset_index()
            vendor_count.columns = ['Vendor', 'Anomaly Count']
            fig3 = px.bar(vendor_count, x='Vendor', y='Anomaly Count',
                          title='异常交易按供应商分布')
            st.plotly_chart(fig3, use_container_width=True)
        
        # 直方图
        fig4 = px.histogram(df, x='Amount', color='Anomaly', nbins=50,
                            title='金额分布：异常 vs 正常',
                            barmode='overlay')
        st.plotly_chart(fig4, use_container_width=True)
        
        # 月度趋势（如果有日期）
        if 'Date' in df.columns:
            monthly = df.groupby([df['Date'].dt.to_period('M'), 'Anomaly']).size().reset_index(name='Count')
            monthly['Date'] = monthly['Date'].astype(str)
            fig5 = px.line(monthly, x='Date', y='Count', color='Anomaly',
                           title='月度交易数量趋势')
            st.plotly_chart(fig5, use_container_width=True)

elif analysis == "🌱 ESG与财务绩效":
    st.header("🌱 ESG与财务绩效")
    st.markdown("**方法：** 相关性分析 + 线性回归，探究ESG得分与盈利能力的关系。")
    
    df, esg_col, profit_col = load_esg_data()
    if df is not None:
        corr, slope, intercept, r2 = run_esg_analysis(df, esg_col, profit_col)
        st.success(f"相关性系数 r = {corr:.3f}，回归斜率 = {slope:.4f}，R² = {r2:.3f}")
        
        # 散点图 + 回归线
        fig = px.scatter(df, x=esg_col, y=profit_col,
                         title=f"{esg_col} vs {profit_col}",
                         trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # 相关性热力图（可选）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 title="关键指标相关性矩阵")
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # 行业分析（如果有 Industry 列）
        if 'Industry' in df.columns:
            industry_esg = df.groupby('Industry')[[esg_col, profit_col]].mean().reset_index()
            fig_ind = px.bar(industry_esg, x='Industry', y=esg_col,
                             title="各行业平均ESG得分")
            st.plotly_chart(fig_ind, use_container_width=True)
        
        # 时间趋势（如果有 Year 列）
        if 'Year' in df.columns:
            trend = df.groupby('Year')[[esg_col, profit_col]].mean().reset_index()
            fig_trend = px.line(trend, x='Year', y=[esg_col, profit_col],
                                title="ESG得分与利润率年度趋势")
            st.plotly_chart(fig_trend, use_container_width=True)

else:  # 营销渠道效果
    st.header("📢 营销渠道效果")
    st.markdown("**方法：** 描述性统计 + ROI/CPA分析，评估各渠道投资回报率。")
    
    df_mkt = load_marketing_data()
    summary, df_details = run_marketing_analysis(df_mkt)
    
    st.success(f"共分析 {len(df_details)} 个营销活动，涵盖 {len(summary)} 个渠道")
    
    # ROI 对比
    fig1 = px.bar(summary, x='Channel', y='ROI', title="ROI 对比",
                  text_auto='.1f', color='Channel')
    st.plotly_chart(fig1, use_container_width=True)
    
    # 支出 vs 收入
    fig2 = px.bar(summary, x='Channel', y=['Spend', 'Revenue'],
                  title="营销支出 vs 收入", barmode='group')
    st.plotly_chart(fig2, use_container_width=True)
    
    # 转化率
    fig3 = px.bar(summary, x='Channel', y='Conversion_Rate',
                  title="转化率对比", text_auto='.1f')
    st.plotly_chart(fig3, use_container_width=True)
    
    # CPA
    fig4 = px.bar(summary, x='Channel', y='CPA',
                  title="单次获取成本 (CPA)")
    st.plotly_chart(fig4, use_container_width=True)
    
    # 散点图：支出 vs 收入
    fig5 = px.scatter(df_details, x='Spend', y='Revenue', color='Channel',
                      size='Clicks', hover_name='Campaign',
                      title="各活动支出与收入关系")
    st.plotly_chart(fig5, use_container_width=True)
    
    # 互动指标：CTR 与转化率
    fig6 = px.scatter(df_details, x='CTR', y='Conversion_Rate', color='Channel',
                      size='Spend', title="CTR 与 转化率 关系")
    st.plotly_chart(fig6, use_container_width=True)