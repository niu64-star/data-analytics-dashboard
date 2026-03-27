import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Data Analytics Dashboard", layout="wide")
st.title("📊 Data Analytics Dashboard")
st.markdown("Interactive presentation based on ACCFIN 5254 paper")

# Sidebar: select analysis
analysis = st.sidebar.radio(
    "Select Analysis",
    ["🔍 Accounts Payable Anomaly Detection", "🌱 ESG & Financial Performance", "📢 Marketing Channel Effectiveness"]
)

# Data loading functions (unchanged)
@st.cache_data
def load_fraud_data():
    try:
        df = pd.read_csv("accounts_payable_with_anomalies.csv")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            st.warning("No 'Date' column found, using simulated dates.")
            df['Date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        else:
            st.error("Missing 'Amount' column.")
            return None
        return df
    except FileNotFoundError:
        st.error("File 'accounts_payable_with_anomalies.csv' not found.")
        return None

@st.cache_data
def load_esg_data():
    try:
        df = pd.read_csv("company_esg_financial_dataset.csv")
        esg_col = None
        for col in ['ESG_Score', 'Overall_ESG', 'ESG', 'esg_score', 'Overall ESG']:
            if col in df.columns:
                esg_col = col
                break
        profit_col = None
        for col in ['Profit_Margin', 'Profit Margin', 'profit_margin', 'Net Profit Margin']:
            if col in df.columns:
                profit_col = col
                break
        if esg_col is None or profit_col is None:
            st.warning("ESG or profit column not found, using simulated data for demonstration.")
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
        st.error("File 'company_esg_financial_dataset.csv' not found.")
        return None, None, None

@st.cache_data
def load_marketing_data():
    try:
        df = pd.read_csv("SalesMind_Marketing_Campaigns_2026.csv")
        required = ['Channel', 'Spend', 'Revenue', 'Clicks', 'Conversions']
        if all(col in df.columns for col in required):
            return df
        else:
            st.warning("Marketing data missing required columns, using simulated data.")
            return None
    except FileNotFoundError:
        st.warning("File 'SalesMind_Marketing_Campaigns_2026.csv' not found, using simulated data.")
        return None

# Analysis functions (unchanged)
def run_anomaly_detection(df):
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
    corr = df[[esg_col, profit_col]].corr().iloc[0,1]
    X = df[esg_col].values.reshape(-1,1)
    y = df[profit_col].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    return corr, slope, intercept, r2

def run_marketing_analysis(df):
    if df is None:
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
    df['ROI'] = (df['Revenue'] - df['Spend']) / df['Spend'] * 100
    df['Conversion_Rate'] = df['Conversions'] / df['Clicks'] * 100
    if 'CTR' not in df.columns:
        ctr_map = {'Email': 4.3, 'Paid Search': 2.8, 'Social Media': 5.2}
        df['CTR'] = df['Channel'].map(ctr_map)
    summary = df.groupby('Channel').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Clicks': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    summary['ROI'] = (summary['Revenue'] - summary['Spend']) / summary['Spend'] * 100
    summary['Conversion_Rate'] = summary['Conversions'] / summary['Clicks'] * 100
    summary['CPA'] = summary['Spend'] / summary['Conversions']
    ctr_mean = df.groupby('Channel')['CTR'].mean().reset_index()
    summary = summary.merge(ctr_mean, on='Channel')
    return summary, df

# Display based on selection
if analysis == "🔍 Accounts Payable Anomaly Detection":
    st.header("🔍 Accounts Payable Anomaly Detection")
    st.markdown("**Method:** Unsupervised Learning (Isolation Forest) to identify unusual payment patterns.")
    df_raw = load_fraud_data()
    if df_raw is not None:
        df = run_anomaly_detection(df_raw)
        anomalies = df[df['Anomaly'] == 'Anomaly']
        st.success(f"Detected {len(anomalies)} anomalous transactions ({len(anomalies)/len(df)*100:.1f}% of total)")
        fig1 = px.scatter(df, x='Date', y='Amount', color='Anomaly',
                          title='Transaction Amount Over Time (Anomalies Highlighted)',
                          color_discrete_map={'Normal':'blue', 'Anomaly':'red'})
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(df, x='Anomaly', y='Amount', color='Anomaly',
                      title='Amount Distribution: Normal vs Anomaly')
        st.plotly_chart(fig2, use_container_width=True)
        if 'Vendor' in df.columns:
            vendor_count = anomalies['Vendor'].value_counts().reset_index()
            vendor_count.columns = ['Vendor', 'Anomaly Count']
            fig3 = px.bar(vendor_count, x='Vendor', y='Anomaly Count',
                          title='Anomaly Distribution by Vendor')
            st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.histogram(df, x='Amount', color='Anomaly', nbins=50,
                            title='Amount Distribution Comparison',
                            barmode='overlay')
        st.plotly_chart(fig4, use_container_width=True)
        if 'Date' in df.columns:
            monthly = df.groupby([df['Date'].dt.to_period('M'), 'Anomaly']).size().reset_index(name='Count')
            monthly['Date'] = monthly['Date'].astype(str)
            fig5 = px.line(monthly, x='Date', y='Count', color='Anomaly',
                           title='Monthly Transaction Trend')
            st.plotly_chart(fig5, use_container_width=True)

elif analysis == "🌱 ESG & Financial Performance":
    st.header("🌱 ESG & Financial Performance")
    st.markdown("**Method:** Correlation analysis + Linear regression to explore relationship between ESG scores and profitability.")
    df, esg_col, profit_col = load_esg_data()
    if df is not None:
        corr, slope, intercept, r2 = run_esg_analysis(df, esg_col, profit_col)
        st.success(f"Correlation coefficient r = {corr:.3f}, regression slope = {slope:.4f}, R² = {r2:.3f}")
        fig = px.scatter(df, x=esg_col, y=profit_col,
                         title=f"{esg_col} vs {profit_col}",
                         trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig, use_container_width=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 title="Correlation Matrix of Key Indicators")
            st.plotly_chart(fig_heat, use_container_width=True)
        if 'Industry' in df.columns:
            industry_esg = df.groupby('Industry')[[esg_col, profit_col]].mean().reset_index()
            fig_ind = px.bar(industry_esg, x='Industry', y=esg_col,
                             title="Average ESG Score by Industry")
            st.plotly_chart(fig_ind, use_container_width=True)
        if 'Year' in df.columns:
            trend = df.groupby('Year')[[esg_col, profit_col]].mean().reset_index()
            fig_trend = px.line(trend, x='Year', y=[esg_col, profit_col],
                                title="Annual Trends in ESG Score and Profit Margin")
            st.plotly_chart(fig_trend, use_container_width=True)

else:  # Marketing
    st.header("📢 Marketing Channel Effectiveness")
    st.markdown("**Method:** Descriptive analytics + ROI/CPA analysis to evaluate channel performance.")
    df_mkt = load_marketing_data()
    summary, df_details = run_marketing_analysis(df_mkt)
    st.success(f"Analyzed {len(df_details)} campaigns across {len(summary)} channels")
    fig1 = px.bar(summary, x='Channel', y='ROI', title="ROI Comparison",
                  text_auto='.1f', color='Channel')
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(summary, x='Channel', y=['Spend', 'Revenue'],
                  title="Marketing Spend vs Revenue", barmode='group')
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.bar(summary, x='Channel', y='Conversion_Rate',
                  title="Conversion Rate Comparison", text_auto='.1f')
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.bar(summary, x='Channel', y='CPA',
                  title="Cost Per Acquisition (CPA)")
    st.plotly_chart(fig4, use_container_width=True)
    fig5 = px.scatter(df_details, x='Spend', y='Revenue', color='Channel',
                      size='Clicks', hover_name='Campaign',
                      title="Campaign Spend vs Revenue")
    st.plotly_chart(fig5, use_container_width=True)
    fig6 = px.scatter(df_details, x='CTR', y='Conversion_Rate', color='Channel',
                      size='Spend', title="CTR vs Conversion Rate")
    st.plotly_chart(fig6, use_container_width=True)
