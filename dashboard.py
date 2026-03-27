import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np

# Page configuration
st.set_page_config(page_title="AP Dashboard", layout="wide")
st.title("📊 Accounts Payable Dashboard")
st.markdown("Interactive analysis of vendor invoices, payment status, and anomalies.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("accounts_payable_with_anomalies.csv", parse_dates=["InvoiceDate", "DueDate", "PaidDate"])
    # Convert PaidDate to datetime, handle NaT
    df["PaidDate"] = pd.to_datetime(df["PaidDate"], errors="coerce")
    # Calculate days overdue (only for open/partial invoices)
    today = date.today()
    df["DaysOverdue"] = (pd.to_datetime(today) - df["DueDate"]).dt.days
    df.loc[df["Status"] == "Paid", "DaysOverdue"] = 0
    df.loc[df["DaysOverdue"] < 0, "DaysOverdue"] = 0
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
vendor_filter = st.sidebar.multiselect("Vendor", options=df["Vendor"].unique(), default=df["Vendor"].unique())
status_filter = st.sidebar.multiselect("Status", options=df["Status"].unique(), default=df["Status"].unique())
currency_filter = st.sidebar.multiselect("Currency", options=df["Currency"].unique(), default=df["Currency"].unique())
anomaly_filter = st.sidebar.radio("Anomaly", options=["All", "Normal", "Anomaly"], index=0)

filtered_df = df[df["Vendor"].isin(vendor_filter) & df["Status"].isin(status_filter) & df["Currency"].isin(currency_filter)]
if anomaly_filter != "All":
    filtered_df = filtered_df[filtered_df["type"] == anomaly_filter]

# ==================== KPI Cards ====================
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_amount = filtered_df["amount_usd"].sum()
    st.metric("Total Payable (USD)", f"${total_amount:,.0f}")
with col2:
    open_invoices = filtered_df[filtered_df["Status"] == "Open"].shape[0]
    st.metric("Open Invoices", open_invoices)
with col3:
    paid_invoices = filtered_df[filtered_df["Status"] == "Paid"].shape[0]
    st.metric("Paid Invoices", paid_invoices)
with col4:
    anomaly_cnt = filtered_df[filtered_df["anomaly"] == True].shape[0]
    st.metric("Anomalies Detected", anomaly_cnt)

# ==================== Anomaly Explorer ====================
st.subheader("🚨 Anomaly Explorer")
if anomaly_cnt > 0:
    anomaly_df = filtered_df[filtered_df["anomaly"] == True]
    st.dataframe(anomaly_df[["APID", "Vendor", "InvoiceDate", "Amount", "Currency", "amount_usd", "Status", "type"]], use_container_width=True)
else:
    st.info("No anomalies in the current filtered data.")

# ==================== Vendor Analysis ====================
st.subheader("🏢 Vendor Analysis")
col1, col2 = st.columns(2)

with col1:
    vendor_amount = filtered_df.groupby("Vendor")["amount_usd"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(vendor_amount, x="Vendor", y="amount_usd", title="Total Payable by Vendor (USD)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    vendor_status = filtered_df.groupby(["Vendor", "Status"]).size().reset_index(name="count")
    fig = px.bar(vendor_status, x="Vendor", y="count", color="Status", title="Invoice Status by Vendor", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ==================== Currency Distribution ====================
st.subheader("💰 Currency Distribution")
col1, col2 = st.columns(2)
with col1:
    currency_original = filtered_df.groupby("Currency")["Amount"].sum().reset_index()
    fig = px.pie(currency_original, names="Currency", values="Amount", title="Amount by Original Currency")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    currency_usd = filtered_df.groupby("Currency")["amount_usd"].sum().reset_index()
    fig = px.pie(currency_usd, names="Currency", values="amount_usd", title="Amount in USD Equivalent")
    st.plotly_chart(fig, use_container_width=True)

# ==================== Invoice Trends ====================
st.subheader("📅 Invoice Trends")
# Aggregate by invoice month
monthly = filtered_df.groupby(filtered_df["InvoiceDate"].dt.to_period("M"))["amount_usd"].sum().reset_index()
monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
fig = px.line(monthly, x="InvoiceDate", y="amount_usd", title="Monthly Invoice Amount (USD)")
st.plotly_chart(fig, use_container_width=True)

# ==================== Aging Analysis ====================
st.subheader("⏳ Aging Analysis (Days Overdue)")
aging_data = filtered_df[filtered_df["Status"] != "Paid"].copy()
if not aging_data.empty:
    aging_data["Aging Bucket"] = pd.cut(aging_data["DaysOverdue"], bins=[-1, 0, 30, 60, 90, np.inf], labels=["Current", "1-30", "31-60", "61-90", "90+"])
    aging_summary = aging_data.groupby("Aging Bucket")["amount_usd"].sum().reset_index()
    fig = px.bar(aging_summary, x="Aging Bucket", y="amount_usd", title="Overdue Amount by Aging Bucket (USD)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No open or partial invoices in the filtered data.")

# ==================== Raw Data Explorer ====================
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Dashboard built with Streamlit | Data source: Accounts Payable dataset with anomaly detection")
