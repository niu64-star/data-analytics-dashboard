import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import traceback

# ==================== Page Configuration ====================
st.set_page_config(page_title="Integrated Business Dashboard", layout="wide")
st.title("📊 Integrated Executive Dashboard")
st.markdown("Consolidated view of Accounts Payable, ESG & Financial, and SalesMind analytics.")

# ==================== Helper Functions ====================
def check_file_exists(file_path):
    """Check if file exists and return True/False"""
    return os.path.exists(file_path)

def safe_dataframe_display(df, max_rows=100):
    """Safely display a dataframe, handling empty cases"""
    if df is None or df.empty:
        st.info("No data to display.")
        return
    st.dataframe(df.head(max_rows), use_container_width=True)

# ==================== Module 1: Accounts Payable Dashboard ====================
def show_ap_dashboard():
    st.header("📊 Accounts Payable Dashboard")
    st.markdown("Interactive analysis of vendor invoices, payment status, and anomalies.")

    file_path = "accounts_payable_with_anomalies.csv"
    if not check_file_exists(file_path):
        st.error(f"❌ File not found: `{file_path}`. Please upload the file to the current directory.")
        return

    try:
        # Load data
        @st.cache_data
        def load_ap_data():
            df = pd.read_csv(file_path, parse_dates=["InvoiceDate", "DueDate", "PaidDate"])
            # Ensure required columns exist
            required_cols = ['Vendor', 'Status', 'Currency', 'amount_usd', 'Amount', 'InvoiceDate', 'DueDate', 'anomaly', 'type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            df["PaidDate"] = pd.to_datetime(df["PaidDate"], errors="coerce")
            today = date.today()
            df["DaysOverdue"] = (pd.to_datetime(today) - df["DueDate"]).dt.days
            df.loc[df["Status"] == "Paid", "DaysOverdue"] = 0
            df.loc[df["DaysOverdue"] < 0, "DaysOverdue"] = 0
            return df

        df = load_ap_data()
    except Exception as e:
        st.error(f"Failed to load or process {file_path}: {e}")
        st.code(traceback.format_exc())
        return

    # Filters
    st.subheader("🔍 Filters")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        vendor_filter = st.multiselect("Vendor", options=df["Vendor"].unique(), default=df["Vendor"].unique())
    with col_f2:
        status_filter = st.multiselect("Status", options=df["Status"].unique(), default=df["Status"].unique())
    with col_f3:
        currency_filter = st.multiselect("Currency", options=df["Currency"].unique(), default=df["Currency"].unique())
    with col_f4:
        anomaly_filter = st.radio("Anomaly", options=["All", "Normal", "Anomaly"], index=0, horizontal=True)

    filtered_df = df[df["Vendor"].isin(vendor_filter) &
                     df["Status"].isin(status_filter) &
                     df["Currency"].isin(currency_filter)]
    if anomaly_filter != "All":
        filtered_df = filtered_df[filtered_df["type"] == anomaly_filter]

    # KPI Cards
    st.subheader("Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        total_amount = filtered_df["amount_usd"].sum()
        st.metric("Total Payable (USD)", f"${total_amount:,.0f}")
    with kpi2:
        open_invoices = filtered_df[filtered_df["Status"] == "Open"].shape[0]
        st.metric("Open Invoices", open_invoices)
    with kpi3:
        paid_invoices = filtered_df[filtered_df["Status"] == "Paid"].shape[0]
        st.metric("Paid Invoices", paid_invoices)
    with kpi4:
        anomaly_cnt = filtered_df[filtered_df["anomaly"] == True].shape[0]
        st.metric("Anomalies Detected", anomaly_cnt)

    # Anomaly Explorer
    st.subheader("🚨 Anomaly Explorer")
    if anomaly_cnt > 0:
        anomaly_df = filtered_df[filtered_df["anomaly"] == True]
        st.dataframe(anomaly_df[["APID", "Vendor", "InvoiceDate", "Amount", "Currency", "amount_usd", "Status", "type"]], use_container_width=True)
    else:
        st.info("No anomalies in the current filtered data.")

    # Vendor Analysis
    st.subheader("🏢 Vendor Analysis")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        vendor_amount = filtered_df.groupby("Vendor")["amount_usd"].sum().sort_values(ascending=False).reset_index()
        if not vendor_amount.empty:
            fig = px.bar(vendor_amount, x="Vendor", y="amount_usd", title="Total Payable by Vendor (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vendor data available.")
    with col_v2:
        vendor_status = filtered_df.groupby(["Vendor", "Status"]).size().reset_index(name="count")
        if not vendor_status.empty:
            fig = px.bar(vendor_status, x="Vendor", y="count", color="Status", title="Invoice Status by Vendor", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data available.")

    # Currency Distribution
    st.subheader("💰 Currency Distribution")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        currency_original = filtered_df.groupby("Currency")["Amount"].sum().reset_index()
        if not currency_original.empty:
            fig = px.pie(currency_original, names="Currency", values="Amount", title="Amount by Original Currency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No currency data available.")
    with col_c2:
        currency_usd = filtered_df.groupby("Currency")["amount_usd"].sum().reset_index()
        if not currency_usd.empty:
            fig = px.pie(currency_usd, names="Currency", values="amount_usd", title="Amount in USD Equivalent")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No USD data available.")

    # Invoice Trends
    st.subheader("📅 Invoice Trends")
    try:
        monthly = filtered_df.groupby(filtered_df["InvoiceDate"].dt.to_period("M"))["amount_usd"].sum().reset_index()
        monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
        if not monthly.empty:
            fig = px.line(monthly, x="InvoiceDate", y="amount_usd", title="Monthly Invoice Amount (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No invoice trend data available.")
    except Exception as e:
        st.warning(f"Could not generate invoice trends: {e}")

    # Aging Analysis
    st.subheader("⏳ Aging Analysis (Days Overdue)")
    aging_data = filtered_df[filtered_df["Status"] != "Paid"].copy()
    if not aging_data.empty:
        aging_data["Aging Bucket"] = pd.cut(aging_data["DaysOverdue"], bins=[-1, 0, 30, 60, 90, np.inf], labels=["Current", "1-30", "31-60", "61-90", "90+"])
        aging_summary = aging_data.groupby("Aging Bucket")["amount_usd"].sum().reset_index()
        if not aging_summary.empty:
            fig = px.bar(aging_summary, x="Aging Bucket", y="amount_usd", title="Overdue Amount by Aging Bucket (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No aging data available.")
    else:
        st.info("No open or partial invoices in the filtered data.")

    # Raw Data Explorer
    with st.expander("🔍 View Raw Data"):
        safe_dataframe_display(filtered_df)

    st.caption("Accounts Payable Dashboard — Data source: accounts_payable_with_anomalies.csv")

# ==================== Module 2: ESG & Financial Dashboard ====================
def show_esg_dashboard():
    st.header("🌍 ESG & Financial Performance Dashboard")
    st.markdown("Analyse the relationship between **ESG scores** and **financial metrics** across industries and regions.")

    file_path = "esg_analysis_full_data.csv"
    if not check_file_exists(file_path):
        st.error(f"❌ File not found: `{file_path}`. Please upload the file to the current directory.")
        return

    try:
        @st.cache_data
        def load_esg_data():
            df = pd.read_csv(file_path)
            # Check required columns
            required_cols = ['Year', 'Industry', 'Region', 'CompanyID', 'ESG_Overall', 'ProfitMargin', 'Revenue']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            numeric_cols = ['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
                            'ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
                            'CarbonEmissions', 'WaterUsage', 'EnergyConsumption']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        df = load_esg_data()
    except Exception as e:
        st.error(f"Failed to load or process {file_path}: {e}")
        st.code(traceback.format_exc())
        return

    # Filters
    st.subheader("🔍 Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        years = sorted(df['Year'].unique())
        year_range = st.slider("Select year range", min_value=min(years), max_value=max(years), value=(min(years), max(years)), step=1)
    with col_f2:
        industries = sorted(df['Industry'].unique())
        selected_industries = st.multiselect("Industry", industries, default=industries)
    with col_f3:
        regions = sorted(df['Region'].unique())
        selected_regions = st.multiselect("Region", regions, default=regions)

    filtered_df = df[(df['Year'].between(year_range[0], year_range[1])) &
                     (df['Industry'].isin(selected_industries)) &
                     (df['Region'].isin(selected_regions))]

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        return

    # Key metrics
    st.header("📊 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Average ESG Overall Score", f"{filtered_df['ESG_Overall'].mean():.1f}")
    with kpi2:
        st.metric("Average Profit Margin (%)", f"{filtered_df['ProfitMargin'].mean():.1f}")
    with kpi3:
        st.metric("Average Revenue (M)", f"{filtered_df['Revenue'].mean():,.0f}")
    with kpi4:
        st.metric("Number of Companies", f"{filtered_df['CompanyID'].nunique()}")

    # Tabs for visualisations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 ESG Trends", "💼 Financial Metrics", "🔍 Correlations", "🌍 Geographic View"])

    with tab1:
        st.subheader("ESG Score Evolution Over Time")
        try:
            yearly_esg = filtered_df.groupby(['Year', 'Industry'])[['ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance']].mean().reset_index()
            if not yearly_esg.empty:
                fig1 = px.line(yearly_esg, x='Year', y='ESG_Overall', color='Industry', title='Average ESG Overall Score by Industry')
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No ESG trend data available.")
        except Exception as e:
            st.warning(f"Could not create ESG trend chart: {e}")

        pillars = ['ESG_Environmental', 'ESG_Social', 'ESG_Governance']
        try:
            # Filter to existing pillars
            existing_pillars = [p for p in pillars if p in filtered_df.columns]
            if existing_pillars:
                fig2 = px.box(filtered_df.melt(id_vars=['Industry'], value_vars=existing_pillars, var_name='Pillar', value_name='Score'),
                              x='Pillar', y='Score', color='Industry', title='Distribution of ESG Pillar Scores')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No pillar data available.")
        except Exception as e:
            st.warning(f"Could not create pillar distribution chart: {e}")

    with tab2:
        st.subheader("Profit Margin vs. Revenue")
        try:
            fig3 = px.scatter(filtered_df, x='Revenue', y='ProfitMargin', color='ESG_Overall', size='MarketCap',
                              hover_data=['CompanyName', 'Industry', 'Year'],
                              title='Profit Margin vs. Revenue (size = Market Cap)')
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create scatter plot: {e}")

        st.subheader("Financial Performance Over Time")
        fin_metrics = ['Revenue', 'ProfitMargin', 'GrowthRate', 'MarketCap']
        for metric in fin_metrics:
            if metric in filtered_df.columns:
                try:
                    yearly_metric = filtered_df.groupby(['Year', 'Industry'])[metric].mean().reset_index()
                    if not yearly_metric.empty:
                        fig = px.line(yearly_metric, x='Year', y=metric, color='Industry', title=f'Average {metric} by Industry')
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create {metric} chart: {e}")

    with tab3:
        st.subheader("Correlation Heatmap")
        corr_cols = ['ESG_Overall', 'ProfitMargin', 'GrowthRate', 'Revenue', 'MarketCap',
                     'CarbonEmissions', 'WaterUsage', 'EnergyConsumption']
        existing_corr = [col for col in corr_cols if col in filtered_df.columns and filtered_df[col].notna().all()]
        if len(existing_corr) >= 2:
            try:
                corr_matrix = filtered_df[existing_corr].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                                     title="Correlation between ESG and Financial Indicators")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create heatmap: {e}")
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

        st.subheader("ESG Score vs. Profit Margin by Industry")
        try:
            fig5 = px.scatter(filtered_df, x='ESG_Overall', y='ProfitMargin', color='Industry', facet_col='Industry',
                              title='ESG Overall vs Profit Margin (faceted by Industry)', trendline='ols')
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create faceted scatter plot: {e}")

    with tab4:
        st.subheader("Regional Performance")
        region_stats = filtered_df.groupby('Region')[['ESG_Overall', 'ProfitMargin', 'Revenue']].mean().reset_index()
        if not region_stats.empty:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig6 = px.bar(region_stats, x='Region', y='ESG_Overall', title='Average ESG Score by Region', color='Region')
                st.plotly_chart(fig6, use_container_width=True)
            with col_r2:
                fig7 = px.bar(region_stats, x='Region', y='ProfitMargin', title='Average Profit Margin by Region', color='Region')
                st.plotly_chart(fig7, use_container_width=True)

            region_counts = filtered_df.groupby('Region')['CompanyID'].nunique().reset_index(name='Number of Companies')
            fig8 = px.bar(region_counts, x='Region', y='Number of Companies', title='Number of Companies per Region', color='Region')
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("No regional data available.")

    st.caption("ESG & Financial Dashboard — Data source: esg_analysis_full_data.csv")

# ==================== Module 3: SalesMind Executive Dashboard ====================
def show_salesmind_dashboard():
    st.header("📈 SalesMind Executive Dashboard")
    st.markdown("Sales performance, store & inventory, marketing insights, external factors, and anomaly detection.")

    # List of required files
    required_files = [
        'SalesMind_Sales_Transactions_2026.csv',
        'SalesMind_Stores_Master_2026.csv',
        'SalesMind_Products_Master_2026.csv',
        'SalesMind_Calendar_Dimension_2026.csv',
        'SalesMind_Customer_Segments_2026.csv',
        'SalesMind_External_Factors_2026.csv',
        'SalesMind_Inventory_Supply_2026.csv',
        'SalesMind_Marketing_Campaigns_2026.csv',
        'suspicious_transactions.csv'
    ]

    missing = [f for f in required_files if not check_file_exists(f)]
    if missing:
        st.error(f"❌ Missing files: {', '.join(missing)}. Please upload all required files.")
        return

    try:
        @st.cache_data
        def load_salesmind_data():
            sales = pd.read_csv('SalesMind_Sales_Transactions_2026.csv')
            stores = pd.read_csv('SalesMind_Stores_Master_2026.csv')
            products = pd.read_csv('SalesMind_Products_Master_2026.csv')
            calendar = pd.read_csv('SalesMind_Calendar_Dimension_2026.csv')
            customers = pd.read_csv('SalesMind_Customer_Segments_2026.csv')
            external = pd.read_csv('SalesMind_External_Factors_2026.csv')
            inventory = pd.read_csv('SalesMind_Inventory_Supply_2026.csv')
            campaigns = pd.read_csv('SalesMind_Marketing_Campaigns_2026.csv')
            suspicious = pd.read_csv('suspicious_transactions.csv')

            # Date conversions
            sales['date'] = pd.to_datetime(sales['date'])
            calendar['date'] = pd.to_datetime(calendar['date'])
            external['date'] = pd.to_datetime(external['date'])
            suspicious['InvoiceDate'] = pd.to_datetime(suspicious['InvoiceDate'])
            suspicious['DueDate'] = pd.to_datetime(suspicious['DueDate'])
            if 'PaidDate' in suspicious.columns:
                suspicious['PaidDate'] = pd.to_datetime(suspicious['PaidDate'], errors='coerce')

            # Merges
            df = sales.merge(stores, on='store_id', how='left')
            df = df.merge(products, on='product_id', how='left')
            df = df.merge(calendar, on='date', how='left')
            df = df.merge(customers, on='customer_segment', how='left')
            df = df.merge(external, on='date', how='left')
            df = df.merge(inventory, on=['store_id', 'product_id'], how='left')

            campaigns_by_date = campaigns.groupby('date')['marketing_spend'].sum().reset_index()
            campaigns_by_date['date'] = pd.to_datetime(campaigns_by_date['date'])
            df = df.merge(campaigns_by_date, on='date', how='left')
            df['marketing_spend'].fillna(0, inplace=True)

            return df, stores, products, calendar, customers, external, inventory, campaigns, suspicious

        df, stores, products, calendar, customers, external, inventory, campaigns, suspicious = load_salesmind_data()
    except Exception as e:
        st.error(f"Failed to load or process SalesMind data: {e}")
        st.code(traceback.format_exc())
        return

    # Filters
    st.subheader("🔍 Filters")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        start_date, end_date = st.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    df['date_only'] = df['date'].dt.date
    mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
    filtered_df = df.loc[mask].copy()

    with col_f2:
        store_types = ['All'] + sorted(filtered_df['store_type'].unique().tolist())
        selected_store_type = st.selectbox("Select Store Type", store_types)
    with col_f3:
        customer_segments = ['All'] + sorted(filtered_df['customer_segment'].unique().tolist())
        selected_customer = st.selectbox("Select Customer Segment", customer_segments)
    with col_f4:
        product_cats = ['All'] + sorted(filtered_df['product_category'].unique().tolist())
        selected_product_cat = st.selectbox("Select Product Category", product_cats)

    if selected_store_type != 'All':
        filtered_df = filtered_df[filtered_df['store_type'] == selected_store_type]
    if selected_customer != 'All':
        filtered_df = filtered_df[filtered_df['customer_segment'] == selected_customer]
    if selected_product_cat != 'All':
        filtered_df = filtered_df[filtered_df['product_category'] == selected_product_cat]

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        return

    # KPI Cards
    total_net_sales = filtered_df['net_sales'].sum()
    total_units_sold = filtered_df['units_sold'].sum()
    avg_discount_rate = (filtered_df['total_discount_given'].sum() / filtered_df['total_sales_revenue'].sum()) * 100 if filtered_df['total_sales_revenue'].sum() != 0 else 0
    avg_return_rate = filtered_df['return_rate'].mean() * 100
    gross_profit_margin = (filtered_df['gross_profit'].sum() / filtered_df['net_sales'].sum()) * 100 if filtered_df['net_sales'].sum() != 0 else 0
    stockout_count = inventory[(inventory['stockout_flag'] == 1) & (inventory['store_id'].isin(filtered_df['store_id'].unique()))]['inventory_id'].count()

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric(label="💰 Net Sales", value=f"${total_net_sales:,.0f}")
    with kpi2:
        st.metric(label="📦 Units Sold", value=f"{total_units_sold:,.0f}")
    with kpi3:
        st.metric(label="💸 Avg. Discount", value=f"{avg_discount_rate:.1f}%")
    with kpi4:
        st.metric(label="🔄 Return Rate", value=f"{avg_return_rate:.1f}%")
    with kpi5:
        st.metric(label="📈 Gross Profit Margin", value=f"{gross_profit_margin:.1f}%")
    with kpi6:
        st.metric(label="⚠️ Stockout Events", value=f"{stockout_count:,}")

    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sales Performance", "🏪 Store & Inventory", "🎯 Marketing & Customers", "🌍 External Factors", "🚨 Suspicious Transactions"])

    with tab1:
        st.subheader("Sales Performance Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                sales_over_time = filtered_df.groupby('date_only')['net_sales'].sum().reset_index()
                if not sales_over_time.empty:
                    fig_sales = px.line(sales_over_time, x='date_only', y='net_sales', title='Daily Net Sales')
                    st.plotly_chart(fig_sales, use_container_width=True)
                else:
                    st.info("No sales over time data.")
            except Exception as e:
                st.warning(f"Could not create sales trend: {e}")
        with col_b:
            try:
                sales_by_category = filtered_df.groupby('product_category')['net_sales'].sum().reset_index()
                if not sales_by_category.empty:
                    fig_category = px.pie(sales_by_category, values='net_sales', names='product_category', title='Net Sales by Product Category', hole=0.3)
                    st.plotly_chart(fig_category, use_container_width=True)
                else:
                    st.info("No category data.")
            except Exception as e:
                st.warning(f"Could not create category pie: {e}")
        col_c, col_d = st.columns(2)
        with col_c:
            try:
                sales_by_segment = filtered_df.groupby('customer_segment')['net_sales'].sum().reset_index()
                if not sales_by_segment.empty:
                    fig_segment = px.bar(sales_by_segment, x='customer_segment', y='net_sales', title='Net Sales by Customer Segment', color='customer_segment')
                    st.plotly_chart(fig_segment, use_container_width=True)
                else:
                    st.info("No segment data.")
            except Exception as e:
                st.warning(f"Could not create segment bar: {e}")
        with col_d:
            try:
                segment_metrics = filtered_df.groupby('customer_segment')[['return_rate', 'customer_satisfaction_score']].mean().reset_index()
                if not segment_metrics.empty:
                    fig_satisfaction = px.scatter(segment_metrics, x='return_rate', y='customer_satisfaction_score', size='return_rate', color='customer_segment',
                                                  title='Return Rate vs. Customer Satisfaction')
                    st.plotly_chart(fig_satisfaction, use_container_width=True)
                else:
                    st.info("No satisfaction data.")
            except Exception as e:
                st.warning(f"Could not create satisfaction scatter: {e}")

    with tab2:
        st.subheader("Store & Inventory Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                sales_by_store_type = filtered_df.groupby('store_type')['net_sales'].sum().reset_index()
                if not sales_by_store_type.empty:
                    fig_store = px.bar(sales_by_store_type, x='store_type', y='net_sales', title='Net Sales by Store Type', color='store_type')
                    st.plotly_chart(fig_store, use_container_width=True)
                else:
                    st.info("No store type data.")
            except Exception as e:
                st.warning(f"Could not create store type bar: {e}")

            try:
                stockout_by_product = inventory[inventory['stockout_flag'] == 1].groupby('product_id').size().reset_index(name='stockout_count')
                stockout_by_product = stockout_by_product.merge(products[['product_id', 'product_category']], on='product_id', how='left')
                top_stockout = stockout_by_product.nlargest(10, 'stockout_count')
                if not top_stockout.empty:
                    fig_stockout = px.bar(top_stockout, x='product_id', y='stockout_count', title='Top 10 Products by Stockout Events', color='product_category')
                    st.plotly_chart(fig_stockout, use_container_width=True)
                else:
                    st.info("No stockout data.")
            except Exception as e:
                st.warning(f"Could not create stockout chart: {e}")
        with col_b:
            try:
                avg_inventory = inventory.groupby('store_id')['inventory_level'].mean().reset_index()
                store_sales = filtered_df.groupby('store_id')['net_sales'].sum().reset_index()
                store_turnover = store_sales.merge(avg_inventory, on='store_id')
                store_turnover['turnover_rate'] = store_turnover['net_sales'] / store_turnover['inventory_level']
                if not store_turnover.empty:
                    fig_turnover = px.bar(store_turnover, x='store_id', y='turnover_rate', title='Store Turnover Rate (Net Sales / Avg Inventory)', color='turnover_rate')
                    st.plotly_chart(fig_turnover, use_container_width=True)
                else:
                    st.info("No turnover data.")
            except Exception as e:
                st.warning(f"Could not create turnover chart: {e}")

    with tab3:
        st.subheader("Marketing & Customer Insights")
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                campaign_roi = campaigns.groupby('ad_channel')[['marketing_spend', 'conversion_rate', 'impressions']].mean().reset_index()
                campaign_roi['roi'] = campaign_roi['conversion_rate'] / (campaign_roi['marketing_spend'] / campaign_roi['impressions']) * 1000
                if not campaign_roi.empty:
                    fig_roi = px.bar(campaign_roi, x='ad_channel', y='roi', title='Average ROI by Ad Channel', color='ad_channel')
                    st.plotly_chart(fig_roi, use_container_width=True)
                else:
                    st.info("No ROI data.")
            except Exception as e:
                st.warning(f"Could not create ROI chart: {e}")

            try:
                segment_data = customers[['customer_segment', 'churn_rate', 'loyalty_member_ratio']]
                if not segment_data.empty:
                    fig_loyalty = px.scatter(segment_data, x='loyalty_member_ratio', y='churn_rate', size='churn_rate', color='customer_segment',
                                             title='Loyalty Members vs. Churn Rate')
                    st.plotly_chart(fig_loyalty, use_container_width=True)
                else:
                    st.info("No loyalty data.")
            except Exception as e:
                st.warning(f"Could not create loyalty scatter: {e}")
        with col_b:
            try:
                fig_conversion = px.scatter(campaigns, x='marketing_spend', y='conversion_rate', size='impressions', color='ad_channel',
                                            title='Marketing Spend vs. Conversion Rate')
                st.plotly_chart(fig_conversion, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create conversion scatter: {e}")

    with tab4:
        st.subheader("External Factors Impact")
        try:
            external_merged = filtered_df.groupby('date_only')[['net_sales', 'inflation_rate', 'competitor_price']].mean().reset_index()
            external_merged = external_merged.melt(id_vars='date_only', value_vars=['net_sales', 'inflation_rate', 'competitor_price'], var_name='Metric', value_name='Value')
            if not external_merged.empty:
                fig_ext = px.line(external_merged, x='date_only', y='Value', color='Metric', title='Net Sales vs. Inflation Rate vs. Competitor Price Over Time')
                st.plotly_chart(fig_ext, use_container_width=True)
            else:
                st.info("No external factor data.")
        except Exception as e:
            st.warning(f"Could not create external factors chart: {e}")

        try:
            sales_by_weather = filtered_df.groupby('weather_condition')['net_sales'].sum().reset_index()
            if not sales_by_weather.empty:
                fig_weather = px.bar(sales_by_weather, x='weather_condition', y='net_sales', title='Net Sales by Weather Condition', color='weather_condition')
                st.plotly_chart(fig_weather, use_container_width=True)
            else:
                st.info("No weather data.")
        except Exception as e:
            st.warning(f"Could not create weather chart: {e}")

    with tab5:
        st.subheader("🚨 Anomaly Detection in Accounts Payable")
        st.markdown("This table highlights suspicious transactions detected in the AP system based on size, payment delay, or other rules.")
        if not suspicious.empty:
            try:
                st.dataframe(suspicious[['APID', 'Vendor', 'InvoiceDate', 'DueDate', 'Amount', 'Currency', 'Status', 'amount_usd']], use_container_width=True)
                anomaly_by_vendor = suspicious.groupby('Vendor')['amount_usd'].sum().reset_index().nlargest(10, 'amount_usd')
                if not anomaly_by_vendor.empty:
                    fig_anomaly = px.bar(anomaly_by_vendor, x='Vendor', y='amount_usd', title='Top 10 Vendors with Highest Suspicious Amounts (USD)', color='amount_usd')
                    st.plotly_chart(fig_anomaly, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display suspicious transactions: {e}")
        else:
            st.info("No suspicious transactions found in the selected period.")

    st.caption("SalesMind Dashboard — Data sources: SalesMind_*_2026.csv and suspicious_transactions.csv")

# ==================== Main App ====================
def main():
    # Tabs for each dashboard module
    tab_ap, tab_esg, tab_sales = st.tabs(["📑 Accounts Payable", "🌱 ESG & Financial", "📊 SalesMind Executive"])

    with tab_ap:
        show_ap_dashboard()
    with tab_esg:
        show_esg_dashboard()
    with tab_sales:
        show_salesmind_dashboard()

    st.markdown("---")
    st.caption("Integrated Dashboard — Built with Streamlit | All modules combined from three original dashboards.")

if __name__ == "__main__":
    main()
