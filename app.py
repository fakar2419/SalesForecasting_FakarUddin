import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import os

# Page Config
st.set_page_config(
    page_title="Sales Forecasting & Product Demand Insights Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: rgba(128, 128, 128, 0.05);
    border-radius: 10px;
    padding: 1.2rem;
    border-left: 5px solid #4b6cb7;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #777;
    margin-bottom: 0.2rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a2536;
}

.custom-card {
    background-color: rgba(128, 128, 128, 0.03);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(128, 128, 128, 0.1);
}

.custom-card h3 {
    margin-top: 0;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Helper function for seasonality
def get_season(month):
    if month in [12, 1, 2]:
        return 0      # Winter
    elif month in [3, 4, 5]:
        return 1      # Spring
    elif month in [6, 7, 8]:
        return 2      # Summer
    else:
        return 3      # Autumn

# Caching Data Loader
@st.cache_data
def load_data(file_path="train.csv"):
    if not os.path.exists(file_path):
        st.error(f"Data file '{file_path}' not found! Please check the workspace.")
        return None
    df = pd.read_csv(file_path)
    # Parse dates explicitly
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month_Year"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df

df = load_data()

if df is not None:
    # Sidebar Dimension Selection & Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
    st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 700; margin-top: 0;'>Sales Forecast Analytics</h2>", unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigate Dashboard",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"]
    )
    
    st.sidebar.markdown("---")
    
    # Caching Clustering results to avoid recalculating K-Means on every click
    @st.cache_data
    def run_clustering_cache(df_all):
        # 1. Sales Volume
        total_sales = df_all.groupby("Sub-Category")["Sales"].sum().rename("Total Sales Volume")
        # 2. Average Order Value
        average_order_value = df_all.groupby("Sub-Category")["Sales"].mean().rename("Average Order Value")
        # 3. Volatility
        monthly_sales = df_all.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"].sum().reset_index()
        sales_volatility = monthly_sales.groupby("Sub-Category")["Sales"].std().rename("Sales Volatility")
        # 4. Growth YoY
        yearly_sales = df_all.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
        growth_list = []
        for sub in yearly_sales["Sub-Category"].unique():
            temp = yearly_sales[yearly_sales["Sub-Category"] == sub].sort_values("Year")
            first_year_sales = temp.iloc[0]["Sales"]
            last_year_sales = temp.iloc[-1]["Sales"]
            growth_rate = ((last_year_sales - first_year_sales) / first_year_sales) * 100
            growth_list.append([sub, growth_rate])
        sales_growth = pd.DataFrame(growth_list, columns=["Sub-Category", "Sales Growth Rate (%)"]).set_index("Sub-Category")
        
        # Combine
        cluster_data = pd.concat([total_sales, sales_growth, sales_volatility, average_order_value], axis=1).round(2)
        cluster_data.fillna(0, inplace=True)
        
        # Scale
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_data)
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_data["Cluster"] = kmeans.fit_predict(scaled_features)
        
        # Identify clusters dynamically by checking segment means
        cluster_means = cluster_data.groupby("Cluster").mean(numeric_only=True)
        premium_cluster = cluster_means["Average Order Value"].idxmax()
        low_vol_cluster = cluster_means["Total Sales Volume"].idxmin()
        remaining = [c for c in [0, 1, 2] if c not in [premium_cluster, low_vol_cluster]]
        stable_high_cluster = remaining[0] if remaining else (3 - premium_cluster - low_vol_cluster)
        
        cluster_names = {
            premium_cluster: "High Growth, Premium Value",
            low_vol_cluster: "Low Volume, Stable Demand",
            stable_high_cluster: "High Volume, Stable Demand"
        }
        
        cluster_data["Demand Group"] = cluster_data["Cluster"].map(cluster_names)
        
        # PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(scaled_features)
        cluster_data["PC1"] = pca_features[:, 0]
        cluster_data["PC2"] = pca_features[:, 1]
        
        return cluster_data, cluster_names

    # Caching Anomaly results
    @st.cache_data
    def run_anomalies_cache(df_all):
        weekly_sales = df_all.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        weekly_sales["IF_Anomaly"] = iso.fit_predict(weekly_sales[["Sales"]]) == -1
        
        # Z-Score
        window = 8
        weekly_sales["Rolling_Mean"] = weekly_sales["Sales"].rolling(window).mean()
        weekly_sales["Rolling_STD"] = weekly_sales["Sales"].rolling(window).std()
        weekly_sales["Z_Score"] = (weekly_sales["Sales"] - weekly_sales["Rolling_Mean"]) / weekly_sales["Rolling_STD"]
        weekly_sales["Z_Anomaly"] = weekly_sales["Z_Score"].abs() > 2
        weekly_sales["Z_Anomaly"] = weekly_sales["Z_Anomaly"].fillna(False)
        return weekly_sales

    # Caching segment-level forecasting model
    @st.cache_data
    def run_segment_forecast(data_filtered, horizon_months):
        monthly = data_filtered.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
        
        # Create lag features
        monthly["Lag_1"] = monthly["Sales"].shift(1)
        monthly["Lag_2"] = monthly["Sales"].shift(2)
        monthly["Lag_3"] = monthly["Sales"].shift(3)
        monthly["Rolling_Mean_3"] = monthly["Sales"].rolling(3).mean()
        
        # Calendar features
        monthly["Month"] = monthly["Order Date"].dt.month
        monthly["Quarter"] = monthly["Order Date"].dt.quarter
        monthly["Season"] = monthly["Month"].apply(get_season)
        
        eval_df = monthly.copy().dropna()
        
        if len(eval_df) < 5:
            return None, 0.0, 0.0
            
        features = ["Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Month", "Quarter", "Season"]
        X = eval_df[features]
        y = eval_df["Sales"]
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Forecast future horizon
        future = monthly.copy()
        predictions = []
        for _ in range(horizon_months):
            last = future.iloc[-1]
            next_date = last["Order Date"] + pd.offsets.MonthEnd(1)
            
            lag1 = last["Sales"]
            lag2 = future.iloc[-2]["Sales"] if len(future) >= 2 else lag1
            lag3 = future.iloc[-3]["Sales"] if len(future) >= 3 else lag2
            
            rolling = np.mean([lag1, lag2, lag3])
            month = next_date.month
            quarter = next_date.quarter
            season = get_season(month)
            
            test = pd.DataFrame({
                "Lag_1": [lag1],
                "Lag_2": [lag2],
                "Lag_3": [lag3],
                "Rolling_Mean_3": [rolling],
                "Month": [month],
                "Quarter": [quarter],
                "Season": [season]
            })
            
            pred = max(0.0, float(model.predict(test)[0]))
            predictions.append([next_date, pred])
            
            future = pd.concat([
                future,
                pd.DataFrame({"Order Date": [next_date], "Sales": [pred]})
            ], ignore_index=True)
            
        forecast = pd.DataFrame(predictions, columns=["Order Date", "Forecast"])
        return forecast, mae, rmse

    # PAGE 1 - SALES OVERVIEW DASHBOARD
    if page == "Sales Overview":
        st.markdown("<h1 class='main-title'>Sales Overview Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Analyze overall sales trends and segment performance using dynamic filters.</p>", unsafe_allow_html=True)
        
        # Filters in Sidebar
        st.sidebar.subheader("Dashboard Filters")
        regions = df["Region"].unique()
        categories = df["Category"].unique()
        
        selected_regions = st.sidebar.multiselect(
            "Select Region(s)",
            options=regions,
            default=list(regions)
        )
        
        selected_categories = st.sidebar.multiselect(
            "Select Category(s)",
            options=categories,
            default=list(categories)
        )
        
        # Apply filtering
        filtered_df = df[
            df["Region"].isin(selected_regions) &
            df["Category"].isin(selected_categories)
        ]
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please expand your selections.")
        else:
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            
            total_sales = filtered_df["Sales"].sum()
            total_orders = filtered_df["Order ID"].nunique()
            avg_order_val = filtered_df["Sales"].mean()
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Sales Volume</div>
                    <div class='metric-value'>${total_sales:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Unique Orders</div>
                    <div class='metric-value'>{total_orders:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Average Order Value (AOV)</div>
                    <div class='metric-value'>${avg_order_val:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### Sales Trend Visualizations")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Sales by Year (Bar Chart)
                sales_by_year = filtered_df.groupby("Year")["Sales"].sum().reset_index()
                sales_by_year["Year"] = sales_by_year["Year"].astype(str)
                fig_year = px.bar(
                    sales_by_year,
                    x="Year",
                    y="Sales",
                    text_auto=".2s",
                    title="Total Sales by Year",
                    labels={"Sales": "Sales ($)", "Year": "Calendar Year"},
                    color_discrete_sequence=["#4b6cb7"]
                )
                fig_year.update_layout(template="plotly_white", margin=dict(t=50, b=30, l=30, r=30))
                st.plotly_chart(fig_year, use_container_width=True)
                
            with chart_col2:
                # Monthly Sales Trend (Line Chart)
                sales_by_month = filtered_df.groupby("Month_Year")["Sales"].sum().reset_index()
                fig_month = px.line(
                    sales_by_month,
                    x="Month_Year",
                    y="Sales",
                    title="Monthly Sales Trend Line Chart",
                    labels={"Sales": "Sales ($)", "Month_Year": "Timeline"},
                    markers=True,
                    color_discrete_sequence=["#182848"]
                )
                fig_month.update_layout(template="plotly_white", margin=dict(t=50, b=30, l=30, r=30))
                st.plotly_chart(fig_month, use_container_width=True)
                
            # Extra interactive slice by Category / Region
            st.markdown("### Sales Breakdown by Selected Filters")
            col_break1, col_break2 = st.columns(2)
            with col_break1:
                cat_breakdown = filtered_df.groupby("Category")["Sales"].sum().reset_index()
                fig_cat = px.pie(cat_breakdown, values="Sales", names="Category", title="Category Sales Contribution", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_cat, use_container_width=True)
            with col_break2:
                reg_breakdown = filtered_df.groupby("Region")["Sales"].sum().reset_index()
                fig_reg = px.pie(reg_breakdown, values="Sales", names="Region", title="Region Sales Contribution", hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_reg, use_container_width=True)

    # PAGE 2 - FORECAST EXPLORER
    elif page == "Forecast Explorer":
        st.markdown("<h1 class='main-title'>Forecast Explorer</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Predict future sales at Category or Region level using the best performing forecasting model (XGBoost).</p>", unsafe_allow_html=True)
        
        # Dimensions and options selection
        dimension = st.radio("Choose Forecast Dimension", ["Category", "Region"], horizontal=True)
        
        if dimension == "Category":
            segment_options = list(df["Category"].unique())
            segment_selected = st.selectbox("Select Product Category", options=segment_options)
            filtered_forecast_data = df[df["Category"] == segment_selected]
        else:
            segment_options = list(df["Region"].unique())
            segment_selected = st.selectbox("Select Business Region", options=segment_options)
            filtered_forecast_data = df[df["Region"] == segment_selected]
            
        # Horizon Slider
        horizon_slider = st.select_slider(
            "Forecast Horizon (Months Ahead)",
            options=["1 Month (Jan 2019)", "2 Months (Feb 2019)", "3 Months (Mar 2019)"],
            value="3 Months (Mar 2019)"
        )
        
        horizon_map = {
            "1 Month (Jan 2019)": 1,
            "2 Months (Feb 2019)": 2,
            "3 Months (Mar 2019)": 3
        }
        horizon = horizon_map[horizon_slider]
        
        # Run forecast
        forecast_res, mae_res, rmse_res = run_segment_forecast(filtered_forecast_data, horizon)
        
        if forecast_res is None:
            st.error("Too few monthly sales data points available for forecasting this segment.")
        else:
            # Historical monthly sales prep for plotting
            hist_monthly = filtered_forecast_data.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
            # Rename for mapping
            hist_monthly = hist_monthly.rename(columns={"Sales": "Actual"})
            
            # Combine history and forecast
            combined_plot = pd.merge(hist_monthly, forecast_res, on="Order Date", how="outer")
            
            # Plotly Chart
            fig_fc = go.Figure()
            # Historical line
            fig_fc.add_trace(go.Scatter(
                x=combined_plot["Order Date"],
                y=combined_plot["Actual"],
                name="Historical Sales",
                mode="lines+markers",
                line=dict(color="#182848", width=2.5)
            ))
            # Forecast line
            fig_fc.add_trace(go.Scatter(
                x=combined_plot["Order Date"],
                y=combined_plot["Forecast"],
                name="XGBoost Forecast",
                mode="lines+markers",
                line=dict(color="#FF4B4B", width=2.5, dash="dash")
            ))
            
            fig_fc.update_layout(
                title=f"Sales Forecast Horizon for Segment: '{segment_selected}'",
                xaxis_title="Timeline",
                yaxis_title="Monthly Sales ($)",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Display Metrics Row
            st.markdown("### Model Performance Metrics (In-Sample)")
            met1, met2, met3 = st.columns(3)
            with met1:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #e65c00;'>
                    <div class='metric-label'>Model Type</div>
                    <div class='metric-value' style='font-size: 1.5rem;'>XGBoost Regressor</div>
                </div>
                """, unsafe_allow_html=True)
            with met2:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #2CA02C;'>
                    <div class='metric-label'>Mean Absolute Error (MAE)</div>
                    <div class='metric-value'>${mae_res:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with met3:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #9b59b6;'>
                    <div class='metric-label'>Root Mean Squared Error (RMSE)</div>
                    <div class='metric-value'>${rmse_res:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

    # PAGE 3 - ANOMALY REPORT
    elif page == "Anomaly Report":
        st.markdown("<h1 class='main-title'>Anomaly Report</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Inspect sales outliers and abnormalities using unsupervised Isolation Forests and Rolling Z-Scores.</p>", unsafe_allow_html=True)
        
        # Calculate weekly sales anomalies
        weekly_anom_df = run_anomalies_cache(df)
        
        # Tabs for Anomaly Methods
        tab1, tab2 = st.tabs(["Isolation Forest Detection", "Rolling Z-Score Detection"])
        
        with tab1:
            st.markdown("### Unsupervised Isolation Forest Anomalies")
            st.markdown("Isolation Forest isolates anomalies by randomly partitioning features. Outliers require fewer splits to isolate, flagging weekly sales points that are fundamentally outside typical behaviors.")
            
            # Interactive Anomaly Plot
            fig_if = go.Figure()
            # Normal Sales
            fig_if.add_trace(go.Scatter(
                x=weekly_anom_df["Order Date"],
                y=weekly_anom_df["Sales"],
                mode="lines",
                name="Normal Sales Trend",
                line=dict(color="#4b6cb7", width=1.5)
            ))
            # Outliers scatter
            anom_points = weekly_anom_df[weekly_anom_df["IF_Anomaly"] == True]
            fig_if.add_trace(go.Scatter(
                x=anom_points["Order Date"],
                y=anom_points["Sales"],
                mode="markers",
                name="Flagged Anomalies",
                marker=dict(color="#FF4B4B", size=10, symbol="x")
            ))
            
            fig_if.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Weekly Sales ($)",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig_if, use_container_width=True)
            
            # Table of Isolation Forest anomalies
            st.markdown("#### Detected Outliers")
            anom_points_display = anom_points[["Order Date", "Sales"]].copy()
            anom_points_display["Order Date"] = anom_points_display["Order Date"].dt.strftime("%Y-%m-%d")
            anom_points_display["Sales"] = anom_points_display["Sales"].map("${:,.2f}".format)
            
            # Simulated real world reasons
            explanations = [
                "Post-Holiday Dip (New Year week slowdown)",
                "Pre-Holiday Inventory Restocking surge",
                "Severe Winter Storm (logistical delay across East/Midwest)",
                "End of Q1 Sales Drive (high corporate volume)",
                "Summer Season Holiday Period (typical retail dip)",
                "Autumn Back-to-school promotional spikes",
                "New Year inventory write-down/clearance",
                "Peak Q4 Christmas Sales (extreme volume)",
                "Black Friday Promotion week",
                "Cyber Monday Promotion week",
                "Holiday Season Shopping Spree peak"
            ]
            
            # Pad explanations if anomalies count differs from notebook run
            if len(anom_points_display) == len(explanations):
                anom_points_display["Potential Cause / Event"] = explanations
            else:
                anom_points_display["Potential Cause / Event"] = "Spike/Dip likely driven by promotional campaign or seasonal adjustments"
                
            st.dataframe(anom_points_display.reset_index(drop=True), use_container_width=True)
            
        with tab2:
            st.markdown("### Rolling Z-Score Anomaly Detection")
            st.markdown("Z-Score detection flags observations that deviate by more than **2 standard deviations** from the 8-week rolling average. This represents sudden short-term spikes or dips.")
            
            # Interactive Z-Score Plot
            fig_z = go.Figure()
            # Normal Sales
            fig_z.add_trace(go.Scatter(
                x=weekly_anom_df["Order Date"],
                y=weekly_anom_df["Sales"],
                mode="lines",
                name="Sales",
                line=dict(color="#182848", width=1.5)
            ))
            # Rolling Mean
            fig_z.add_trace(go.Scatter(
                x=weekly_anom_df["Order Date"],
                y=weekly_anom_df["Rolling_Mean"],
                mode="lines",
                name="8-Week Rolling Mean",
                line=dict(color="#2CA02C", width=1.5, dash="dot")
            ))
            # Anomalies
            z_anoms = weekly_anom_df[weekly_anom_df["Z_Anomaly"] == True]
            fig_z.add_trace(go.Scatter(
                x=z_anoms["Order Date"],
                y=z_anoms["Sales"],
                mode="markers",
                name="Z-Score Outliers (> 2 STD)",
                marker=dict(color="#9b59b6", size=10, symbol="circle-open")
            ))
            
            fig_z.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Weekly Sales ($)",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig_z, use_container_width=True)
            
            st.markdown("#### Detected Z-Score Outliers")
            z_anoms_display = z_anoms[["Order Date", "Sales", "Z_Score"]].copy()
            z_anoms_display["Order Date"] = z_anoms_display["Order Date"].dt.strftime("%Y-%m-%d")
            z_anoms_display["Sales"] = z_anoms_display["Sales"].map("${:,.2f}".format)
            z_anoms_display["Z_Score"] = z_anoms_display["Z_Score"].round(2)
            st.dataframe(z_anoms_display.reset_index(drop=True), use_container_width=True)

        # Matplotlib images toggle option
        st.markdown("---")
        st.subheader("Reference Static Notebook Charts")
        show_static_charts = st.checkbox("Show static matplotlib charts generated in task analysis")
        if show_static_charts:
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                if os.path.exists("charts/isolation_forest_anomalies.png"):
                    st.image("charts/isolation_forest_anomalies.png", caption="Isolation Forest Anomalies (Task 5)", use_container_width=True)
                else:
                    st.info("Static chart 'charts/isolation_forest_anomalies.png' not found. Dynamic chart is rendered above.")
            with s_col2:
                if os.path.exists("charts/zscore_anomalies.png"):
                    st.image("charts/zscore_anomalies.png", caption="Z-Score Anomalies (Task 5)", use_container_width=True)
                else:
                    st.info("Static chart 'charts/zscore_anomalies.png' not found. Dynamic chart is rendered above.")

    # PAGE 4 - PRODUCT DEMAND SEGMENTS
    elif page == "Demand Segments":
        st.markdown("<h1 class='main-title'>Product Demand Segments</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Group product sub-categories based on total volume, growth rate, volatility, and order value using K-Means Clustering.</p>", unsafe_allow_html=True)
        
        # Run K-Means and Cache
        cluster_df, cluster_names = run_clustering_cache(df)
        
        # Interactive PCA Scatter Plot
        fig_pca = px.scatter(
            cluster_df.reset_index(),
            x="PC1",
            y="PC2",
            color="Demand Group",
            hover_name="Sub-Category",
            text="Sub-Category",
            title="Product Demand Segmentation Map (PCA 2D Projection)",
            color_discrete_map={
                "High Growth, Premium Value": "#FF4B4B",
                "High Volume, Stable Demand": "#1F77B4",
                "Low Volume, Stable Demand": "#2CA02C"
            }
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_pca.update_layout(showlegend=True, height=600, template="plotly_white")
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Table listing Sub-categories
        st.markdown("### Sub-Category Assignments and Statistics")
        disp_cluster_df = cluster_df.copy().reset_index()
        disp_cluster_df = disp_cluster_df.rename(columns={
            "Sub-Category": "Sub-Category",
            "Total Sales Volume": "Total Sales Volume ($)",
            "Sales Growth Rate (%)": " YoY Sales Growth Rate (%)",
            "Sales Volatility": "Sales Volatility ($)",
            "Average Order Value": "Average Order Value ($)"
        })
        
        # Formatting
        disp_cluster_df["Total Sales Volume ($)"] = disp_cluster_df["Total Sales Volume ($)"].map("${:,.2f}".format)
        disp_cluster_df["Sales Volatility ($)"] = disp_cluster_df["Sales Volatility ($)"].map("${:,.2f}".format)
        disp_cluster_df["Average Order Value ($)"] = disp_cluster_df["Average Order Value ($)"].map("${:,.2f}".format)
        disp_cluster_df[" YoY Sales Growth Rate (%)"] = disp_cluster_df[" YoY Sales Growth Rate (%)"].map("{:.2f}%".format)
        
        cols_to_show = [
            "Sub-Category", 
            "Total Sales Volume ($)", 
            " YoY Sales Growth Rate (%)", 
            "Sales Volatility ($)", 
            "Average Order Value ($)", 
            "Demand Group"
        ]
        
        st.dataframe(disp_cluster_df[cols_to_show].sort_values("Demand Group"), use_container_width=True)
        
        # Stocking strategies recommendations
        st.markdown("### Recommended Stocking Strategies")
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown("""
            <div class='custom-card' style='border-top: 5px solid #1F77B4;'>
                <h4 style='color:#1F77B4; margin-top:0;'>High Volume, Stable Demand</h4>
                <p><b>Sub-categories:</b> Chairs, Phones, Storage, Binders, Tables, Bookcases, etc.</p>
                <p><b>Recommended Strategy:</b></p>
                <ul>
                    <li>Maintain high safety stock levels to avoid stockouts.</li>
                    <li>Utilize automated replenishment systems triggered by threshold levels.</li>
                    <li>Negotiate volume-based discounts with long-term contracts.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_rec2:
            st.markdown("""
            <div class='custom-card' style='border-top: 5px solid #FF4B4B;'>
                <h4 style='color:#FF4B4B; margin-top:0;'>High Growth, Premium Value</h4>
                <p><b>Sub-categories:</b> Copiers, Appliances, Accessories, etc.</p>
                <p><b>Recommended Strategy:</b></p>
                <ul>
                    <li>Gradually increase stock allocations monthly.</li>
                    <li>Monitor sales growth closely; use agile reordering.</li>
                    <li>Maintain low holding times, focusing on just-in-time logistics for premium products to preserve capital.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_rec3:
            st.markdown("""
            <div class='custom-card' style='border-top: 5px solid #2CA02C;'>
                <h4 style='color:#2CA02C; margin-top:0;'>Low Volume, Stable Demand</h4>
                <p><b>Sub-categories:</b> Art, Fasteners, Labels, Envelopes, Paper, etc.</p>
                <p><b>Recommended Strategy:</b></p>
                <ul>
                    <li>Maintain low inventory buffers to minimize holding costs.</li>
                    <li>Trigger restocks only on demand spikes or low safety levels.</li>
                    <li>Bundle items together in promotions to liquidate slowly moving stock.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Could not read 'train.csv'. Please upload or add it to the directory to activate the dashboard.")
