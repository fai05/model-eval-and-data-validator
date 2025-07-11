# Imports
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

# Utility: Find periods with common issues across variables
def detect_common_issue_periods(ds, time_col, variables):
    issue_flags = {}
    for var in variables:
        # Convert variable to DataFrame and flag missing/flatline
        var_series = ds[var].to_dataframe().reset_index().set_index(time_col).sort_index()
        var_series = var_series[[var]].rename(columns={var: 'value'})
        var_series['missing'] = var_series['value'].isna()
        var_series['flatline'] = var_series['value'].rolling(window=24, min_periods=1).std() < 1e-5
        issue_flags[var] = var_series[['missing', 'flatline']]
    # Combine all flags into one DataFrame
    combined_issues = pd.concat(issue_flags.values(), axis=1, keys=issue_flags.keys())
    flat_df = combined_issues.copy()
    flat_df.columns = ['_'.join(col).strip() for col in flat_df.columns.values]
    # Find times where all variables are missing, flatline, or overlap
    all_missing_mask = flat_df.filter(like='missing').all(axis=1)
    all_flatline_mask = flat_df.filter(like='flatline').all(axis=1)
    any_overlap_mask = flat_df.filter(like='missing').any(axis=1) & flat_df.filter(like='flatline').any(axis=1)
    return {
        "all_missing_times": all_missing_mask[all_missing_mask].index.to_list(),
        "all_flatline_times": all_flatline_mask[all_flatline_mask].index.to_list(),
        "overlap_times": any_overlap_mask[any_overlap_mask].index.to_list()
    }

# Streamlit UI setup
st.set_page_config(page_title="Hydro Data Validator", page_icon="📈", layout="wide")

st.title("🌊 Hydrological Input Data Validator")

# File upload
uploaded_file = st.file_uploader("📂 Upload a NetCDF file", type=["nc"])

if uploaded_file is not None:
    ds = xr.open_dataset(uploaded_file)
    # Find datetime coordinate
    time_coords = [c for c in ds.coords if np.issubdtype(ds[c].dtype, np.datetime64)]
    if not time_coords:
        st.error("No datetime coordinate found.")
        st.stop()
    time_col = time_coords[0]
    variables = list(ds.data_vars)
    st.sidebar.markdown("### 🧪 Variable Selection")
    selected_var = st.sidebar.selectbox("Select a variable to validate", variables)

    # Prepare DataFrame for selected variable
    df = ds[selected_var].to_dataframe().reset_index().set_index(time_col).sort_index()
    df = df.rename(columns={selected_var: 'value'})

    # Date range filter
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    st.sidebar.markdown("### 🗓️ Select Date Range")
    start_date, end_date = st.sidebar.date_input("Filter by date", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

    # Flatline counting mode toggle
    use_flatline_per_event = st.sidebar.toggle("Count flatline per event instead of daily", value=True)

    # Set bounds for out-of-range detection
    lower_bound, upper_bound = None, None
    selected_var_lower = selected_var.lower()
    if 'discharge' in selected_var_lower or 'q' in selected_var_lower:
        lower_bound = 0
    elif 'p' in selected_var_lower:
        lower_bound = 0
    elif 'e' in selected_var_lower:
        lower_bound = 0
    elif 'temp' in selected_var_lower:
        lower_bound = None
        upper_bound = None

    # Flagging issues
    df['missing'] = df['value'].isna()
    df['zscore'] = zscore(df['value'].ffill())
    df['spike'] = np.abs(df['zscore']) > 3
    df['flatline'] = df['value'].rolling(window=24, min_periods=1).std() < 1e-5
    df['out_of_range'] = False
    if lower_bound is not None:
        df['out_of_range'] |= df['value'] < lower_bound
    if upper_bound is not None:
        df['out_of_range'] |= df['value'] > upper_bound
    elif 'temp' not in selected_var_lower:
        # For non-temperature, flag extreme high values
        df['out_of_range'] |= df['value'] > df['value'].quantile(0.999)

    # Detect constant value periods
    df['constant_flag'] = df['value'].diff().abs() < 1e-10
    df['const_group'] = (df['constant_flag'] != df['constant_flag'].shift()).cumsum()

    # Group flatline sequences
    df['flatline_group'] = (df['flatline'] != df['flatline'].shift()).cumsum()
    flatline_events = df[df['flatline']].groupby('flatline_group')

    # Flatline summary: per event or daily
    if use_flatline_per_event:
        flatline_summary = []
        for _, group in flatline_events:
            start_time = pd.to_datetime(group.index.min())
            end_time = pd.to_datetime(group.index.max())
            flatline_summary.append({
                'start': start_time,
                'end': end_time,
                'month': start_time.strftime('%Y-%m')
            })
        flatline_summary = pd.DataFrame(flatline_summary)
        if not flatline_summary.empty:
            flatline_monthly_counts = flatline_summary.groupby('month').size().rename('flatline')
        else:
            flatline_monthly_counts = pd.Series(dtype='int', name='flatline')
    else:
        df['time_bin'] = df.index.floor('D')
        flatline_daily = df.groupby('time_bin')['flatline'].any().astype(int)
        flatline_daily = flatline_daily.reset_index()
        flatline_daily['month'] = pd.to_datetime(flatline_daily['time_bin']).dt.to_period('M').astype(str)
        flatline_monthly_counts = flatline_daily.groupby('month')['flatline'].sum().rename('flatline')

    # Monthly aggregation of issues
    df['time_bin'] = df.index.floor('D')
    issue_daily = df.groupby('time_bin')[['missing', 'spike', 'out_of_range']].any().astype(int)
    issue_daily['month'] = pd.to_datetime(issue_daily.index).to_period('M').astype(str)
    monthly_agg = issue_daily.groupby('month')[['missing', 'spike', 'out_of_range']].sum()
    monthly_agg = monthly_agg.merge(flatline_monthly_counts, left_index=True, right_index=True, how='left')
    monthly_agg['flatline'] = monthly_agg['flatline'].fillna(0).astype(int)

    # Quality scores
    df['any_issue_with_spikes'] = df[['missing', 'flatline', 'spike', 'out_of_range']].any(axis=1)
    score_with_spikes = round(100 - df['any_issue_with_spikes'].mean() * 100, 2)
    df['any_issue_no_spikes'] = df[['missing', 'flatline', 'out_of_range']].any(axis=1)
    score_no_spikes = round(100 - df['any_issue_no_spikes'].mean() * 100, 2)

    # Streamlit Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary",
        "🔥 Heatmap",
        "🔹 Constant Periods",
        "📈 Time Series",
        "🌐 Hydrographs",
        "🚨 Critical Overlaps"
    ])

    # Tab 1: Summary
    with tab1:
        st.subheader("🔍 Data Quality")
        col1, col2 = st.columns(2)
        col1.metric("Quality Score (with spikes)", score_with_spikes)
        col2.metric("Quality Score (no spikes)", score_no_spikes)
        st.subheader("📅 Monthly Issue")
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            f"Missing",
            f"Flatline",
            f"Spike",
            f"Out-of-Range"
        ])
        with subtab1:
            st.plotly_chart(px.bar(monthly_agg, y='missing', title="Monthly Missing Values"), use_container_width=True)
        with subtab2:
            st.plotly_chart(px.bar(monthly_agg, y='flatline', title="Monthly Flatline Flags (Daily Max 1)"), use_container_width=True)
        with subtab3:
            st.plotly_chart(px.bar(monthly_agg, y='spike', title="Monthly Spike Flags"), use_container_width=True)
        with subtab4:
            st.plotly_chart(px.bar(monthly_agg, y='out_of_range', title="Monthly Out-of-Range Flags"), use_container_width=True)

    # Tab 2: Heatmap
    with tab2:
        st.subheader("🔥 Heatmap of Issues by Month")
        fig2, ax = plt.subplots(figsize=(min(15, len(monthly_agg)), 5))
        sns.heatmap(monthly_agg.T, cmap="YlOrRd", linewidths=.5, annot=True, fmt='g', ax=ax)
        st.pyplot(fig2)

    # Tab 3: Constant Value Periods
    with tab3:
        st.subheader("🔹 Constant Value Periods (≥30 days)")
        include_zeros = st.checkbox("Include constant zero values (≥30 days)", value=True)
        if include_zeros:
            const_periods = df[df['constant_flag']].groupby('const_group').filter(lambda x: len(x) >= 30)
        else:
            const_periods = df[df['constant_flag']].groupby('const_group').filter(
                lambda x: len(x) >= 30 and x['value'].iloc[0] != 0
            )
        if not const_periods.empty:
            const_summary = const_periods.groupby('const_group').agg(
                start=('value', lambda x: x.index.min()),
                end=('value', lambda x: x.index.max()),
                value=('value', 'first'),
                days=('value', 'count')
            )
            st.dataframe(const_summary)
        else:
            st.info("No constant value periods longer than 30 days detected.")

    # Tab 4: Time Series Plot
    with tab4:
        st.subheader("📈 Time Series Plot with Issue Markers")
        fig = px.line(df, y='value', title=f"{selected_var} Time Series")
        for flag, color in {'missing': 'red', 'flatline': 'orange', 'out_of_range': 'purple'}.items():
            flagged = df[df[flag]]
            fig.add_scatter(x=flagged.index, y=flagged['value'], mode='markers', name=flag, marker=dict(color=color, size=6))
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Hydrographs
    with tab5:  
        st.subheader("🌐 Hydrographs")
        tabs = st.tabs(variables + ["Combined Hydrograph"])
        all_data = ds.to_dataframe().reset_index().set_index(time_col).sort_index()
        all_data = all_data[(all_data.index.date >= start_date) & (all_data.index.date <= end_date)]
        def filter_variable_data(data, var):
            return data[[var]].dropna()
        for var, tab in zip(variables, tabs[:-1]):
            with tab:
                filtered_var = filter_variable_data(all_data, var)
                fig_var = px.line(filtered_var, y=var, title=f"Hydrograph - {var}")
                fig_var.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                st.plotly_chart(fig_var, use_container_width=True)
        with tabs[-1]:
            combined_df = all_data[['P', 'Q_meas_dist']].dropna(how='all')
            fig_combined = go.Figure()
            if not combined_df['P'].dropna().empty:
                fig_combined.add_trace(go.Bar(x=combined_df.index, y=combined_df['P'], name="Precipitation (P)", yaxis="y2", marker_color='blue', opacity=0.5))
            if not combined_df['Q_meas_dist'].dropna().empty:
                fig_combined.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Q_meas_dist'], mode='lines', name="Discharge (Q)", line=dict(color='black')))
            if not fig_combined.data:
                st.warning("No valid data for 'P' or 'Q_meas_dist' in selected date range.")
            else:
                fig_combined.update_layout(
                    title="Combined Hydrograph: Discharge and Precipitation",
                    xaxis_title="Date",
                    yaxis=dict(title="Discharge (m³/s)", side='left'),
                    yaxis2=dict(title="Precipitation (mm)", overlaying='y', side='right'),
                    legend=dict(x=0, y=1.1, orientation="h"),
                    height=500,
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                st.plotly_chart(fig_combined, use_container_width=True)

    # Tab 6: Critical Overlaps
    with tab6:
        st.subheader("🚨 Periods with Critical Overlaps Across Variables")
        result = detect_common_issue_periods(ds, time_col, variables)
        # Prepare flat_df to check which vars were involved
        flat_df = pd.concat([
            ds[var].to_dataframe().reset_index().set_index(time_col).sort_index()[[var]].rename(columns={var: 'value'}).assign(
                missing=lambda x: x['value'].isna(),
                flatline=lambda x: x['value'].rolling(window=24, min_periods=1).std() < 1e-5
            )[['missing', 'flatline']]
            for var in variables
        ], axis=1, keys=variables)
        flat_df.columns = ['_'.join(col).strip() for col in flat_df.columns.values]
        # Combine and label
        all_times = (
            [(t, 'All Missing') for t in result['all_missing_times']] +
            [(t, 'All Flatline') for t in result['all_flatline_times']] +
            [(t, 'Missing & Flatline') for t in result['overlap_times']]
        )
        overlap_df = pd.DataFrame(all_times, columns=['Timestamp', 'Issue Type'])
        # Filter by selected date range
        overlap_df = overlap_df[
            (overlap_df['Timestamp'] >= pd.to_datetime(start_date)) &
            (overlap_df['Timestamp'] <= pd.to_datetime(end_date))
        ]
        # Compute affected vars per timestamp
        def get_affected_vars(ts):
            involved = []
            for var in variables:
                miss_col = f"{var}_missing"
                flat_col = f"{var}_flatline"
                if ts in flat_df.index:
                    if flat_df.at[ts, miss_col]:
                        involved.append(f"{var} (missing)")
                    if flat_df.at[ts, flat_col]:
                        involved.append(f"{var} (flatline)")
            return ', '.join(involved)
        if not overlap_df.empty:
            overlap_df['Affected Variables'] = overlap_df['Timestamp'].apply(get_affected_vars)
            st.dataframe(overlap_df.sort_values("Timestamp"))
            # Timeline plot
            st.subheader("🧯 Timeline of Critical Overlaps")
            bar_data = overlap_df.copy()
            bar_data["Start"] = pd.to_datetime(bar_data["Timestamp"])
            bar_data["End"] = bar_data["Start"] + pd.Timedelta(hours=1)
            fig4 = px.timeline(
                bar_data,
                x_start="Start",
                x_end="End",
                y="Issue Type",
                color="Issue Type",
                hover_data=["Affected Variables"],
                title="Timeline of Critical Overlaps",
                height=400
            )
            fig4.update_yaxes(autorange="reversed")
            fig4.update_layout(
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Issue Type",
                margin=dict(t=40, l=20, r=20, b=40)
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.success("✅ No critical overlaps during selected period.")

    # Download flagged data
    csv = df.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Full Flagged Data as CSV", data=csv, file_name=f"{selected_var}_validated.csv", mime='text/csv')

