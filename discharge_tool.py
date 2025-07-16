# Imports
import io
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# Frequency detection and resampling utilities
def detect_freq(idx):
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            raise TypeError("Cannot convert index to datetime; make sure your 'date' column is parsed correctly.")
    f = pd.infer_freq(idx)
    if f is not None:
        if f.endswith("H"):
            return "Hourly"
        if f.endswith("D"):
            return "Daily"
    deltas = np.diff(idx.values).astype("timedelta64[s]").astype(int)
    med = np.median(deltas)
    return "Hourly" if med < 24*3600 else "Daily"

def to_freq(df: pd.DataFrame, native: str, target: str) -> pd.DataFrame:
    if native == target:
        return df
    if native == "Hourly" and target == "Daily":
        return df.resample("D").mean()
    if native == "Daily" and target == "Hourly":
        return df.resample("H").ffill()
    raise ValueError(f"Cannot convert from {native} to {target}")

# Streamlit page config and custom CSS
st.set_page_config(page_title="Discharge Evaluation", page_icon="📊", layout="wide")
st.markdown("""
<style>
h1 { text-align: center; color: #0077BE !important; }
h2, h3, h4 { color: #005F8A !important; }
.css-1aumxhk, .css-1abqosx { background-color: #E0F7FA !important; border-radius: 8px; }
.stButton>button { background-color: #0077BE !important; color: white !important; border-radius: 6px !important; padding: 0.6em 1.2em !important; }
.stMetric { background-color: #B2EBF2 !important; border: 1px solid #90DFF3 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #2F4F4F; font-family: sans-serif;'>Discharge Evaluation Tool</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Input format selection
st.markdown("<h3 style='text-align: center;'>Choose your input format</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    input_fmt = st.radio("", ["CSV files", "Pickle file"], horizontal=True, label_visibility="collapsed")
st.markdown("---")

df_pred = df_obs = None

if input_fmt == "CSV files":
    # CSV upload
    pred_file = st.file_uploader("Upload predicted CSV", type="csv")
    obs_file  = st.file_uploader("Upload observed CSV",  type="csv")
    if pred_file and obs_file:
        df_pred = pd.read_csv(pred_file, parse_dates=["date"])
        df_obs  = pd.read_csv(obs_file,  parse_dates=["date"])
        df_pred = df_pred.rename(columns={"predicted_discharge": "predicted"})
        df_obs  = df_obs .rename(columns={"QObs(mm/h)_obs":  "observed"})
    else:
        st.stop()
else:
    # Pickle upload (multiple pickles supported)
    pkl_files = st.file_uploader(
        "Upload one or more pickle(s) (`test_results.p`)", 
        type=["p", "pkl"], 
        accept_multiple_files=True
    )
    if not pkl_files:
        st.stop()
    all_runs = []
    for idx, pkl in enumerate(pkl_files, start=1):
        results = pickle.load(io.BytesIO(pkl.getvalue()))
        records = []
        for basin, basin_data in results.items():
            for freq, fdata in basin_data.items():
                xr = fdata.get("xr")
                if xr is None:
                    continue
                dfx = xr.to_dataframe().reset_index()
                dfx["basin"] = basin
                dfx["freq"]  = freq
                records.append(dfx)
        if records:
            df_run = pd.concat(records, ignore_index=True)
            df_run["model"] = f"Model{idx}"
            all_runs.append(df_run)
    full = pd.concat(all_runs, ignore_index=True)
    # Model selection
    model_sel = st.selectbox("Which run do you want to evaluate?", full["model"].unique().tolist())
    full = full[full["model"] == model_sel]
    # Unit conversion option
    do_conv = st.radio(
        "Convert from mm/s to m³/s using basin areas?",
        ["No, data already in m³/s", "Yes, convert now"],
        horizontal=True
    )
    if do_conv == "Yes, convert now":
        st.markdown("**Upload basin–area lookup** (CSV with columns `gemaal_id, oppervlak`):")
        area_file = st.file_uploader("Basin areas CSV", type="csv", key="area_csv")
        if not area_file:
            st.error("We need your basin–area CSV to convert units!")
            st.stop()
        area_df = pd.read_csv(area_file)
        if not {"gemaal_id","oppervlak"}.issubset(area_df.columns):
            st.error("Your area file must have columns: `gemaal_id, oppervlak`")
            st.stop()
        area_df = area_df.rename(columns={"gemaal_id":"basin", "oppervlak":"area_km2"})
        full = full.merge(area_df, on="basin", how="left")
        if full["area_km2"].isnull().any():
            missing = full.loc[full["area_km2"].isnull(), "basin"].unique().tolist()
            st.error(f"No area for basins: {missing}")
            st.stop()
        obs_col = [c for c in full.columns if c.endswith("_obs")][0]
        sim_col = [c for c in full.columns if c.endswith("_sim")][0]
        full["observed"]  = full[obs_col] * full["area_km2"] / 1000.0
        full["predicted"] = full[sim_col] * full["area_km2"] / 1000.0
    else:
        # Rename columns if needed
        full = full.rename(
            columns={ [c for c in full.columns if c.endswith("_obs")][0]: "observed",
                      [c for c in full.columns if c.endswith("_sim")][0]: "predicted" }
        )
    df_obs  = full[["date","observed"]]
    df_pred = full[["date","predicted"]]

# Frequency detection and resampling
freq_pred = detect_freq(df_pred.index)
freq_obs  = detect_freq(df_obs.index)
st.write(f"**Detected predicted‐series frequency:** {freq_pred or 'Unknown'}")
st.write(f"**Detected observed‐series frequency:** {freq_obs or 'Unknown'}")
if freq_pred is None or freq_obs is None:
    st.warning("Could not unambiguously detect one of your input frequencies.  If this is wrong, please resample your data manually before upload.")
eval_freq = st.selectbox("Evaluate at", ["Hourly","Daily"])
alias = {"Hourly": "H", "Daily": "D"}
native_pred = alias.get(freq_pred)
native_obs  = alias.get(freq_obs)
target      = alias[eval_freq]
df_pred = df_pred.set_index("date").sort_index()
df_obs  = df_obs .set_index("date").sort_index()
freq_pred = detect_freq(df_pred.index)
freq_obs  = detect_freq(df_obs.index)
df_pred = to_freq(df_pred, freq_pred, eval_freq)
df_obs  = to_freq(df_obs,  freq_obs,  eval_freq)

# Date range selection
st.sidebar.subheader("Select evaluation period")
min_date = max(df_pred.index.min(), df_obs.index.min()).date()
max_date = min(df_pred.index.max(), df_obs.index.max()).date()
d = st.sidebar.date_input(
    "From ⇨ To",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
if isinstance(d, (list, tuple)) and len(d) == 2:
    start_date, end_date = d
else:
    st.error("Please select a start and end date.")
    st.stop()
mask_pred = (df_pred.index.date >= start_date) & (df_pred.index.date <= end_date)
mask_obs  = (df_obs.index .date >= start_date) & (df_obs.index .date <= end_date)
df_pred = df_pred.loc[mask_pred]
df_obs  = df_obs.loc[mask_obs]
if df_pred.empty or df_obs.empty:
    st.error("No data in that date range after resampling!")
    st.stop()

# Merge and deduplicate
# Remove repeated timestamps and join on index
df_obs   = df_obs[~df_obs.index.duplicated(keep="first")]
df_pred  = df_pred[~df_pred.index.duplicated(keep="first")]
df = (
    pd.concat(
        [df_obs["observed"], df_pred["predicted"]],
        axis=1,
        join="inner",
    )
    .dropna(how="any")
)
if df.empty:
    st.error("No overlapping dates after resampling!")
    st.stop()
times  = df.index
y_true = df["observed"].values
y_pred = df["predicted"].values

# Metric functions
def compute_kge(obs, sim):
    if len(obs) < 2:
        return np.nan
    r, _ = pearsonr(obs, sim)
    α    = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    β    = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (α-1)**2 + (β-1)**2)

def compute_peak_timing(obs, sim, window=3):
    peaks, _ = find_peaks(obs)
    if len(peaks)==0:
        return np.nan
    errs = []
    for i in peaks:
        lo = max(0, i-window)
        hi = min(len(sim)-1, i+window)
        j  = lo + np.argmax(sim[lo:hi+1])
        errs.append(abs(j - i))
    return float(np.mean(errs))

# Compute metrics
mae         = mean_absolute_error(y_true, y_pred)
rmse        = np.sqrt(mean_squared_error(y_true, y_pred))
r2          = r2_score(y_true, y_pred)
nse         = 1 - np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
kge         = compute_kge(y_true, y_pred)
pbias       = 100 * np.sum(y_pred-y_true)/np.sum(y_true)
peak_timing = compute_peak_timing(y_true, y_pred)

# Metrics and plots tabs
tab1, tab2 = st.tabs(["📊 Metrics", "📈 Plots"])

with tab1:
    st.subheader("Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",   f"{mae:.3f}")
    c2.metric("RMSE",  f"{rmse:.3f}")
    c3.metric("R²",    f"{r2:.3f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("NSE",   f"{nse:.3f}")
    c5.metric("KGE",   f"{kge:.3f}")
    c6.metric("PBIAS", f"{pbias:.3f}%")
    c7, c8, c9 = st.columns(3)
    c7.metric("Peak‐Timing", f"{peak_timing:.3f}")

with tab2:
    scatter_tab, hydro_tab, resid_tab, hist_tab, fdc_tab, seasonal_tab, efficiency_tab = st.tabs([
        "1:1 Scatter", "Hydrograph", "Residuals", "Histogram",
        "Flow‐Duration Curve", "Seasonal Error", "Efficiency"
    ])

    with scatter_tab:
        fig = px.scatter(x=y_true, y=y_pred, labels={"x":"Observed","y":"Predicted"}, title="1:1 Scatter")
        st.plotly_chart(fig)

    with hydro_tab:
        df_line = pd.DataFrame({"Date": times, "Observed": y_true, "Predicted": y_pred})
        fig = px.line(df_line, x="Date", y=["Observed","Predicted"], labels={"value":"Discharge (m³/s)","variable":"Legend"}, title="Hydrograph")
        fig.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)

    with resid_tab:
        res = y_pred - y_true
        df_res = pd.DataFrame({"Observed":y_true,"Residual":res})
        fig = px.scatter(df_res, x="Observed", y="Residual", title="Residuals vs Observed")
        st.plotly_chart(fig)

    with hist_tab:
        df_res = pd.DataFrame({"residual": res})
        fig = px.histogram(df_res, x="residual", nbins=50, opacity=0.7, labels={"residual": "Residual (Pred − Obs)"}, title="Residual Histogram")
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero error", annotation_position="top left")
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with fdc_tab:
        obs_s  = np.sort(y_true)[::-1]
        pred_s = np.sort(y_pred)[::-1]
        n = len(obs_s)
        exceedance = np.arange(1, n+1) / n * 100
        df_fdc = pd.DataFrame({"Exceedance (%)": exceedance, "Obs FDC": obs_s, "Pred FDC": pred_s})
        fig = px.line(df_fdc, x="Exceedance (%)", y=["Obs FDC", "Pred FDC"], title="Flow‑Duration Curve", labels={"Exceedance (%)": "Exceedance Probability (%)", "value": "Discharge", "variable": "Series"})
        st.plotly_chart(fig, use_container_width=True)

    with seasonal_tab:
        view = st.radio("Select Seasonal View", ["Climatology (mean)", "Year–month overlay", "Small multiples", "Monthly residual boxplot"], horizontal=True)
        df["year"]  = df.index.year
        df["month"] = df.index.month
        monthly = df.groupby(["year","month"])[["predicted","observed"]].apply(lambda g: pd.Series({"RMSE": np.sqrt(((g.predicted - g.observed)**2).mean()), "MAE":  np.abs(g.predicted - g.observed).mean()})).reset_index()
        rmse_pivot = monthly.pivot(index="month", columns="year", values="RMSE")
        mae_pivot  = monthly.pivot(index="month", columns="year", values="MAE")
        all_years  = sorted(rmse_pivot.columns)
        if view == "Climatology (mean)":
            avg = pd.DataFrame({"RMSE": rmse_pivot.mean(axis=1), "MAE":  mae_pivot.mean(axis=1)}).reset_index().melt(id_vars="month", var_name="Metric", value_name="Error")
            fig = px.line(avg, x="month", y="Error", color="Metric", markers=True, labels={"month":"Month", "Error":"Error"}, title="Climatological Monthly RMSE & MAE")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        elif view == "Year–month overlay":
            years = st.multiselect("Choose years to overlay", all_years, default=all_years)
            overlay = monthly[monthly["year"].isin(years)].melt(id_vars=["year","month"], value_vars=["RMSE","MAE"], var_name="Metric", value_name="Error")
            fig = px.line(overlay, x="month", y="Error", color="year", line_dash="Metric", markers=True, labels={"month":"Month","Error":"Error","year":"Year"}, title="Monthly RMSE & MAE by Year")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        elif view == "Small multiples":
            years = st.multiselect("Choose years to show panels", all_years, default=all_years)
            sm = monthly[monthly["year"].isin(years)].melt(id_vars=["year","month"], value_vars=["RMSE","MAE"], var_name="Metric", value_name="Error")
            fig = px.line(sm, x="month", y="Error", color="Metric", facet_col="year", facet_col_wrap=2, markers=True, labels={"month":"Month","Error":"Error"}, title="Monthly RMSE & MAE (small multiples)")
            fig.update_xaxes(dtick=1)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            res = df["predicted"] - df["observed"]
            box_df = pd.DataFrame({"month": np.repeat(df.index.month.values, 1), "residual": res})
            fig = px.box(box_df, x="month", y="residual", points="all", labels={"month":"Month","residual":"Residual (Pred−Obs)"}, title="Monthly Residual Distribution")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)

    with efficiency_tab:
        def compute_nse(obs, sim):
            if len(obs) < 2 or len(sim) < 2:
                return np.nan
            return 1 - np.sum((sim - obs)**2) / np.sum((obs - obs.mean())**2)
        def compute_kge(obs, sim):
            if len(obs) < 2 or len(sim) < 2:
                return np.nan
            r, _ = pearsonr(obs, sim)
            α    = np.std(sim, ddof=1) / np.std(obs, ddof=1)
            β    = np.mean(sim)      / np.mean(obs)
            return 1 - np.sqrt((r - 1)**2 + (α - 1)**2 + (β - 1)**2)
        df["year"]  = df.index.year
        df["month"] = df.index.month
        def safe_efficiency(g):
            obs = g.observed.values
            pred = g.predicted.values
            if len(obs) < 2 or len(pred) < 2:
                return pd.Series({"NSE": np.nan, "KGE": np.nan})
            return pd.Series({
                "NSE": compute_nse(obs, pred),
                "KGE": compute_kge(obs, pred)
            })
        monthly_eff = df.groupby(["year","month"]).apply(safe_efficiency).reset_index()
        clim_eff = monthly_eff.groupby("month")[["NSE","KGE"]].mean().reset_index().melt(id_vars="month", var_name="Metric", value_name="Value")
        fig = px.line(clim_eff, x="month", y="Value", color="Metric", markers=True, labels={"month":"Month", "Value":"Score"}, title="Climatological Monthly NSE & KGE")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)