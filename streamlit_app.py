# t20bat.py
"""
T20 Pitchmap App (Streamlit)
- Place your CSV at: data/t20_bbb.csv
- Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from html import escape
import streamlit.components.v1 as components

DATA_PATH = "t20_bbb.parquet"

# -------------------------
# Wagon Wheel plotter
# -------------------------
def wagon_wheel_plot(df, plot_type="boundaries", bg="white"):
    bgc = get_bg_colors(bg)
    
    if df is None or df.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor=bgc["plot_bg"],
            paper_bgcolor=bgc["panel_bg"],
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(text="No data available", font=dict(color=bgc["text_col"], size=14))
        )
        return fig
    
    if plot_type == "boundaries":
        df_plot = df[(df["batruns"] == 4) | (df["batruns"] == 6)].copy()
        title = "Wagon Wheel: Boundaries"
    else:
        df_plot = df[df["dismissal"].astype(str).str.strip().str.lower() == "caught"].copy()
        title = "Wagon Wheel: Caught Out Dismissals"
    
    df_plot = df_plot.dropna(subset=["wagonX", "wagonY"])
    
    if df_plot.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor=bgc["plot_bg"],
            paper_bgcolor=bgc["panel_bg"],
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(text=f"{title}<br>(No data)", font=dict(color=bgc["text_col"], size=14))
        )
        return fig
    
    df_plot["wagonX"] = pd.to_numeric(df_plot["wagonX"], errors="coerce")
    df_plot["wagonY"] = pd.to_numeric(df_plot["wagonY"], errors="coerce")
    df_plot = df_plot.dropna(subset=["wagonX", "wagonY"])
    
    wagonX_center = 83.26519
    wagonY_center = 87.70546

    df_plot["wagonX_relative"] = df_plot["wagonX"] - wagonX_center
    df_plot["wagonY_relative"] = wagonY_center - df_plot["wagonY"]

    df_plot["theta"] = np.arctan2(df_plot["wagonY_relative"], df_plot["wagonX_relative"])
    df_plot["r"] = np.sqrt(df_plot["wagonX_relative"]**2 + df_plot["wagonY_relative"]**2)
    
    fig = go.Figure()
    
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_r = 300
    circle_x = circle_r * np.cos(circle_theta)
    circle_y = circle_r * np.sin(circle_theta)
    
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color='lightgreen', width=2),
        fill='toself',
        fillcolor='rgba(144, 238, 144, 0.3)',
        hoverinfo='skip',
        showlegend=False
    ))
    
    if plot_type == "boundaries":
        df_plot["plot_r"] = circle_r
        df_4s = df_plot[df_plot["batruns"] == 4]
        df_6s = df_plot[df_plot["batruns"] == 6]
        for _, row in df_4s.iterrows():
            x_end = row["plot_r"] * np.cos(row["theta"])
            y_end = row["plot_r"] * np.sin(row["theta"])
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines', line=dict(color='blue', width=2), hoverinfo='skip', showlegend=False))
        for _, row in df_6s.iterrows():
            x_end = row["plot_r"] * np.cos(row["theta"])
            y_end = row["plot_r"] * np.sin(row["theta"])
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines', line=dict(color='red', width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', width=3), name='Fours', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=3), name='Sixes', showlegend=True))
    else:
        max_r = df_plot["r"].max()
        if max_r > 0:
            df_plot["plot_r"] = (df_plot["r"] / max_r) * (circle_r * 0.9)
        else:
            df_plot["plot_r"] = circle_r * 0.5
        for _, row in df_plot.iterrows():
            x_end = row["plot_r"] * np.cos(row["theta"])
            y_end = row["plot_r"] * np.sin(row["theta"])
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines', line=dict(color='darkorange', width=2), hoverinfo='skip', showlegend=False))
    
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='black', size=8), hoverinfo='skip', showlegend=False))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=bgc["text_col"], size=14), x=0.5, xanchor='center'),
        xaxis=dict(visible=False, range=[-circle_r*1.1, circle_r*1.1], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-circle_r*1.1, circle_r*1.1]),
        plot_bgcolor=bgc["plot_bg"],
        paper_bgcolor=bgc["panel_bg"],
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=(plot_type == "boundaries"),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
        height=500
    )
    
    return fig

# -------------------------
# Grid orders & labels
# -------------------------
LENGTH_ORDER = [
    "FULL_TOSS", "YORKER", "FULL",
    "GOOD_LENGTH", "SHORT_OF_A_GOOD_LENGTH", "SHORT"
]
LENGTH_LABELS = [
    "Full Toss", "Yorker", "Full",
    "Good Length", "Short Of Good Length", "Short"
]

LINE_ORDER_LHB = [
    "WIDE_DOWN_LEG", "DOWN_LEG", "ON_THE_STUMPS", "OUTSIDE_OFFSTUMP", "WIDE_OUTSIDE_OFFSTUMP"
]
LINE_LABELS_LHB = [
    "Wide Down", "Down Leg", "On Stumps", "Outside Off", "Wide Outside"
]

LINE_ORDER_RHB = [
    "WIDE_OUTSIDE_OFFSTUMP", "OUTSIDE_OFFSTUMP", "ON_THE_STUMPS", "DOWN_LEG", "WIDE_DOWN_LEG"
]
LINE_LABELS_RHB = [
    "Wide Outside", "Outside Off", "On Stumps", "Down Leg", "Wide Down"
]

# -------------------------
# Color helpers
# -------------------------
def get_bg_colors(bg="dark"):
    if bg == "white":
        return {
            "page_bg": "#FFFFFF",
            "panel_bg": "#FFFFFF",
            "plot_bg": "#FFFFFF",
            "text_col": "#111111",
            "tile_na": "#FFFFFF",
            "tile_border": "#D9D9D9",
            "accent": "#00E5FF",
            "note_col": "#111111"
        }
    else:
        return {
            "page_bg": "#0b0c2a",
            "panel_bg": "#1F1F22",
            "plot_bg": "#0b0c2a",
            "text_col": "#111111",
            "tile_na": "#1F1F22",
            "tile_border": "#2C2C2E",
            "accent": "#00E5FF",
            "note_col": "#111111"
        }

# -------------------------
# Load data
# -------------------------
import time

@st.cache_resource(show_spinner="Loading 1M+ T20 deliveries... (first load may take some while)")
def load_data(path=DATA_PATH):
    for attempt in range(3):
        try:
            with st.spinner(f"Downloading full dataset... (attempt {attempt+1}/3)"):
                df = pd.read_parquet(path)
            st.success(f"Loaded {len(df):,} deliveries!")
            df.columns = [c.strip() for c in df.columns]
            for c in df.select_dtypes(include=["object"]).columns:
                df[c] = df[c].replace("", np.nan)
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                except:
                    pass
            return df
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {str(e)[:100]}")
            time.sleep(5)
    st.error("Could not load data after 3 attempts. Running with empty dataset.")
    return pd.DataFrame()

# -------------------------
# Empty plot helper
# -------------------------
def empty_pitch_plot(bg="dark", title=""):
    bgc = get_bg_colors(bg)
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0,5], visible=False),
        yaxis=dict(range=[0,6], visible=False, autorange="reversed"),
        plot_bgcolor=bgc["plot_bg"],
        paper_bgcolor=bgc["panel_bg"],
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text=title, font=dict(color=bgc["text_col"]))
    )
    return fig

# -------------------------
# Pitchmap plotter
# -------------------------
def pitchmap_generic(df_results, handedness="RHB", metric="control", bg="white"):
    if handedness == "LHB":
        line_order = LINE_ORDER_LHB
        line_labels_raw = LINE_LABELS_LHB
    else:
        line_order = LINE_ORDER_RHB
        line_labels_raw = LINE_LABELS_RHB

    row_labels = []
    for lab in LENGTH_LABELS:
        if lab == "Short Of Good Length":
            row_labels.append(["Short Of", "Good Length"])
        elif " " in lab:
            row_labels.append(lab.split(" "))
        else:
            row_labels.append([lab])

    col_labels = []
    for lab in line_labels_raw:
        if " " in lab:
            col_labels.append(lab.split(" "))
        else:
            col_labels.append([lab])

    grid = []
    for r_idx, L in enumerate(LENGTH_ORDER, start=1):
        for c_idx, LN in enumerate(line_order, start=1):
            grid.append({"Row": r_idx, "Col": c_idx, "length": L, "line": LN})
    grid_df = pd.DataFrame(grid)

    if df_results is None or df_results.shape[0] == 0:
        merged = grid_df.copy()
        merged["value"] = np.nan
    else:
        df2 = df_results.copy()
        if "length" in df2.columns:
            df2["length"] = df2["length"].astype(str).str.upper().str.strip()
        if "line" in df2.columns:
            df2["line"] = df2["line"].astype(str).str.upper().str.strip()
        df2["value"] = pd.to_numeric(df2["value"], errors="coerce")
        merged = grid_df.merge(df2[["length", "line", "value"]], how="left", on=["length", "line"])

    z_matrix = merged["value"].to_numpy(dtype=float).reshape((6, 5))
    ztext = np.where(np.isnan(z_matrix), "", np.round(z_matrix, 1).astype(str))

    if metric == "control":
        colorscale = [[0.0, "white"], [0.15, "#ff3300"], [0.40, "#ff9900"], [0.65, "#ffff66"], [0.85, "#99ff66"], [1.0, "#33cc33"]]
        zmin, zmax = 0, 100
        title_text = "Control % Pitchmap"
    elif metric == "average":
        vals = merged["value"].dropna()
        max_limit = float(np.ceil(max(vals.max(), 65.0))) if vals.size > 0 else 65.0
        stops = np.array([0, 15, 30, 40, 65, max_limit], dtype=float)
        res = (stops - stops.min()) / (stops.max() - stops.min()) if stops.max() != stops.min() else [0] * len(stops)
        colors = ["white", "#ff3300", "#ff9900", "#ffff66", "#99ff66", "#33cc33"]
        colorscale = [[float(res[i]), colors[i]] for i in range(len(colors))]
        zmin, zmax = 0, max_limit
        title_text = "Average Pitchmap"
    else:
        title_text = "Strike Rate Pitchmap"
        colorscale = [[0.0, "white"], [0.2, "#ff3300"], [0.4, "#ff9900"], [0.6, "#ffff66"], [0.8, "#99ff66"], [1.0, "#33cc33"]]

        def sr_to_bin(sr):
            if pd.isna(sr): return np.nan
            if sr == 0: return 0
            if sr <= 100: return 1
            if sr <= 120: return 2
            if sr <= 140: return 3
            if sr <= 180: return 4
            return 5

        z_matrix = np.vectorize(sr_to_bin, otypes=[float])(z_matrix)
        zmin, zmax = 0, 5

    x_vals = [1, 2, 3, 4, 5]
    y_vals = [1, 2, 3, 4, 5, 6]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z_matrix, x=x_vals, y=y_vals, text=ztext, hoverinfo="text",
                             colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False, xgap=1, ygap=1))

    annotations = []
    for yi, row in enumerate(y_vals):
        for xi, col in enumerate(x_vals):
            annotations.append(dict(x=col, y=row, text=ztext[yi][xi], showarrow=False, xanchor="center", yanchor="middle",
                                    font=dict(color=get_bg_colors(bg)["text_col"], size=13)))

    for idx, lab in enumerate(row_labels, start=1):
        for i, line in enumerate(lab):
            annotations.append(dict(x=0.2, y=idx - 0.25 + (len(lab) - 1 - i) * 0.25, text=line, showarrow=False, xanchor="right",
                                    font=dict(color=get_bg_colors(bg)["text_col"], size=10)))

    for idx, lab in enumerate(col_labels, start=1):
        for i, line in enumerate(lab):
            annotations.append(dict(x=idx, y=6.8 - i * 0.25, text=line, showarrow=False, yanchor="bottom",
                                    font=dict(color=get_bg_colors(bg)["text_col"], size=10)))

    shapes = []
    for r in range(1, 7):
        for c in range(1, 6):
            shapes.append(dict(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                               line=dict(color=get_bg_colors(bg)["tile_border"], width=1), fillcolor="rgba(0,0,0,0)"))

    fig.update_layout(
        title=dict(text=title_text, font=dict(color=get_bg_colors(bg)["text_col"], size=15)),
        xaxis=dict(showgrid=False, zeroline=False, tickmode='array', tickvals=x_vals, range=[0.5, 5.5], visible=False),
        yaxis=dict(showgrid=False, zeroline=False, tickmode='array', tickvals=y_vals, range=[0.5, 6.5], autorange='reversed', visible=False),
        plot_bgcolor=get_bg_colors(bg)["plot_bg"],
        paper_bgcolor=get_bg_colors(bg)["panel_bg"],
        margin=dict(l=80, r=20, t=50, b=30),
        annotations=annotations, shapes=shapes
    )
    return fig

# -------------------------
# Results calculation functions
# -------------------------
def results_control_from_df(df):
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    d = df.dropna(subset=["length", "line"])
    if d.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    d_ctrl = d[~d["control"].isna()] if "control" in d.columns else pd.DataFrame(columns=d.columns)
    if d_ctrl.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    agg = d_ctrl.groupby(["length", "line"]).agg(
        ones=("control", lambda s: (s == 1).sum()),
        total=("control", lambda s: s.notna().sum())
    ).reset_index()
    agg["value"] = 100.0 * agg["ones"] / agg["total"]
    return agg[["length", "line", "value"]]

def results_average_from_df(df):
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    if not ("batruns" in df.columns and "out" in df.columns):
        return pd.DataFrame(columns=["length", "line", "value"])
    d = df.dropna(subset=["length", "line"])
    if d.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    def safe_sum(s):
        return pd.to_numeric(s, errors="coerce").dropna().sum()
    def safe_outs(s):
        s2 = s.dropna().apply(lambda v: True if v is True else (str(v).strip().lower() == "true") or (str(v).strip() == "1"))
        return int(s2.sum())
    agg_runs = d.groupby(["length", "line"])["batruns"].agg(lambda s: safe_sum(s)).reset_index(name="total_runs")
    agg_outs = d.groupby(["length", "line"])["out"].agg(lambda s: safe_outs(s)).reset_index(name="outs")
    agg = pd.merge(agg_runs, agg_outs, on=["length", "line"], how="outer").fillna(np.nan)
    agg["value"] = agg.apply(lambda r: (r["total_runs"] / r["outs"]) if (not pd.isna(r["outs"]) and r["outs"] > 0) else np.nan, axis=1)
    return agg[["length", "line", "value"]]

def results_sr_from_df(df):
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    if "batruns" not in df.columns:
        return pd.DataFrame(columns=["length", "line", "value"])
    d = df.dropna(subset=["length", "line"])
    if d.shape[0] == 0:
        return pd.DataFrame(columns=["length", "line", "value"])
    def safe_sum(s):
        return pd.to_numeric(s, errors="coerce").dropna().sum()
    def safe_count(s):
        return s.notna().sum()
    agg_runs = d.groupby(["length", "line"])["batruns"].agg(lambda s: safe_sum(s)).reset_index(name="total_runs")
    agg_balls = d.groupby(["length", "line"])["batruns"].agg(lambda s: safe_count(s)).reset_index(name="balls")
    agg = pd.merge(agg_runs, agg_balls, on=["length", "line"], how="outer").fillna(np.nan)
    agg["value"] = agg.apply(lambda r: (100.0 * r["total_runs"] / r["balls"]) if (not pd.isna(r["balls"]) and r["balls"] > 0) else np.nan, axis=1)
    return agg[["length", "line", "value"]]

# -------------------------
# Helper: compute raw stats
# -------------------------
def compute_raw_stats(df):
    if df is None or df.shape[0] == 0:
        return None, None, None, None, None, None

    df_clean = df.copy()
    df_clean["batruns"] = pd.to_numeric(df_clean["batruns"], errors="coerce")
    df_clean["ballfaced"] = pd.to_numeric(df_clean["ballfaced"], errors="coerce")
    df_clean = df_clean.dropna(subset=["batruns", "ballfaced"], how="all")

    if "p_match" in df_clean.columns:
        pm = df_clean["p_match"].dropna()
        innings = int(pm.nunique()) if pm.size > 0 else None
    else:
        innings = None

    br = df_clean["batruns"].dropna()
    runs = int(br.sum()) if br.size > 0 else None

    if "out" in df_clean.columns:
        out_series = df_clean["out"].dropna().apply(lambda v: True if v is True else str(v).strip().lower() in ["true", "1"])
        outs_count = int(out_series.sum())
    else:
        outs_count = 0

    avg = float(runs) / outs_count if (runs is not None and outs_count > 0) else None

    bf = df_clean["ballfaced"].dropna()
    total_balls = float(bf.sum()) if bf.size > 0 else None

    sr = (runs * 100.0) / total_balls if (runs is not None and total_balls and total_balls > 0) else None

    dot_count = int((df_clean["batruns"] == 0).sum()) if br.size > 0 else 0
    if total_balls is not None and total_balls > 0:
        dot_pct = (dot_count * 100.0) / total_balls
    else:
        dot_pct = None

    if br.size > 0:
        boundary_count = int(((df_clean["batruns"] == 4) | (df_clean["batruns"] == 6)).sum())
    else:
        boundary_count = 0
    if total_balls is not None and total_balls > 0:
        boundary_pct = (boundary_count * 100.0) / total_balls
    else:
        boundary_pct = None

    return innings, runs, avg, sr, dot_pct, boundary_pct

# -------------------------
# Helper: Shots analysis
# -------------------------
def _find_shot_column(df):
    if df is None or df.shape[0] == 0:
        return None
    cols = list(df.columns)
    candidates = []
    for c in cols:
        cl = str(c).strip().lower()
        if "shot" in cl:
            candidates.append(c)
        if cl in ["type", "shot_type", "shot_type_name", "shotname", "shot_type_detail", "stroke", "stroketype"]:
            if c not in candidates:
                candidates.append(c)
    return candidates[0] if len(candidates) > 0 else None

def build_filtered_df_all():
    df = df_all.copy()
    df = apply_multifilter(df, "team_bat", selected_team_bat)
    df = apply_multifilter(df, "bowl_kind", selected_bowltype)
    df = apply_multifilter(df, "team_bowl", selected_opposition)
    df = apply_multifilter(df, "country", selected_host)
    df = apply_multifilter(df, "bowl", selected_bowler)
    df = apply_multifilter(df, "ground", selected_ground)
    df = apply_multifilter(df, "bowl_style", selected_bowlstyle)
    df = apply_multifilter(df, "competition", selected_tournament)

    if selected_inns and "All" not in selected_inns and "inns" in df.columns:
        try:
            ints = [int(x) for x in selected_inns if x != "All"]
            if len(ints) > 0:
                df = df[df["inns"].isin(ints)]
        except Exception:
            pass

    if "over" in df.columns:
        df = df[(pd.to_numeric(df["over"], errors="coerce") >= over_range[0]) &
                (pd.to_numeric(df["over"], errors="coerce") <= over_range[1])]
    if "cur_bat_bf" in df.columns:
        df = df[(pd.to_numeric(df["cur_bat_bf"], errors="coerce") >= balls_faced[0]) &
                (pd.to_numeric(df["cur_bat_bf"], errors="coerce") <= balls_faced[1])]

    if sel_date_range is not None and "date" in df.columns:
        try:
            start_d, end_d = sel_date_range
            df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
        except Exception:
            pass

    return df

def compute_shots_table(df_batter, df_baseline):
    if df_batter is None or df_batter.shape[0] == 0:
        return pd.DataFrame()

    shot_col = _find_shot_column(df_batter)
    if shot_col is None:
        return pd.DataFrame()

    d = df_batter.copy()
    d[shot_col] = d[shot_col].astype(str).str.strip()
    d.loc[d[shot_col].str.lower().isin(["nan", "none", ""]), shot_col] = np.nan
    d = d.dropna(subset=[shot_col])
    if d.shape[0] == 0:
        return pd.DataFrame()

    baseline = df_baseline.copy() if (df_baseline is not None) else pd.DataFrame()
    if baseline.shape[0] > 0 and shot_col in baseline.columns:
        baseline[shot_col] = baseline[shot_col].astype(str).str.strip()
        baseline.loc[baseline[shot_col].str.lower().isin(["nan", "none", ""]), shot_col] = np.nan
        baseline = baseline.dropna(subset=[shot_col])
    else:
        baseline = pd.DataFrame()

    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce")
    d["ballfaced"] = pd.to_numeric(d["ballfaced"], errors="coerce")
    if "control" in d.columns:
        def _to_ctrl(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().lower()
            if s in ["1", "true", "t", "yes", "y"]:
                return 1.0
            if s in ["0", "false", "f", "no", "n"]:
                return 0.0
            try:
                return float(s)
            except Exception:
                return np.nan
        d["_control_num"] = d["control"].apply(_to_ctrl)
    else:
        d["_control_num"] = np.nan

    if "out" in d.columns:
        outs_series = d["out"].dropna().apply(lambda v: True if v is True else (str(v).strip().lower() in ["true", "1", "t", "yes", "y"]))
        d["_is_out"] = False
        d.loc[outs_series.index, "_is_out"] = outs_series
    else:
        d["_is_out"] = False

    grouped = d.groupby(shot_col).agg(
        Balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Outs = ("_is_out", lambda s: int(s.sum())),
        ControlSum = ("_control_num", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
    ).reset_index()

    grouped["Balls"] = grouped["Balls"].fillna(0).astype(float)
    grouped["Runs"] = grouped["Runs"].fillna(0).astype(float)
    grouped["Outs"] = grouped["Outs"].fillna(0).astype(int)
    grouped["ControlSum"] = grouped["ControlSum"].fillna(0).astype(float)

    if baseline is not None and baseline.shape[0] > 0:
        baseline["batruns"] = pd.to_numeric(baseline["batruns"], errors="coerce")
        baseline["ballfaced"] = pd.to_numeric(baseline["ballfaced"], errors="coerce")
        baseline_group = baseline.groupby(shot_col).agg(
            total_runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
            total_balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
        ).reset_index()
        baseline_group["runs_per_ball"] = baseline_group.apply(lambda r: (r["total_runs"] / r["total_balls"]) if (r["total_balls"] and r["total_balls"] > 0) else np.nan, axis=1)
    else:
        baseline_group = pd.DataFrame(columns=[shot_col, "total_runs", "total_balls", "runs_per_ball"])

    merged = grouped.merge(baseline_group[[shot_col, "runs_per_ball"]], how="left", left_on=shot_col, right_on=shot_col)

    def safe_avg(runs, outs):
        if outs is None or outs == 0:
            return None
        return runs / outs

    merged["Average"] = merged.apply(lambda r: safe_avg(r["Runs"], r["Outs"]), axis=1)
    merged["Strike Rate"] = merged.apply(lambda r: (100.0 * r["Runs"] / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    merged["Control %"] = merged.apply(lambda r: (100.0 * r["ControlSum"] / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    merged["ProbabilityOutPerBall"] = merged.apply(lambda r: (r["Outs"] / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    merged["xRuns"] = merged.apply(lambda r: (r["runs_per_ball"] * r["Balls"]) if (not pd.isna(r.get("runs_per_ball")) and (r["Balls"] and r["Balls"] > 0)) else 0.0, axis=1)
    merged["SVA"] = merged["Runs"] - merged["xRuns"]

    def compute_ranc(row):
        balls = row["Balls"]
        if balls is None or balls == 0:
            return np.nan
        mu = (row["Runs"] / balls) if (not pd.isna(row["Runs"])) else np.nan
        C = (100.0 * row["ControlSum"] / balls) if (balls and balls > 0) else np.nan
        p = row["ProbabilityOutPerBall"] if (not pd.isna(row["ProbabilityOutPerBall"])) else np.nan
        if pd.isna(mu) or pd.isna(C) or pd.isna(p):
            return np.nan
        return (mu * (C / 100.0)) - 10.0 * p

    merged["RANC"] = merged.apply(lambda r: compute_ranc(r), axis=1)

    shot_name_map = {
        "DEFENDED": "Defended",
        "CUT_SHOT": "Cut Shot",
        "FLICK": "Flick",
        "ON_DRIVE": "On Drive",
        "PULL": "Pull",
        "LEG_GLANCE": "Leg Glance",
        "STEERED": "Steered",
        "PUSH": "Push",
        "SQUARE_DRIVE": "Square Drive",
        "LEFT_ALONE": "Left Alone",
        "COVER_DRIVE": "Cover Drive",
        "STRAIGHT_DRIVE": "Straight Drive",
        "SLOG_SHOT": "Slog Shot",
        "DAB": "Dab",
        "UPPER_CUT": "Upper Cut",
        "SWEEP_SHOT": "Sweep Shot",
        "HOOK": "Hook",
        "REVERSE_SWEEP": "Reverse Sweep",
        "VERTICAL_FORWARD_ATTACK": "Vertical Forward Attack",
        "PULL_HOOK_ON_BACK_FOOT": "Pull Hook (Back Foot)",
        "BACK_DEFENCE": "Back Defence",
        "SWEEP": "Sweep",
        "PUSH_SHOT": "Push Shot",
        "FORWARD_DEFENCE": "Forward Defence",
        "NO_SHOT": "No Shot",
        "CUT_SHOT_ON_BACK_FOOT": "Cut Shot (Back Foot)",
        "SLOG_SWEEP": "Slog Sweep",
        "PADDLE_SWEEP": "Paddle Sweep",
        "RAMP": "Ramp",
        "REVERSE_PULL": "Reverse Pull",
        "DROP_AND_RUN": "Drop and Run",
        "PADDLE_AWAY": "Paddle Away",
        "REVERSE_SCOOP": "Reverse Scoop",
        "LATE_CUT": "Late Cut"
    }

    def _friendly_name(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        key = s.upper()
        if key in shot_name_map:
            return shot_name_map[key]
        return s.replace("_", " ").title()

    display = pd.DataFrame()
    display["Shot Type"] = merged[shot_col].apply(_friendly_name)
    display["Balls"] = merged["Balls"].astype(int)
    display["Runs"] = merged["Runs"].astype(int)
    display["xRuns"] = merged["xRuns"].apply(lambda x: round(x,2) if not pd.isna(x) else "N/A")
    def _fmt_sva(x):
        if pd.isna(x):
            return "N/A"
        s = round(x,2)
        return ("+" if s>0 else "") + str(s)
    display["SVA"] = merged["SVA"].apply(_fmt_sva)
    display["Outs"] = merged["Outs"].astype(int)
    def _fmt_avg(v, outs):
        if outs == 0 or pd.isna(v):
            return "-"
        return round(v,2)
    display["Average"] = [ _fmt_avg(a,o) for a,o in zip(merged["Average"], merged["Outs"]) ]
    display["Strike Rate"] = merged["Strike Rate"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    display["Control %"] = merged["Control %"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    total_balls_all = merged["Balls"].sum()
    display["Frequency"] = merged.apply(lambda r: (round((r["Balls"] * 100.0 / total_balls_all),1) if (total_balls_all and total_balls_all>0) else np.nan), axis=1)
    display["RANC"] = merged["RANC"].apply(lambda x: (("+" if x>0 else "") + str(round(x,3))) if not pd.isna(x) else "N/A")

    display = display.sort_values(by="Balls", ascending=False).reset_index(drop=True)

    return display


# -------------------------
# Impact analysis
# -------------------------
def compute_impact_table(filtered_df, master_df):
    if filtered_df is None or filtered_df.shape[0] == 0:
        return None, pd.DataFrame(), None

    if "p_match" not in filtered_df.columns or "p_match" not in master_df.columns:
        return None, pd.DataFrame(), None

    p_matches = filtered_df["p_match"].dropna().unique().tolist()
    if len(p_matches) == 0:
        return None, pd.DataFrame(), None

    impact_rows = master_df[master_df["p_match"].isin(p_matches)].copy()
    if impact_rows.shape[0] == 0:
        return impact_rows, pd.DataFrame(), None

    impact_rows["batruns"] = pd.to_numeric(impact_rows["batruns"], errors="coerce")
    impact_rows["ballfaced"] = pd.to_numeric(impact_rows["ballfaced"], errors="coerce")
    if "control" in impact_rows.columns:
        def _to_ctrl(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().lower()
            if s in ["1", "true", "t", "yes", "y"]:
                return 1.0
            if s in ["0", "false", "f", "no", "n"]:
                return 0.0
            try:
                return float(s)
            except Exception:
                return np.nan
        impact_rows["_control_num"] = impact_rows["control"].apply(_to_ctrl)
    else:
        impact_rows["_control_num"] = np.nan

    if "out" in impact_rows.columns:
        impact_rows["_is_out"] = impact_rows["out"].dropna().apply(lambda v: True if v is True else (str(v).strip().lower() in ["true", "1", "t", "yes", "y"]))
        impact_rows["_is_out"] = impact_rows["_is_out"].fillna(False)
    else:
        impact_rows["_is_out"] = False

    grp = impact_rows.groupby("bat").agg(
        Balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Outs = ("_is_out", lambda s: int(s.sum())),
        Fours = ("batruns", lambda s: int((pd.to_numeric(s, errors="coerce") == 4).sum())),
        Sixes = ("batruns", lambda s: int((pd.to_numeric(s, errors="coerce") == 6).sum())),
        Dots = ("batruns", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
        ControlSum = ("_control_num", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
    ).reset_index()

    grp["Balls"] = grp["Balls"].fillna(0).astype(float)
    grp["Runs"] = grp["Runs"].fillna(0).astype(float)
    grp["Outs"] = grp["Outs"].fillna(0).astype(int)
    grp["Fours"] = grp["Fours"].fillna(0).astype(int)
    grp["Sixes"] = grp["Sixes"].fillna(0).astype(int)
    grp["Dots"] = grp["Dots"].fillna(0).astype(int)
    grp["ControlSum"] = grp["ControlSum"].fillna(0).astype(float)

    def safe_avg(runs, outs):
        if outs is None or outs == 0:
            return None
        return runs / outs

    grp["Average"] = grp.apply(lambda r: safe_avg(r["Runs"], r["Outs"]), axis=1)
    grp["Boundary %"] = grp.apply(lambda r: ((r["Fours"] + r["Sixes"]) * 100.0 / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    grp["Dot ball %"] = grp.apply(lambda r: (r["Dots"] * 100.0 / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    grp["Strike Rate"] = grp.apply(lambda r: ((r["Runs"] * 100.0) / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    grp["Control %"] = grp.apply(lambda r: ((r["ControlSum"] * 100.0) / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)

    total_balls_all = grp["Balls"].sum()
    total_runs_all = grp["Runs"].sum()
    total_fours_all = grp["Fours"].sum()
    total_sixes_all = grp["Sixes"].sum()
    total_control_sum_all = grp["ControlSum"].sum()

    if total_balls_all and total_balls_all > 0:
        overall_boundary_pct = ( (total_fours_all + total_sixes_all) * 100.0 ) / total_balls_all
        overall_sr = ( total_runs_all * 100.0 ) / total_balls_all
        overall_control_pct = ( total_control_sum_all * 100.0 ) / total_balls_all
    else:
        overall_boundary_pct = np.nan
        overall_sr = np.nan
        overall_control_pct = np.nan

    grp["Boundary Impact"] = grp["Boundary %"].apply(lambda x: (x - overall_boundary_pct) if (not pd.isna(x) and not pd.isna(overall_boundary_pct)) else np.nan)
    grp["SR Impact"] = grp["Strike Rate"].apply(lambda x: (x - overall_sr) if (not pd.isna(x) and not pd.isna(overall_sr)) else np.nan)
    grp["Control Impact"] = grp["Control %"].apply(lambda x: (x - overall_control_pct) if (not pd.isna(x) and not pd.isna(overall_control_pct)) else np.nan)

    display = pd.DataFrame()
    display["Batter"] = grp["bat"]
    display["Balls"] = grp["Balls"].astype(int)
    display["Runs"] = grp["Runs"].astype(int)
    def _fmt_avg(v, outs):
        if outs == 0 or pd.isna(v):
            return "-"
        return round(v, 2)
    display["Average"] = [ _fmt_avg(a,o) for a,o in zip(grp["Average"], grp["Outs"]) ]
    display["Outs"] = grp["Outs"].astype(int)
    display["4s"] = grp["Fours"].astype(int)
    display["6s"] = grp["Sixes"].astype(int)
    display["Boundary %"] = grp["Boundary %"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    display["Boundary Impact"] = grp["Boundary Impact"].apply(lambda x: ( ("+" if x>0 else "") + str(round(x,1)) + "%") if not pd.isna(x) else "N/A")
    display["Dot ball %"] = grp["Dot ball %"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    display["Strike Rate"] = grp["Strike Rate"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    display["SR Impact"] = grp["SR Impact"].apply(lambda x: ( ("+" if x>0 else "") + str(round(x,1)) ) if not pd.isna(x) else "N/A")
    display["Control %"] = grp["Control %"].apply(lambda x: (round(x,1) if not pd.isna(x) else "N/A"))
    display["Control Impact"] = grp["Control Impact"].apply(lambda x: ( ("+" if x>0 else "") + str(round(x,1)) + "%") if not pd.isna(x) else "N/A")

    display = display.sort_values(by="Balls", ascending=False).reset_index(drop=True)

    styled = None
    try:
        def _color_impact(val):
            if isinstance(val, str):
                if val.startswith("+"):
                    return "color: green;"
                if val.startswith("-"):
                    return "color: red;"
                return ""
            try:
                v = float(val)
                if v > 0:
                    return "color: green;"
                elif v < 0:
                    return "color: red;"
                else:
                    return ""
            except Exception:
                return ""
        styled = display.style.hide_index().applymap(lambda v: _color_impact(v), subset=["Boundary Impact","SR Impact","Control Impact"])
    except Exception:
        styled = None

    return impact_rows, display, styled

# -------------------------
# Other metrics helpers
# -------------------------
def compute_advanced_shot_metrics(df_batter, df_baseline):
    """
    For the selected batter+filters, compute per-shot other metrics:
    - Balls, Runs
    - xRuns_per_ball (baseline runs per ball for that shot)
    - batter_mu (runs/ball)
    - sd_runs_per_ball, CV, downside deviation (below baseline runs-per-ball)
    - Dismissal rate p (outs per ball), Survival rate (1 - p)
    - Frequency %
    """
    if df_batter is None or df_batter.shape[0] == 0:
        return pd.DataFrame()
    shot_col = _find_shot_column(df_batter)
    if shot_col is None:
        return pd.DataFrame()

    df = df_batter.copy()
    df[shot_col] = df[shot_col].astype(str).str.strip()
    df.loc[df[shot_col].str.lower().isin(["nan", "none", ""]), shot_col] = np.nan
    df = df.dropna(subset=[shot_col])
    if df.shape[0] == 0:
        return pd.DataFrame()

    baseline = df_baseline.copy() if (df_baseline is not None) else pd.DataFrame()
    if baseline.shape[0] > 0 and shot_col in baseline.columns:
        baseline[shot_col] = baseline[shot_col].astype(str).str.strip()
        baseline.loc[baseline[shot_col].str.lower().isin(["nan", "none", ""]), shot_col] = np.nan
        baseline = baseline.dropna(subset=[shot_col])
    else:
        baseline = pd.DataFrame()

    df["batruns"] = pd.to_numeric(df["batruns"], errors="coerce")
    df["ballfaced"] = pd.to_numeric(df["ballfaced"], errors="coerce")
    if "out" in df.columns:
        df["_is_out"] = df["out"].dropna().apply(lambda v: True if v is True else (str(v).strip().lower() in ["true","1","t","yes","y"]))
    else:
        df["_is_out"] = False

    if baseline is not None and baseline.shape[0] > 0:
        baseline["batruns"] = pd.to_numeric(baseline["batruns"], errors="coerce")
        baseline["ballfaced"] = pd.to_numeric(baseline["ballfaced"], errors="coerce")
        baseline_group = baseline.groupby(shot_col).agg(
            total_runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
            total_balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
        ).reset_index()
        baseline_group["runs_per_ball"] = baseline_group.apply(lambda r: (r["total_runs"] / r["total_balls"]) if (r["total_balls"] and r["total_balls"] > 0) else np.nan, axis=1)
    else:
        baseline_group = pd.DataFrame(columns=[shot_col,"total_runs","total_balls","runs_per_ball"])

    grp_runs = df.groupby(shot_col).agg(
        Balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Outs = ("_is_out", lambda s: int(s.sum()))
    ).reset_index()

    grp_runs = grp_runs.merge(baseline_group[[shot_col, "runs_per_ball"]], how="left", left_on=shot_col, right_on=shot_col)
    grp_runs["mu"] = grp_runs.apply(lambda r: (r["Runs"] / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    grp_runs["p_out_per_ball"] = grp_runs.apply(lambda r: (r["Outs"] / r["Balls"]) if (r["Balls"] and r["Balls"] > 0) else np.nan, axis=1)
    grp_runs["survival_rate"] = grp_runs.apply(lambda r: (1.0 - r["p_out_per_ball"]) if (not pd.isna(r["p_out_per_ball"])) else np.nan, axis=1)

    sd_list = []
    downside_list = []
    for shot_val in grp_runs[shot_col].tolist():
        arr = df[df[shot_col] == shot_val]["batruns"].dropna().astype(float).values
        if arr.size == 0:
            sd_list.append(np.nan)
            downside_list.append(np.nan)
            continue
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        sd_list.append(sd)
        baseline_rpb = grp_runs.loc[grp_runs[shot_col] == shot_val, "runs_per_ball"].values
        bval = baseline_rpb[0] if baseline_rpb.size > 0 else np.nan
        if np.isnan(bval):
            downside_list.append(np.nan)
        else:
            diffs = np.minimum(0, arr - bval)
            downside = float(np.sqrt(np.mean(diffs**2))) if diffs.size > 0 else np.nan
            downside_list.append(downside)

    grp_runs["sd_runs_per_ball"] = sd_list
    grp_runs["CV"] = grp_runs.apply(lambda r: (r["sd_runs_per_ball"] / r["mu"]) if (not pd.isna(r["sd_runs_per_ball"]) and not pd.isna(r["mu"]) and r["mu"]!=0) else np.nan, axis=1)
    grp_runs["downside_deviation"] = downside_list

    grp_runs["xRuns_per_ball"] = grp_runs["runs_per_ball"]
    grp_runs["xRuns"] = grp_runs.apply(lambda r: (r["xRuns_per_ball"] * r["Balls"]) if (not pd.isna(r["xRuns_per_ball"]) and (r["Balls"] and r["Balls"]>0)) else 0.0, axis=1)

    total_balls_all = grp_runs["Balls"].sum()
    grp_runs["Frequency %"] = grp_runs.apply(lambda r: (100.0 * r["Balls"] / total_balls_all) if (total_balls_all and total_balls_all>0) else np.nan, axis=1)

    def _friendly(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        key = s.upper()
        shot_map = {
            "DEFENDED": "Defended",
            "CUT_SHOT": "Cut Shot",
            "FLICK": "Flick",
            "ON_DRIVE": "On Drive",
            "PULL": "Pull",
            "LEG_GLANCE": "Leg Glance",
            "STEERED": "Steered",
            "PUSH": "Push",
            "SQUARE_DRIVE": "Square Drive",
            "LEFT_ALONE": "Left Alone",
            "COVER_DRIVE": "Cover Drive",
            "STRAIGHT_DRIVE": "Straight Drive",
            "SLOG_SHOT": "Slog Shot",
            "DAB": "Dab",
            "UPPER_CUT": "Upper Cut",
            "SWEEP_SHOT": "Sweep Shot",
            "HOOK": "Hook",
            "REVERSE_SWEEP": "Reverse Sweep",
            "VERTICAL_FORWARD_ATTACK": "Vertical Forward Attack",
            "LATE_CUT": "Late Cut"
        }
        return shot_map.get(key, s.replace("_", " ").title())

    disp = pd.DataFrame()
    disp["Shot Type"] = grp_runs[shot_col].apply(_friendly)
    disp["Balls"] = grp_runs["Balls"].astype(int)
    disp["Runs"] = grp_runs["Runs"].astype(int)
    disp["xRuns per ball"] = grp_runs["xRuns_per_ball"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["xRuns"] = grp_runs["xRuns"].apply(lambda x: round(x,2) if not pd.isna(x) else "N/A")
    disp["μ (runs/ball)"] = grp_runs["mu"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["sd (runs/ball)"] = grp_runs["sd_runs_per_ball"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["CV"] = grp_runs["CV"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["Downside Dev"] = grp_runs["downside_deviation"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["Dismissal rate (p)"] = grp_runs["p_out_per_ball"].apply(lambda x: round(x,4) if not pd.isna(x) else "N/A")
    disp["Survival rate"] = grp_runs["survival_rate"].apply(lambda x: round(x,4) if not pd.isna(x) else "N/A")
    disp["Frequency %"] = grp_runs["Frequency %"].apply(lambda x: round(x,1) if not pd.isna(x) else "N/A")

    disp = disp.sort_values(by="Balls", ascending=False).reset_index(drop=True)
    return disp

def compute_advanced_zone_metrics(df_batter):
    """
    Aggregates by length and line (grid) and computes metrics:
    - Balls, Runs, xRuns per ball (computed relative to baseline across same filters but all batters)
    - sd, CV, downside deviation, dismissal rate, frequency
    Ignores rows where length or line are blank/NA.
    """
    # Defensive checks
    if df_batter is None or df_batter.shape[0] == 0:
        return pd.DataFrame()
    # Ensure required columns exist
    if "length" not in df_batter.columns or "line" not in df_batter.columns:
        return pd.DataFrame()

    # Work on a local copy
    df = df_batter.copy()

    # Normalize strings and convert placeholder strings to real NaN
    df["length"] = df["length"].astype(str).str.upper().str.strip()
    df["line"] = df["line"].astype(str).str.upper().str.strip()

    missing_vals = ["", "NAN", "NONE", "NA", "NULL", "-", "NONE "]
    df.loc[df["length"].isin(missing_vals), "length"] = np.nan
    df.loc[df["line"].isin(missing_vals), "line"] = np.nan

    # Drop rows missing either length or line
    d = df.dropna(subset=["length", "line"])
    if d.shape[0] == 0:
        return pd.DataFrame()

    # Convert numeric fields
    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce")
    d["ballfaced"] = pd.to_numeric(d["ballfaced"], errors="coerce")

    # Out flag
    if "out" in d.columns:
        d["_is_out"] = d["out"].dropna().apply(lambda v: True if v is True else (str(v).strip().lower() in ["true","1","t","yes","y"]))
    else:
        d["_is_out"] = False

    # Build baseline grouped metrics (from current global filters excluding batter)
    baseline = build_filtered_df_all()
    if baseline is not None and baseline.shape[0] > 0 and "length" in baseline.columns and "line" in baseline.columns:
        baseline_zone = baseline.copy()
        baseline_zone["length"] = baseline_zone["length"].astype(str).str.upper().str.strip()
        baseline_zone["line"] = baseline_zone["line"].astype(str).str.upper().str.strip()
        baseline_zone.loc[baseline_zone["length"].isin(missing_vals), "length"] = np.nan
        baseline_zone.loc[baseline_zone["line"].isin(missing_vals), "line"] = np.nan
        baseline_zone = baseline_zone.dropna(subset=["length", "line"])
        baseline_zone["batruns"] = pd.to_numeric(baseline_zone["batruns"], errors="coerce")
        baseline_zone["ballfaced"] = pd.to_numeric(baseline_zone["ballfaced"], errors="coerce")
        baseline_group = baseline_zone.groupby(["length","line"]).agg(
            total_runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
            total_balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
        ).reset_index()
        baseline_group["runs_per_ball"] = baseline_group.apply(lambda r: (r["total_runs"] / r["total_balls"]) if (r["total_balls"] and r["total_balls"]>0) else np.nan, axis=1)
    else:
        baseline_group = pd.DataFrame(columns=["length","line","total_runs","total_balls","runs_per_ball"])

    # Group selected batter's data by length and line
    grp = d.groupby(["length","line"]).agg(
        Balls = ("ballfaced", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Runs = ("batruns", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        Outs = ("_is_out", lambda s: int(s.sum()))
    ).reset_index()

    if grp.shape[0] == 0:
        return pd.DataFrame()

    # Merge with baseline runs_per_ball
    grp = grp.merge(baseline_group[["length","line","runs_per_ball"]], how="left", on=["length","line"])

    # Compute basic metrics
    grp["mu"] = grp.apply(lambda r: (r["Runs"] / r["Balls"]) if (r["Balls"] and r["Balls"]>0) else np.nan, axis=1)
    grp["p_out_per_ball"] = grp.apply(lambda r: (r["Outs"] / r["Balls"]) if (r["Balls"] and r["Balls"]>0) else np.nan, axis=1)
    grp["survival_rate"] = grp.apply(lambda r: (1.0 - r["p_out_per_ball"]) if (not pd.isna(r["p_out_per_ball"])) else np.nan, axis=1)

    # Compute sd and downside deviation per group
    sd_list = []
    downside_list = []
    for idx, row in grp.iterrows():
        L = row["length"]
        LN = row["line"]
        arr = d[(d["length"] == L) & (d["line"] == LN)]["batruns"].dropna().astype(float).values
        if arr.size == 0:
            sd_list.append(np.nan)
            downside_list.append(np.nan)
            continue
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        sd_list.append(sd)
        baseline_rpb = row.get("runs_per_ball", np.nan)
        if pd.isna(baseline_rpb):
            downside_list.append(np.nan)
        else:
            diffs = np.minimum(0, arr - baseline_rpb)
            downside = float(np.sqrt(np.mean(diffs**2))) if diffs.size > 0 else np.nan
            downside_list.append(downside)

    grp["sd_runs_per_ball"] = sd_list
    grp["CV"] = grp.apply(lambda r: (r["sd_runs_per_ball"] / r["mu"]) if (not pd.isna(r["sd_runs_per_ball"]) and not pd.isna(r["mu"]) and r["mu"]!=0) else np.nan, axis=1)
    grp["downside_deviation"] = downside_list

    # xRuns and expected runs
    grp["xRuns_per_ball"] = grp["runs_per_ball"]
    grp["Expected Runs (xRuns)"] = grp.apply(lambda r: (r["xRuns_per_ball"] * r["Balls"]) if (not pd.isna(r["xRuns_per_ball"]) and (r["Balls"] and r["Balls"]>0)) else 0.0, axis=1)

    total_balls_all = grp["Balls"].sum()
    grp["Frequency %"] = grp.apply(lambda r: (100.0 * r["Balls"] / total_balls_all) if (total_balls_all and total_balls_all>0) else np.nan, axis=1)

    # Prepare display dataframe with friendly labels
    disp = pd.DataFrame()
    disp["Length"] = grp["length"]
    disp["Line"] = grp["line"]
    disp["Balls"] = grp["Balls"].astype(int)
    disp["Runs"] = grp["Runs"].astype(int)
    disp["xRuns per ball"] = grp["xRuns_per_ball"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["Expected Runs (xRuns)"] = grp["Expected Runs (xRuns)"].apply(lambda x: round(x,2) if not pd.isna(x) else "N/A")
    disp["μ (runs/ball)"] = grp["mu"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["sd (runs/ball)"] = grp["sd_runs_per_ball"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["CV"] = grp["CV"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["Downside Dev"] = grp["downside_deviation"].apply(lambda x: round(x,3) if not pd.isna(x) else "N/A")
    disp["Dismissal rate (p)"] = grp["p_out_per_ball"].apply(lambda x: round(x,4) if not pd.isna(x) else "N/A")
    disp["Survival rate"] = grp["survival_rate"].apply(lambda x: round(x,4) if not pd.isna(x) else "N/A")
    disp["Frequency %"] = grp["Frequency %"].apply(lambda x: round(x,1) if not pd.isna(x) else "N/A")

    disp = disp.sort_values(by=["Balls"], ascending=False).reset_index(drop=True)
    return disp

# -------------------------
# Inns progression helpers
# -------------------------
def plot_strike_rate_progression(df, window=10):
    """
    Plot rolling strike rate over a fixed window (in balls).
    'window' is the number of most recent balls used for the rolling calculation.
    """
    if df is None or df.shape[0] == 0:
        return go.Figure()
    d = df.copy()
    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce")
    d["ballfaced"] = pd.to_numeric(d["ballfaced"], errors="coerce")
    if "over" in d.columns and "ball" in d.columns:
        try:
            d["over_num"] = pd.to_numeric(d["over"], errors="coerce").fillna(0)
            d["ball_num"] = pd.to_numeric(d["ball"], errors="coerce").fillna(0)
            d = d.sort_values(by=["p_match","over_num","ball_num"]).reset_index(drop=True)
        except Exception:
            d = d.reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    d["ball_index"] = d.groupby(["p_match","inns"]).cumcount()+1

    traces = []
    for (pm, inns), grp in d.groupby(["p_match","inns"]):
        if grp.shape[0] == 0:
            continue
        use_window = min(window, max(1, grp.shape[0]))  # can't be larger than available balls
        ser_runs = grp["batruns"].fillna(0)
        # rolling sum of runs over 'use_window' balls divided by use_window => runs/ball, *100 => SR
        rolling_runs = ser_runs.rolling(window=use_window, min_periods=1).sum()
        # number of balls in each rolling window (for start) — conservative: use use_window or the current index
        rolling_counts = grp["ball_index"].rolling(window=use_window, min_periods=1).apply(lambda x: len(x))
        rolling_sr = (rolling_runs / rolling_counts) * 100.0
        traces.append(go.Scatter(x=grp["ball_index"], y=rolling_sr, mode="lines+markers", name=f"Match {pm} Inns {inns} (w={use_window})", hoverinfo="x+y+name"))
    fig = go.Figure(traces)
    fig.update_layout(title=f"Rolling Strike Rate progression (window = {window} balls)", xaxis_title="Ball Index (within innings)", yaxis_title="Strike Rate")
    return fig


def plot_cumulative_runs(df):
    if df is None or df.shape[0] == 0:
        return go.Figure()
    d = df.copy()
    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce")
    d = d.reset_index(drop=True)
    d["ball_index"] = d.groupby(["p_match","inns"]).cumcount()+1
    traces = []
    for (pm, inns), grp in d.groupby(["p_match","inns"]):
        traces.append(go.Scatter(x=grp["ball_index"], y=grp["batruns"].cumsum(), mode="lines+markers", name=f"Match {pm} Inns {inns}"))
    fig = go.Figure(traces)
    fig.update_layout(title="Cumulative runs vs Ball Index (per innings)", xaxis_title="Ball Index", yaxis_title="Cumulative Runs")
    return fig

def plot_boundaries_and_dots(df, window=6):
    """
    Rolling counts of boundaries and dot balls using a window in balls.
    """
    if df is None or df.shape[0] == 0:
        return go.Figure()
    d = df.copy()
    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce").fillna(0)
    d = d.reset_index(drop=True)
    d["ball_index"] = d.groupby(["p_match","inns"]).cumcount()+1
    traces = []
    for (pm, inns), grp in d.groupby(["p_match","inns"]):
        use_window = min(window, max(1, grp.shape[0]))
        is_boundary = grp["batruns"].isin([4,6]).astype(int)
        is_dot = (grp["batruns"] == 0).astype(int)
        roll_bound = is_boundary.rolling(window=use_window, min_periods=1).sum()
        roll_dot = is_dot.rolling(window=use_window, min_periods=1).sum()
        traces.append(go.Scatter(x=grp["ball_index"], y=roll_bound, mode="lines", name=f"Boundaries M{pm}I{inns} (w={use_window})", hoverinfo="x+y+name", line=dict(dash="solid")))
        traces.append(go.Scatter(x=grp["ball_index"], y=roll_dot, mode="lines", name=f"Dots M{pm}I{inns} (w={use_window})", hoverinfo="x+y+name", line=dict(dash="dot")))
    fig = go.Figure(traces)
    fig.update_layout(title=f"Rolling boundaries (count) and dot balls (count) over window {window}", xaxis_title="Ball Index", yaxis_title=f"Count (window={window})")
    return fig

def approx_win_probability_second_innings(df):
    """
    Approximate Win Probability for second innings only:
    Simple heuristic using runs remaining, balls remaining and wickets remaining if available.
    WP = sigmoid( alpha * (RRR_diff) + beta * wickets_term )
    This is an approximation and will be labeled clearly.
    """
    if df is None or df.shape[0] == 0:
        return go.Figure()
    if "inns" not in df.columns:
        return go.Figure()
    d = df.copy()
    d2 = d[d["inns"].astype(str) == "2"].copy()
    if d2.shape[0] == 0:
        return go.Figure()
    plots = []
    for pm, grp in d2.groupby("p_match"):
        grp = grp.sort_values(by=["over"] if "over" in grp.columns else grp.index).reset_index(drop=True)
        grp["cum_runs_bat"] = grp["batruns"].cumsum()
        try:
            opp_rows = df[(df["p_match"] == pm) & (df["inns"].astype(str) == "1")]
            target = None
            if opp_rows.shape[0] > 0 and "batruns" in opp_rows.columns:
                target = opp_rows["batruns"].sum() + 1
            else:
                if "target" in grp.columns:
                    target = grp["target"].iloc[0]
            if target is None:
                continue
        except Exception:
            continue
        total_balls = 120
        grp["ball_index"] = grp.groupby("p_match").cumcount()+1
        grp["balls_left"] = total_balls - grp["ball_index"]
        grp["runs_needed"] = target - grp["cum_runs_bat"]
        grp["rrr"] = grp.apply(lambda r: (r["runs_needed"] * 6.0 / r["balls_left"]) if r["balls_left"]>0 else 999.0, axis=1)
        grp["crr"] = grp.apply(lambda r: (r["cum_runs_bat"] * 6.0 / r["ball_index"]) if r["ball_index"]>0 else 0.0, axis=1)
        grp["state_score"] = -0.3 * grp["runs_needed"] - 0.5 * grp["balls_left"]
        grp["wp"] = 1.0 / (1.0 + np.exp(-0.01 * grp["state_score"]))
        plots.append(go.Scatter(x=grp["ball_index"], y=grp["wp"], mode="lines+markers", name=f"Match {pm}", hoverinfo="x+y+name"))
    fig = go.Figure(plots)
    fig.update_layout(title="Approx Win Probability (2nd innings) — heuristic approximation", xaxis_title="Ball Index", yaxis_title="Win Probability")
    return fig

# -------------------------
# Helper: format stat values
# -------------------------
def fmt(x, fmt_round=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if fmt_round is None:
        return str(x)
    else:
        return str(round(x, fmt_round))

# -------------------------
# Helper: apply multifilter
# -------------------------
def apply_multifilter(df, colname, selected_list):
    if colname not in df.columns or selected_list is None:
        return df
    sel = list(selected_list) if isinstance(selected_list, (list, tuple)) else [selected_list]
    if "All" in sel and len(sel) > 1:
        sel = [s for s in sel if s != "All"]
    if (len(sel) == 0) or (len(sel) == 1 and sel[0] == "All"):
        return df
    return df[df[colname].isin(sel)]

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Men's T20s: Batters' Analysis", layout="wide")

base_css = f"""
<style>
body {{ background-color: {get_bg_colors('dark')['page_bg']}; color: {get_bg_colors('dark')['text_col']}; font-family: Helvetica, Arial, sans-serif; }}
.sidebar .css-1d391kg {{ background-color: #232326 !important; border-radius: 8px; padding: 12px; }}
.main-panel-white {{ background-color: #FFFFFF !important; color: #111111 !important; }}
.custom-note {{ color: {get_bg_colors('dark')['note_col']}; font-size:15px; line-height:1.4; }}
.btn-primary {{ background-color: #00E5FF !important; color: #4a4a4a !important; }}

/* Navigation button styles */
.nav-button {{
    display: inline-block;
    padding: 12px 24px;
    margin: 4px;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFE5E5 100%);
    border: 2px solid #CC0000;
    border-radius: 8px;
    color: #CC0000;
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
}}

.nav-button:hover {{
    background: linear-gradient(135deg, #FFE5E5 0%, #FFCCCC 100%);
    border-color: #990000;
    color: #990000;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}}

.nav-button-active {{
    background: linear-gradient(135deg, #CC0000 0%, #990000 100%);
    color: #FFFFFF;
    border-color: #CC0000;
}}

.nav-button-active:hover {{
    background: linear-gradient(135deg, #990000 0%, #770000 100%);
    color: #FFFFFF;
}}
.svtable-wrapper {{ overflow-x:auto; }}
.svtable-wrapper table {{ width:100%; table-layout:auto; border-collapse:collapse; }}
.svtable-wrapper th, .svtable-wrapper td {{ padding:6px 8px; border:1px solid #eee; text-align:left; vertical-align:top; }}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

st.title("Men's T20s: Batters' Analysis")

df_all = load_data()
if df_all.shape[0] == 0:
    st.error(f"Could not find data at '{DATA_PATH}'. Place your CSV there.")
    st.stop()

# -------------------------
# Sidebar filters setup
# -------------------------
def get_sorted_unique(col):
    if col in df_all.columns:
        vals = df_all[col].dropna().unique().tolist()
        vals_sorted = sorted(vals)
        return vals_sorted
    return []

batters = get_sorted_unique("bat")
bowl_kind_choices = ["All"] + get_sorted_unique("bowl_kind")
team_choices = ["All"] + get_sorted_unique("team_bowl")
country_choices = ["All"] + get_sorted_unique("country")
bowler_choices = ["All"] + get_sorted_unique("bowl")
ground_choices = ["All"] + get_sorted_unique("ground")
bowlstyle_choices = ["All"] + get_sorted_unique("bowl_style")
inns_choices = ["All"] + [str(x) for x in sorted(df_all["inns"].dropna().unique().astype(int))] if "inns" in df_all.columns else ["All"]
team_bat_choices = ["All"] + get_sorted_unique("team_bat")
tournament_choices = ["All"] + get_sorted_unique("competition")

min_date = df_all["date"].min() if "date" in df_all.columns else None
max_date = df_all["date"].max() if "date" in df_all.columns else None

min_over = 1
max_over = 20
default_over = (min_over, max_over)
min_bf = int(df_all["cur_bat_bf"].dropna().min()) if "cur_bat_bf" in df_all.columns else 1
max_bf = int(df_all["cur_bat_bf"].dropna().max()) if "cur_bat_bf" in df_all.columns else 120
default_bf = (min_bf, max_bf)

with st.sidebar:
    st.header("Filters")
    selected_batter = st.selectbox("Select Batter:", options=batters)
    selected_team_bat = st.multiselect("For Team:", options=team_bat_choices, default=["All"])
    selected_opposition = st.multiselect("Opposition:", options=team_choices, default=["All"])
    selected_bowltype = st.multiselect("Bowler Type:", options=bowl_kind_choices, default=["All"])
    selected_bowler = st.multiselect("Bowler:", options=bowler_choices, default=["All"])
    selected_tournament = st.multiselect("Tournament:", options=tournament_choices, default=["All"])
    selected_host = st.multiselect("Host Country:", options=country_choices, default=["All"])
    selected_ground = st.multiselect("Ground:", options=ground_choices, default=["All"])

    if min_date is not None and max_date is not None:
        sel_date_range = st.date_input("Select Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        sel_date_range = None

    st.markdown("---")
    st.markdown("### Advanced filters")
    selected_bowlstyle = st.multiselect("Bowling Style:", options=bowlstyle_choices, default=["All"])
    selected_inns = st.multiselect("Innings:", options=inns_choices, default=["All"])
    balls_faced = st.slider("Balls faced:", min_bf, max_bf, value=default_bf)
    over_range = st.slider("Over range:", min_over, max_over, value=default_over)

    # add these two lines into the sidebar block (after over_range or where appropriate)
    sr_window = st.slider("SR rolling window (balls): For Inns progression", 1, 30, 10, help="Window size (in balls) used for rolling strike-rate")
    bd_window = st.slider("Boundaries/Dots window (balls): For Inns progression", 1, 12, 6, help="Window (in balls) used to compute rolling boundary/dot counts")


    run_btn = st.button("Generate")

if "run_pressed" not in st.session_state:
    st.session_state["run_pressed"] = False
if run_btn:
    st.session_state["run_pressed"] = True

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Pitchmaps"

# -------------------------
# build_filtered_df (unchanged)
# -------------------------
def build_filtered_df():
    df = df_all.copy()
    if selected_batter is None:
        return pd.DataFrame()
    df = df[df["bat"].astype(str) == str(selected_batter)]

    df = apply_multifilter(df, "team_bat", selected_team_bat)
    df = apply_multifilter(df, "bowl_kind", selected_bowltype)
    df = apply_multifilter(df, "team_bowl", selected_opposition)
    df = apply_multifilter(df, "country", selected_host)
    df = apply_multifilter(df, "bowl", selected_bowler)
    df = apply_multifilter(df, "ground", selected_ground)
    df = apply_multifilter(df, "bowl_style", selected_bowlstyle)
    df = apply_multifilter(df, "competition", selected_tournament)

    if selected_inns and "All" not in selected_inns and "inns" in df.columns:
        try:
            ints = [int(x) for x in selected_inns if x != "All"]
            if len(ints) > 0:
                df = df[df["inns"].isin(ints)]
        except Exception:
            pass

    if "over" in df.columns:
        df = df[(pd.to_numeric(df["over"], errors="coerce") >= over_range[0]) &
                (pd.to_numeric(df["over"], errors="coerce") <= over_range[1])]
    if "cur_bat_bf" in df.columns:
        df = df[(pd.to_numeric(df["cur_bat_bf"], errors="coerce") >= balls_faced[0]) &
                (pd.to_numeric(df["cur_bat_bf"], errors="coerce") <= balls_faced[1])]

    if sel_date_range is not None and "date" in df.columns:
        try:
            start_d, end_d = sel_date_range
            df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
        except Exception:
            pass

    return df

# -------------------------
# MAIN PANEL
# -------------------------
main_container = st.container()
with main_container:
    panel_bg = "white" if st.session_state["run_pressed"] else "dark"
    note_col = get_bg_colors(panel_bg)["note_col"]

    st.markdown(f'<div class="main-panel-white" style="padding:15px; border-radius:8px; background-color:{get_bg_colors(panel_bg)["panel_bg"]};">', unsafe_allow_html=True)

    if not st.session_state["run_pressed"]:
        # Initial landing content (rich)
        intro_col_text_color = get_bg_colors("dark")["text_col"]
        note_col = get_bg_colors("dark")["note_col"]
        st.markdown(
            f'''
            <div style="text-align:center; margin-top:12px;">
                <p style="font-size:20px; color:{intro_col_text_color}; margin-bottom:14px;">
                    To view raw stats, Pitchmaps, Wagon wheels, Impact and several other comparison metrics,
                    start by selecting a batter and set suitable filters.
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # slight gap
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # small gap before developer credit
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # Developer credit rendered with components.html for reliable HTML/SVG rendering
        dev_note_color = note_col  # uses earlier defined note_col
        dev_html = f"""
        <div style="font-size:15px; color:{dev_note_color}; display:flex; align-items:center; gap:12px; justify-content:center;">
          <div style="text-align:center;">
            <div style="font-weight:700; margin-bottom:4px;">Developed by Rhitankar Bandyopadhyay</div>
            <div style="display:flex; align-items:center; gap:12px; justify-content:center;">
              <!-- Twitter / X -->
              <a href="https://x.com/_rhitankar_" target="_blank" rel="noopener" style="text-decoration:none; color:{dev_note_color}; display:flex; align-items:center; gap:6px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="{dev_note_color}" aria-hidden="true">
                  <path d="M22.46 6c-.77.35-1.6.58-2.46.69a4.3 4.3 0 0 0 1.88-2.37 8.6 8.6 0 0 1-2.72 1.04 4.28 4.28 0 0 0-7.3 3.9A12.14 12.14 0 0 1 3.15 4.6a4.27 4.27 0 0 0 1.33 5.71 4.24 4.24 0 0 1-1.94-.54v.06a4.28 4.28 0 0 0 3.43 4.2c-.48.13-.98.2-1.5.2-.37 0-.73-.03-1.08-.1a4.29 4.29 0 0 0 4 2.98A8.59 8.59 0 0 1 2 19.54 12.12 12.12 0 0 0 8.29 21c7.55 0 11.69-6.26 11.69-11.69l-.01-.53A8.36 8.36 0 0 0 22.46 6z"/>
                </svg>
                <span style="font-weight:500;">@_rhitankar_</span>
              </a>

              <!-- GitHub -->
              <a href="https://github.com/rhitankar8616" target="_blank" rel="noopener" style="text-decoration:none; color:{dev_note_color}; display:flex; align-items:center; gap:6px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="{dev_note_color}" aria-hidden="true">
                  <path d="M12 .5C5.37.5 0 5.86 0 12.5c0 5.29 3.44 9.77 8.21 11.36.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.73-4.04-1.61-4.04-1.61-.54-1.38-1.32-1.75-1.32-1.75-1.08-.74.08-.73.08-.73 1.2.09 1.83 1.23 1.83 1.23 1.06 1.82 2.78 1.3 3.46.99.11-.77.42-1.3.76-1.6-2.66-.3-5.46-1.33-5.46-5.92 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.31 1.23a11.5 11.5 0 0 1 6.02 0c2.3-1.55 3.31-1.23 3.31-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.6-2.8 5.61-5.47 5.91.43.37.82 1.1.82 2.22 0 1.6-.01 2.89-.01 3.28 0 .32.21.69.83.57C20.56 22.27 24 17.8 24 12.5 24 5.86 18.63.5 12 .5z"/>
                </svg>
                <span style="font-weight:500;">@rhitankar8616</span>
              </a>
            </div>
          </div>
        </div>
        """
        # render the HTML block via components.html for consistent display (height tuned)
        components.html(dev_html, height=120, scrolling=False)

        # small spacer to separate from the rest of the page
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    else:
        batter_name = selected_batter if selected_batter is not None else ""
        st.markdown(f'<h2 style="margin-top:3px; margin-bottom:6px; color:{get_bg_colors("white")["text_col"]};">Batter : {batter_name}</h2>', unsafe_allow_html=True)

        # Navigation: now 6 buttons, with Inns progression before Other metrics
        st.markdown('<div style="text-align:center; margin:20px 0;">', unsafe_allow_html=True)
        btn_cols = st.columns([1,1,1,1,1,1])

        with btn_cols[0]:
            if st.button("Pitchmaps", key="btn_pitchmaps"):
                st.session_state["active_tab"] = "Pitchmaps"
        with btn_cols[1]:
            if st.button("Wagon Wheels", key="btn_wagon"):
                st.session_state["active_tab"] = "Wagon Wheels"
        with btn_cols[2]:
            if st.button("Shots Analysis", key="btn_shots"):
                st.session_state["active_tab"] = "Shots Analysis"
        with btn_cols[3]:
            if st.button("Impact", key="btn_impact"):
                st.session_state["active_tab"] = "Impact"
        with btn_cols[4]:
            if st.button("Inns Progression", key="btn_innings"):
                st.session_state["active_tab"] = "Inns Progression"
        with btn_cols[5]:
            if st.button("Other metrics", key="btn_adv"):
                st.session_state["active_tab"] = "Other metrics"

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # filtered dataframes
        filtered = build_filtered_df()
        baseline_filtered = build_filtered_df_all()

        # RAW stats
        innings, runs, avg_val, sr_val, dot_pct, boundary_pct = compute_raw_stats(filtered)

        stat_cols = st.columns([1, 1, 1, 1, 1, 1])
        with stat_cols[0]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(innings)}</div><div>Innings</div></div>", unsafe_allow_html=True)
        with stat_cols[1]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(runs)}</div><div>Runs</div></div>", unsafe_allow_html=True)
        with stat_cols[2]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(avg_val, 2)}</div><div>Average</div></div>", unsafe_allow_html=True)
        with stat_cols[3]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(sr_val, 1)}</div><div>Strike Rate</div></div>", unsafe_allow_html=True)
        with stat_cols[4]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(dot_pct, 1)}</div><div>Dot ball %</div></div>", unsafe_allow_html=True)
        with stat_cols[5]:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:20px; font-weight:bold'>{fmt(boundary_pct, 1)}</div><div>Boundary %</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Tabs logic
        if st.session_state["active_tab"] == "Pitchmaps":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if filtered.shape[0] == 0 or "control" not in filtered.columns:
                    st.plotly_chart(empty_pitch_plot(bg="white"))
                else:
                    fc = filtered.copy()
                    fc["length"] = fc["length"].astype(str).str.upper().str.strip()
                    fc["line"] = fc["line"].astype(str).str.upper().str.strip()
                    ctrl_res = results_control_from_df(fc)
                    if ctrl_res.shape[0] == 0 or ctrl_res["value"].dropna().size == 0:
                        st.plotly_chart(empty_pitch_plot(bg="white"))
                    else:
                        handed = fc["bat_hand"].dropna().unique()[0] if "bat_hand" in fc.columns and fc["bat_hand"].dropna().size > 0 else "RHB"
                        figc = pitchmap_generic(ctrl_res, handedness=handed if pd.notna(handed) else "RHB", metric="control", bg="white")
                        st.plotly_chart(figc)
            with col2:
                fa = filtered.copy()
                fa["length"] = fa["length"].astype(str).str.upper().str.strip()
                fa["line"] = fa["line"].astype(str).str.upper().str.strip()
                avg_res = results_average_from_df(fa)
                if avg_res.shape[0] == 0 or avg_res["value"].dropna().size == 0:
                    st.plotly_chart(empty_pitch_plot(bg="white"))
                else:
                    handed = fa["bat_hand"].dropna().unique()[0] if "bat_hand" in fa.columns and fa["bat_hand"].dropna().size > 0 else "RHB"
                    figa = pitchmap_generic(avg_res, handedness=handed if pd.notna(handed) else "RHB", metric="average", bg="white")
                    st.plotly_chart(figa)
            with col3:
                fs = filtered.copy()
                fs["length"] = fs["length"].astype(str).str.upper().str.strip()
                fs["line"] = fs["line"].astype(str).str.upper().str.strip()
                sr_res = results_sr_from_df(fs)
                if sr_res.shape[0] == 0 or sr_res["value"].dropna().size == 0:
                    st.plotly_chart(empty_pitch_plot(bg="white"))
                else:
                    handed = fs["bat_hand"].dropna().unique()[0] if "bat_hand" in fs.columns and fs["bat_hand"].dropna().size > 0 else "RHB"
                    figs = pitchmap_generic(sr_res, handedness=handed if pd.notna(handed) else "RHB", metric="sr", bg="white")
                    st.plotly_chart(figs)

        elif st.session_state["active_tab"] == "Wagon Wheels":
            col1, col2 = st.columns([1, 1])
            with col1:
                ww_boundaries = wagon_wheel_plot(filtered, plot_type="boundaries", bg="white")
                st.plotly_chart(ww_boundaries, use_container_width=True)
            with col2:
                ww_caught = wagon_wheel_plot(filtered, plot_type="caught", bg="white")
                st.plotly_chart(ww_caught, use_container_width=True)
            st.markdown('<div style="margin-top:10px; margin-bottom:15px; padding:10px; background-color:#FFF3CD; border-left:4px solid #FFA500; border-radius:4px;">'
                       '<p style="color:#856404; margin:0; font-size:14px;"><b>Note:</b> The Wagon Wheels may display erroneous plots at times — working on it, modified version will come soon.</p></div>', unsafe_allow_html=True)

        elif st.session_state["active_tab"] == "Shots Analysis":
            st.markdown("<h3>Shots Analysis</h3>", unsafe_allow_html=True)
            if filtered is None or filtered.shape[0] == 0:
                st.info("No data available for the selected batter / filters.")
            else:
                shots_table = compute_shots_table(filtered.copy(), baseline_filtered)
                if shots_table is None or shots_table.shape[0] == 0:
                    shot_col_guess = _find_shot_column(filtered)
                    if shot_col_guess is None:
                        st.warning("No shot-type column could be detected in the dataset. Ensure your data contains a column with shot names (e.g., 'shot', 'shot_type', 'stroke').")
                    else:
                        st.info("No valid shot rows found after ignoring blank/NA shot entries.")
                else:
                    st.markdown("<div style='margin-bottom:6px; font-size:13px;'>"
                                "<b>SVA</b>: Shot Value Added is the excess runs over expected runs for that shot.<br>"
                                "<b>RANC</b>: Risk-Adjusted Net Contribution the expected increase in team's runs per ball, attempted from a specific shot type, after eliminating the expected run-equivalent cost incurred whenever that shot results in the batter’s dismissal.<br>"
                                "<b>A higher SVA denotes more runs above xRuns (Expected runs) while a higher RANC denotes better risk-adjusted contribution by the batter.</div>", unsafe_allow_html=True)

                    # show shots_table sortable and color SVA/RANC
                    def _color_signed(val):
                        try:
                            if val is None:
                                return ""
                            s = str(val).strip()
                            if s in ["N/A", "-"]:
                                return ""
                            if s.startswith("+"):
                                return "color: green;"
                            if s.startswith("-"):
                                return "color: red;"
                            s2 = s.replace("%","").replace("+","")
                            v = float(s2)
                            if v>0:
                                return "color: green;"
                            if v<0:
                                return "color: red;"
                        except Exception:
                            pass
                        return ""

                    color_cols = [c for c in ["SVA","RANC"] if c in shots_table.columns]
                    try:
                        sty = shots_table.style.hide_index()
                        if len(color_cols)>0:
                            sty = sty.applymap(lambda v: _color_signed(v), subset=color_cols)
                        st.dataframe(sty, use_container_width=True)
                    except Exception:
                        st.dataframe(shots_table.reset_index(drop=True), use_container_width=True)

                    csv = shots_table.to_csv(index=False)
                    st.download_button("Download Shots Table (CSV)", csv, file_name=f"{selected_batter}_shots_analysis.csv", mime="text/csv")

                    # --------------------------------------------
                    # Explanation for Risk–Reward Frontier
                    # --------------------------------------------
                    st.markdown(
                        """
                        <div style='margin-bottom:10px; font-size:13px;'>
                        <b>Risk–Reward Frontier</b><br/>
                        This chart visualises how each shot type balances <i>reward</i> (runs per ball or SVA) 
                        against <i>risk</i> (dismissal probability or low control).<br/><br/>
                        • <b>High reward, low risk</b> → optimal shots worth playing more.<br/>
                        • <b>High reward, high risk</b> → power options; useful situationally.<br/>
                        • <b>Low reward, low risk</b> → stable, strike-rotation shots.<br/>
                        • <b>Low reward, high risk</b> → negative value shots often hurting innings flow.<br/><br/>
                        This plot helps identify which shots contribute positively and which ones need strategic adjustment or reduced usage.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Risk-Reward Frontier plot under Shots table
                    try:
                        rr = shots_table.copy()
                        # prepare numeric fields: SVA per 100 balls (remove + sign), RANC numeric
                        def _to_num_sva(x):
                            if isinstance(x, str):
                                s = x.replace("+","")
                                try:
                                    return float(s)
                                except:
                                    return np.nan
                            return float(x)
                        rr["SVA_num"] = rr["SVA"].apply(_to_num_sva)
                        rr["RANC_num"] = rr["RANC"].apply(lambda x: float(str(x).replace("+","")) if (not pd.isna(x) and str(x)!="N/A") else np.nan)
                        rr["Freq"] = rr["Balls"].astype(float)
                        rr["ControlNum"] = rr["Control %"].apply(lambda x: float(str(x)) if (x!="N/A" and not pd.isna(x)) else np.nan)

                        fig = px.scatter(rr, x="SVA_num", y="RANC_num", size="Freq", color="ControlNum",
                                         hover_data=["Shot Type","Balls","Runs","xRuns","Strike Rate","Control %"],
                                         labels={"SVA_num":"SVA (runs)", "RANC_num":"RANC"})
                        median_sva = rr["SVA_num"].median()
                        median_ranc = rr["RANC_num"].median()
                        fig.add_shape(type="line", x0=median_sva, x1=median_sva, y0=rr["RANC_num"].min(), y1=rr["RANC_num"].max(), line=dict(dash="dash"))
                        fig.add_shape(type="line", x0=rr["SVA_num"].min(), x1=rr["SVA_num"].max(), y0=median_ranc, y1=median_ranc, line=dict(dash="dash"))
                        fig.update_layout(title="Risk-Reward Frontier: Shots (SVA vs RANC)", xaxis_title="SVA (runs)", yaxis_title="RANC")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

        elif st.session_state["active_tab"] == "Impact":
            st.markdown("<h3>Impact Analysis</h3>", unsafe_allow_html=True)
            if filtered is None or filtered.shape[0] == 0:
                st.info("No data available for the selected batter / filters to compute Impact.")
            else:
                impact_rows, impact_table, styled = compute_impact_table(filtered.copy(), df_all)
                if impact_table is None or impact_table.shape[0] == 0:
                    st.info("No Impact rows or no data to show for Impact analysis.")
                else:
                    st.markdown("<p style='margin-bottom:6px;'>The table includes all deliveries from the same matches as the selected batter and filters.</p>", unsafe_allow_html=True)
                    def _color_signed_imp(val):
                        try:
                            if val is None:
                                return ""
                            s = str(val).strip()
                            if s in ["N/A","-"]:
                                return ""
                            if s.startswith("+"):
                                return "color: green;"
                            if s.startswith("-"):
                                return "color: red;"
                            s2 = s.replace("%","").replace("+","")
                            v = float(s2)
                            if v>0:
                                return "color: green;"
                            if v<0:
                                return "color: red;"
                        except Exception:
                            pass
                        return ""
                    imp_color_cols = [c for c in ["Boundary Impact","SR Impact","Control Impact"] if c in impact_table.columns]
                    try:
                        if styled is not None:
                            st.dataframe(styled, use_container_width=True)
                        else:
                            sty_imp = impact_table.style.hide_index()
                            if len(imp_color_cols)>0:
                                sty_imp = sty_imp.applymap(lambda v: _color_signed_imp(v), subset=imp_color_cols)
                            st.dataframe(sty_imp, use_container_width=True)
                    except Exception:
                        st.dataframe(impact_table.reset_index(drop=True), use_container_width=True)
                    csv = impact_table.to_csv(index=False)
                    st.download_button("Download Impact Table (CSV)", csv, file_name=f"{selected_batter}_impact_table.csv", mime="text/csv")
                    try:
                        num_matches = impact_rows["p_match"].nunique()
                        num_rows = impact_rows.shape[0]
                        st.markdown(f"<div style='margin-top:8px; font-size:13px; color:{get_bg_colors('dark')['note_col']};'><b>Impact rows:</b> {num_rows} deliveries across {num_matches} matches.</div>", unsafe_allow_html=True)
                    except Exception:
                        pass

        elif st.session_state["active_tab"] == "Inns Progression":
            st.markdown(
                """
                <div style='margin-bottom:10px; font-size:13px;'>
                <b>Inns Progression</b><br/>
                • <b>Rolling Strike Rate</b>: Strike Rate computed over a recent window of balls. Rolling SR denotes recent scoring intensity.<br/>
                • <b>Rolling boundaries & dot counts</b>: Boundaries hit and dot balls consumed in the last N balls. Small windows show immediate momentum; larger windows show sustained form.<br/>
                Adjust the window on the left sidepanel to trade off responsiveness (small window) vs smoothness (large window). Innings progression plots in this section are better visualized and produce better inferential interpretations when observed on filtered datasets, instead of career graphics.
                </div>
                """,
               unsafe_allow_html=True
            )

            if filtered is None or filtered.shape[0] == 0:
                st.info("No data available for the selected batter / filters.")
            else:
                # three plots: rolling SR, cumulative runs, boundaries/dots
                colp1, colp2 = st.columns([1,1])
                with colp1:
                    fig_sr = plot_strike_rate_progression(filtered.copy(), window=sr_window)
                    st.plotly_chart(fig_sr, use_container_width=True)
                with colp2:
                    fig_cum = plot_cumulative_runs(filtered.copy())
                    st.plotly_chart(fig_cum, use_container_width=True)

                fig_bd = plot_boundaries_and_dots(filtered.copy(), window=bd_window)
                st.plotly_chart(fig_bd, use_container_width=True)

                # approximate win probability for 2nd innings
                fig_wp = approx_win_probability_second_innings(filtered.copy())
                st.plotly_chart(fig_wp, use_container_width=True)

        elif st.session_state["active_tab"] == "Other metrics":
            st.markdown("<h3>Other Metrics: xPerformance via Consistency, Reliability metrics</h3>", unsafe_allow_html=True)
            if filtered is None or filtered.shape[0] == 0:
                st.info("No data available for the selected batter / filters.")
            else:
                st.markdown(
                    """
                    <div style='margin-bottom:10px; font-size:13px;'>
                    <b>Other metrics</b><br/>
                    • <b>μ (runs/ball)</b>: Runs per ball = Total runs / Balls (Reward per delivery).<br/>
                    • <b>SD (runs/ball)</b>: Standard Deviation of runs per delivery (Volatility of reward).<br/>
                    • <b>CV</b>: Coefficient of Variation = SD / μ (Measures relative variability).<br/>
                    • <b>Downside Dev</b>: RMS of negative deviations from the baseline runs-per-ball (measures downside risk). This signifies how much the batter underperforms relative to baseline — useful to measure negative surprises (risk).<br/>
                    • <b>Dismissal rate (p)</b>: outs / balls<br/>
                    • <b>Survival rate</b>: Notion of primary risk (1 − Dismissal rate).<br/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
                shot_adv = compute_advanced_shot_metrics(filtered.copy(), baseline_filtered)
                if shot_adv is None or shot_adv.shape[0] == 0:
                    st.info("No shot-level data available to compute other metrics.")
                else:
                    try:
                        st.dataframe(shot_adv, use_container_width=True)
                    except Exception:
                        st.markdown(f"<div class='svtable-wrapper'>{shot_adv.to_html(index=False)}</div>", unsafe_allow_html=True)
                    csv1 = shot_adv.to_csv(index=False)
                    st.download_button("Download Shot-level Other Metrics (CSV)", csv1, file_name=f"{selected_batter}_advanced_shots.csv", mime="text/csv")

                st.markdown("<hr><p> </p>", unsafe_allow_html=True)
                zone_adv = compute_advanced_zone_metrics(filtered.copy())
                if zone_adv is None or zone_adv.shape[0] == 0:
                    st.info("No line/length data available to compute other metrics.")
                else:
                    try:
                        st.dataframe(zone_adv, width=900)
                    except Exception:
                        st.markdown(f"<div class='svtable-wrapper'>{zone_adv.to_html(index=False)}</div>", unsafe_allow_html=True)
                    csv2 = zone_adv.to_csv(index=False)
                    st.download_button("Download Zone Other Metrics (CSV)", csv2, file_name=f"{selected_batter}_advanced_zones.csv", mime="text/csv")

    # About the data
    st.markdown(f'<div class="custom-note" style="color:{note_col};"><b>About the data:</b><br/>The original dataset used for the study, compiled by Himanish Ganjoo, consists of ball by ball data for 20,91,210 deliveries bowled in T20 cricket from January 01, 2015 to October 18, 2025, involving 7646 batters and 5662 bowlers in 9171 matches. Explicit data on lines and lengths are available for not all, but a significant subset of the deliveries in the dataset, depending on which the pitchmaps have been plotted.<br/><br/><i>Data last updated on: October 18, 2025</i><br/><i>Software last updated on: December 12, 2025</i></div>', unsafe_allow_html=True)

    if st.session_state["run_pressed"]:
        try:
            rc = results_control_from_df(filtered.copy())
            ra = results_average_from_df(filtered.copy())
            rs = results_sr_from_df(filtered.copy())
            empty_control = (rc.shape[0] == 0) or (rc["value"].dropna().size == 0) or (rc["value"].dropna().eq(0).all())
            empty_avg = (ra.shape[0] == 0) or (ra["value"].dropna().size == 0)
            empty_sr = (rs.shape[0] == 0) or (rs["value"].dropna().size == 0)
            if empty_control and empty_avg and empty_sr:
                st.markdown("<b><span style='color:red; font-size:18px;'>Sorry, no ball by ball data is available for your query!</span></b>", unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)



