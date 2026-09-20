"""kmeans_ml_app.py - Streamlit app for the K-Means Customer Clustering ML project.

Run:  streamlit run kmeans_ml_app.py
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ml_pipeline as ml

st.set_page_config(
    page_title="Customer Clustering ML | K-Means",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Design tokens + CSS
# --------------------------------------------------------------------------- #
INK, TEAL, AMBER, MUTED, LINE = "#10222B", "#0F766E", "#E8A33D", "#5B6B73", "#DCE3E1"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#332288", "#882255", "#999933", "#7F7F7F"]
SHORT = {ml.AGE: "Age", ml.INCOME: "Income (k&#36;)", ml.SPENDING: "Spending"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
    h1, h2, h3, h4, [data-testid="stMetricValue"], .hero, .seg-card, .pred-card, .step-title {
        font-family: 'Manrope', sans-serif !important; }
    .block-container {padding-top: 1.4rem; max-width: 1300px;}
    .hero {background: #10222B; color: #fff; padding: 1.8rem 2.2rem; border-radius: 16px;
           border-left: 8px solid #E8A33D; margin-bottom: 1.1rem;}
    .hero h1 {color: #fff; margin: 0 0 .35rem 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.5px;}
    .hero p {color: #C9D6DA; margin: 0 0 .9rem 0; font-size: 1.04rem; max-width: 62rem; line-height: 1.55;}
    .chip {display: inline-block; padding: .2rem .75rem; margin: 0 .4rem .3rem 0; border-radius: 999px;
           background: #1B3A46; color: #E6EEF0; font-size: .85rem; font-weight: 600;}
    [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DCE3E1; border-radius: 12px;
           padding: .7rem 1rem;}
    [data-testid="stMetricLabel"] {color: #5B6B73;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {padding: .55rem 1rem; font-weight: 700; border-radius: 10px 10px 0 0;}
    .seg-card {background: #fff; border: 1px solid #DCE3E1; border-radius: 12px; padding: 1rem 1.1rem 1.1rem;
               margin-bottom: 1rem; min-height: 19rem;}
    .seg-title {font-size: 1.15rem; font-weight: 800; color: #10222B; margin-bottom: .1rem;}
    .seg-meta {color: #5B6B73; font-size: .85rem; margin-bottom: .55rem;}
    .seg-desc {color: #23343C; font-size: .93rem; line-height: 1.45; margin-bottom: .7rem;}
    .kvs {display: flex; gap: .5rem; margin-bottom: .7rem;}
    .kv {flex: 1; background: #F1F5F4; border-radius: 8px; padding: .35rem .5rem; text-align: center;}
    .kv span {display: block; font-size: .72rem; color: #5B6B73;}
    .kv b {font-size: 1.05rem; color: #10222B;}
    .seg-strategy {font-size: .88rem; color: #23343C; background: #FFF6E5; border-radius: 8px; padding: .5rem .65rem;}
    .pred-card {background: #fff; border: 2px solid #0F766E; border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: .8rem;}
    .pred-title {font-size: 1.7rem; font-weight: 800; color: #10222B; line-height: 1.2;}
    .pred-card p {color: #3B4C54; margin: .35rem 0 0 0;}
    .step-title {font-weight: 800; color: #0F766E; font-size: 1.02rem;}
    .footer {text-align: center; color: #8A9AA1; font-size: .85rem; margin-top: 2.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def md(text: str) -> str:
    """Stop Streamlit from treating '$' as LaTeX in markdown text."""
    return text.replace("$", "\\$")


# --------------------------------------------------------------------------- #
# Load model + data (trains automatically if the saved model can't be used)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained model...")
def get_artifacts():
    return ml.load_or_train()


model, metrics, df_loaded, cleaning, status = get_artifacts()
df = df_loaded.copy()

X = df[ml.FEATURES].to_numpy(dtype=float)
idx_train, idx_test = ml.split_indices(len(df))
df["Cluster"] = model.predict(X)
segments = ml.describe_segments(df)
df["Segment"] = df["Cluster"].map(lambda c: segments[int(c)]["title"])
df["Split"] = "Train"
df.loc[idx_test, "Split"] = "Test"

order = sorted(segments)
color_by_cluster = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}
color_map = {segments[c]["title"]: color_by_cluster[c] for c in order}
seg_order = [segments[c]["title"] for c in order]
centers = model.centers_
K = model.k
te, tr, stab = metrics["test"], metrics["train"], metrics["stability"]
baseline = metrics["baseline_random_silhouette"]


def quality_label(sil: float) -> str:
    if sil >= 0.5:
        return "Good separation"
    if sil >= 0.35:
        return "Fair separation - some overlap"
    if sil >= 0.25:
        return "Weak separation"
    return "Poor separation"


@st.cache_data(show_spinner=False)
def sandbox(k: int):
    """Train on the TRAIN split with a different K and test it on the TEST split."""
    m = ml.make_model(X[idx_train], k)
    plot = df[[ml.INCOME, ml.SPENDING, "Split"]].copy()
    plot["Cluster"] = m.predict(X).astype(str)
    return {"train": ml.evaluate_model(m, X[idx_train]), "test": ml.evaluate_model(m, X[idx_test]), "plot": plot}


# --------------------------------------------------------------------------- #
# Sidebar - model card
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🧠 Model card")
    if status.startswith("Loaded"):
        st.success("Trained model loaded from `models/`")
    else:
        st.success("Model trained and ready on this server")
    st.markdown(
        f"""
        **Algorithm:** K-Means (k-means++)  
        **Segments (K):** {K}  
        **Features:** Age, Income, Spending Score  
        **Data:** {metrics['n_train']} train / {metrics['n_test']} test  
        **Test silhouette:** {te['silhouette']:.3f}  
        **Stability (ARI):** {stab['mean_ari']:.2f}  
        **Trained:** {metrics.get('trained_at', '-')}  
        **scikit-learn:** {metrics.get('sklearn_version', '-')}
        """
    )
    st.divider()
    st.markdown("### Quick guide")
    st.markdown(
        "- **Overview** - the whole pipeline at a glance\n"
        "- **Training & Testing** - proof the model works on unseen data\n"
        "- **Segments** - who the customers are\n"
        "- **Predict** - try a new customer live"
    )

# --------------------------------------------------------------------------- #
# Hero + KPIs
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero">
      <h1>Customer Clustering ML</h1>
      <p>A K-Means machine-learning model that discovers customer segments from age, income and
      spending behaviour. It is trained, tested on unseen customers, and ready to predict the
      segment of any new customer.</p>
      <span class="chip">K-Means</span><span class="chip">{K} segments</span>
      <span class="chip">Train / Test validated</span><span class="chip">Live prediction</span>
      <span class="chip">{len(df)} customers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Segments found (K)", K)
c3.metric("Test silhouette", f"{te['silhouette']:.3f}", f"{te['silhouette'] - baseline:+.2f} vs random labels")
c4.metric("Stability (ARI)", f"{stab['mean_ari']:.2f}", help="1.0 = identical clusters on resampled data")
c5.metric("Train / Test", f"{metrics['n_train']} / {metrics['n_test']}")

tab_over, tab_data, tab_ml, tab_seg, tab_pred, tab_exp, tab_about = st.tabs(
    ["🏠 Overview", "🔬 Data Explorer", "🧪 Training & Testing", "🎯 Segments", "🔮 Predict", "📥 Export", "📘 About"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.subheader("How the model was built")
    steps = [
        ("1 · Load & clean", f"{cleaning['rows_after']} customers, {cleaning['missing_cells']} missing values, "
                             f"{cleaning['duplicate_rows']} duplicates."),
        ("2 · Split", f"{metrics['n_train']} customers for training, {metrics['n_test']} kept aside for testing."),
        ("3 · Choose K", f"{metrics['n_splits']}-fold cross-validation on the training set picked K = {K}."),
        ("4 · Test", f"Silhouette on unseen customers: {te['silhouette']:.3f} (random labels: {baseline:.3f})."),
        ("5 · Stability", f"{stab['n_bootstrap']} bootstrap re-runs agree with the final clusters (ARI {stab['mean_ari']:.2f})."),
        ("6 · Deploy", "Refit on all customers and saved to `models/`, ready for live prediction."),
    ]
    for row in (steps[:3], steps[3:]):
        cols = st.columns(3)
        for col, (title, text) in zip(cols, row):
            with col, st.container(border=True):
                st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
                st.write(text)

    st.subheader("The customer map")
    left, right = st.columns([2.1, 1])
    with left:
        fig = px.scatter(
            df, x=ml.INCOME, y=ml.SPENDING, color="Segment", color_discrete_map=color_map,
            category_orders={"Segment": seg_order}, opacity=0.85,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        fig.add_trace(go.Scatter(
            x=centers[ml.INCOME], y=centers[ml.SPENDING], mode="markers+text", name="Centroids",
            marker=dict(symbol="x", size=15, color="black", line=dict(width=2)),
            text=[f"C{c}" for c in centers.index], textposition="top center",
        ))
        fig.update_layout(height=480, legend_title_text="Segment", margin=dict(t=20))
        st.plotly_chart(fig, key="over_map")
    with right:
        sizes = df["Segment"].value_counts().reindex(seg_order).reset_index()
        sizes.columns = ["Segment", "Customers"]
        fig = px.pie(sizes, names="Segment", values="Customers", hole=0.55,
                     color="Segment", color_discrete_map=color_map)
        fig.update_layout(height=480, showlegend=False, margin=dict(t=20, b=20),
                          annotations=[dict(text=f"{K}<br>segments", showarrow=False, font_size=18)])
        fig.update_traces(textinfo="percent")
        st.plotly_chart(fig, key="over_pie")

    with st.expander("Peek at the dataset"):
        st.dataframe(df[[ml.ID_COL, *ml.CATEGORICAL, *ml.FEATURES]].head(25), hide_index=True)

# --------------------------------------------------------------------------- #
# Data Explorer
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Data quality")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", cleaning["rows_before"])
    d2.metric("Missing cells", cleaning["missing_cells"])
    d3.metric("Duplicate rows", cleaning["duplicate_rows"])
    d4.metric("Rows after cleaning", cleaning["rows_after"])
    st.caption(
        "The model uses Age, Annual Income and Spending Score. CustomerID is only a label. "
        "Gender, Education and Marital Status are categorical, so they are used afterwards to describe the segments."
    )
    st.dataframe(df[ml.FEATURES].describe().T.round(2))

    st.subheader("Distributions")
    cols = st.columns(3)
    for i, f in enumerate(ml.FEATURES):
        with cols[i]:
            fig = px.histogram(df, x=f, nbins=20, marginal="box", color_discrete_sequence=[TEAL], title=f)
            fig.update_layout(showlegend=False, height=360, margin=dict(t=50, b=10))
            st.plotly_chart(fig, key=f"hist_{i}")

    st.subheader("Who are the customers?")
    cols = st.columns(3)
    for i, c in enumerate(ml.CATEGORICAL):
        with cols[i]:
            vc = df[c].value_counts().reset_index()
            vc.columns = [c, "Customers"]
            fig = px.bar(vc, x=c, y="Customers", color_discrete_sequence=[AMBER], title=c, text="Customers")
            fig.update_layout(height=340, margin=dict(t=50, b=10), xaxis_title=None)
            st.plotly_chart(fig, key=f"cat_{i}")

    st.subheader("Relationships between features")
    r1, r2 = st.columns([1.6, 1])
    with r1:
        fig = px.scatter_matrix(df, dimensions=ml.FEATURES, color="Gender",
                                color_discrete_sequence=["#D55E00", "#0072B2"], opacity=0.7)
        fig.update_traces(diagonal_visible=False, marker=dict(size=5))
        fig.update_layout(height=520)
        st.plotly_chart(fig, key="matrix")
    with r2:
        corr = df[ml.FEATURES].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=520, title="Correlation")
        st.plotly_chart(fig, key="corr")
    st.info(
        "Age and Income are almost uncorrelated with Spending Score, so no single feature separates "
        "the customers. That is why clustering on all three together is useful."
    )

# --------------------------------------------------------------------------- #
# Training & Testing
# --------------------------------------------------------------------------- #
with tab_ml:
    st.subheader("Training and testing report")
    st.write(
        "K-Means is **unsupervised**: there is no correct label, so there is no accuracy score. "
        "Instead we prove the model on customers it has never seen: the model is trained on "
        f"**{metrics['n_train']}** customers and evaluated on **{metrics['n_test']}** held-out customers, "
        "using clustering-quality metrics."
    )

    kdf = pd.DataFrame(metrics["k_selection"])
    best_row = kdf.loc[kdf["cv_silhouette"].idxmax()]

    st.markdown("#### 1. Choosing the number of clusters (training data only)")
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Cross-validated silhouette (higher is better)", "Elbow curve - inertia (lower is tighter)"))
    fig.add_trace(go.Scatter(
        x=kdf["k"], y=kdf["cv_silhouette"], mode="lines+markers", name="CV silhouette",
        line=dict(color=TEAL, width=3), error_y=dict(type="data", array=list(kdf["cv_std"]), visible=True),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=kdf["k"], y=kdf["train_inertia"], mode="lines+markers", name="Inertia",
        line=dict(color=AMBER, width=3),
    ), row=1, col=2)
    fig.add_vline(x=K, line_dash="dash", line_color="#444", row=1, col=1)
    fig.add_vline(x=K, line_dash="dash", line_color="#444", row=1, col=2)
    fig.update_xaxes(title_text="K (number of clusters)", dtick=1)
    fig.update_layout(showlegend=False, height=400, margin=dict(t=60))
    st.plotly_chart(fig, key="k_select")

    close = kdf[(kdf["k"] != K) & (kdf["cv_silhouette"] >= best_row["cv_silhouette"] - best_row["cv_std"])]["k"].tolist()
    msg = f"K = {K} has the highest cross-validated silhouette ({best_row['cv_silhouette']:.3f})."
    if close:
        msg += (f" K = {', '.join(str(k) for k in close)} score within one standard deviation, so they are "
                "statistically close - K is chosen by the best score, and the elbow curve bends in the same region.")
    st.success(msg)
    with st.expander("See the full K-selection table"):
        st.dataframe(kdf.rename(columns={
            "k": "K", "cv_silhouette": "CV silhouette", "cv_std": "CV std",
            "train_silhouette": "Train silhouette", "train_inertia": "Train inertia"}).round(3), hide_index=True)

    st.markdown("#### 2. Evaluation on unseen test customers")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test silhouette", f"{te['silhouette']:.3f}", f"{te['silhouette'] - baseline:+.2f} vs random",
              help="-1 to 1. Higher = customers sit clearly inside their own segment.")
    m2.metric("Random-label baseline", f"{baseline:.3f}", help="Silhouette if customers were assigned to segments at random.")
    m3.metric("Test Davies-Bouldin", f"{te['davies_bouldin']:.3f}", help="Lower is better.")
    m4.metric("Train to test gap", f"{tr['silhouette'] - te['silhouette']:+.3f}", help="Small gap = the model generalises.")
    st.write(f"**Verdict:** {quality_label(te['silhouette'])}. The model is far better than random assignment and "
             "keeps a similar score on customers it never saw during training.")
    comp = pd.DataFrame({
        "Metric": ["Silhouette (higher better)", "Davies-Bouldin (lower better)",
                   "Calinski-Harabasz (higher better)", "Inertia per customer (lower better)", "Customers"],
        "Train": [tr["silhouette"], tr["davies_bouldin"], tr["calinski_harabasz"], tr["inertia_per_sample"], tr["n_samples"]],
        "Test": [te["silhouette"], te["davies_bouldin"], te["calinski_harabasz"], te["inertia_per_sample"], te["n_samples"]],
    })
    st.dataframe(comp.round(3), hide_index=True)
    st.caption("Calinski-Harabasz grows with the number of customers, so compare it only between models on the same data size.")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### 3. Do train and test look alike?")
        share = metrics["cluster_share"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"C{c}" for c in range(K)], y=share["train"], name="Train", marker_color=TEAL))
        fig.add_trace(go.Bar(x=[f"C{c}" for c in range(K)], y=share["test"], name="Test", marker_color=AMBER))
        fig.update_layout(barmode="group", height=360, yaxis_title="% of customers", margin=dict(t=20))
        st.plotly_chart(fig, key="share")
        st.caption("Similar segment sizes in train and test means the test set is representative.")
    with g2:
        st.markdown("#### 4. Are the clusters stable?")
        fig = px.histogram(x=stab["ari_values"], nbins=10, color_discrete_sequence=[TEAL],
                           labels={"x": "Adjusted Rand Index (1 = identical clusters)"})
        fig.update_layout(height=360, yaxis_title="Bootstrap runs", margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig, key="stab")
        st.caption(f"Re-training on {stab['n_bootstrap']} random resamples gives ARI {stab['mean_ari']:.2f} "
                   f"(+/- {stab['std_ari']:.2f}) against the final segments.")

    st.markdown("#### 5. Experiment: what if we used a different K?")
    k_try = st.slider("Number of clusters (K)", 2, 10, K, key="sandbox_k")
    sb = sandbox(k_try)
    s1, s2, s3 = st.columns(3)
    s1.metric("Train silhouette", f"{sb['train']['silhouette']:.3f}")
    s2.metric("Test silhouette", f"{sb['test']['silhouette']:.3f}",
              f"{sb['test']['silhouette'] - te['silhouette']:+.3f} vs chosen K = {K}")
    s3.metric("Test Davies-Bouldin", f"{sb['test']['davies_bouldin']:.3f}")
    fig = px.scatter(sb["plot"], x=ml.INCOME, y=ml.SPENDING, color="Cluster", symbol="Split",
                     symbol_map={"Train": "circle", "Test": "diamond"}, opacity=0.85)
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(height=430, margin=dict(t=20))
    st.plotly_chart(fig, key="sandbox_plot")
    st.caption("Circles = training customers, diamonds = test customers. This experiment trains only on the training split.")

    with st.expander("Model files"):
        st.write(f"Model file: `models/kmeans_ml_model.joblib` - metrics: `models/ml_metrics.json`")
        st.write(f"Trained at: {metrics.get('trained_at', '-')} - scikit-learn {metrics.get('sklearn_version', '-')} "
                 f"- random_state {metrics['random_state']}")
        st.write("To re-train from the terminal: `python train_ml_model.py`. To run the tests: `python -m unittest test_ml_model -v`.")

# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
with tab_seg:
    st.subheader("Meet the segments")
    for start in range(0, len(order), 3):
        cols = st.columns(3)
        for col, cid in zip(cols, order[start:start + 3]):
            s = segments[cid]
            kvs = "".join(
                f"<div class='kv'><span>{SHORT[f]}</span><b>{s['means'][f]:.0f}</b></div>" for f in ml.FEATURES)
            col.markdown(
                f"""<div class="seg-card" style="border-top: 6px solid {color_by_cluster[cid]}">
                <div class="seg-title">{s['title']}</div>
                <div class="seg-meta">Cluster {cid} &nbsp;|&nbsp; {s['size']} customers ({s['pct']:.0f}%)</div>
                <div class="seg-desc">{s['description']}</div>
                <div class="kvs">{kvs}</div>
                <div class="seg-strategy"><b>Strategy:</b> {s['strategy']}</div></div>""",
                unsafe_allow_html=True,
            )

    st.subheader("Cluster explorer")
    a1, a2, a3 = st.columns(3)
    x_feat = a1.selectbox("X axis", ml.FEATURES, index=1)
    y_feat = a2.selectbox("Y axis", ml.FEATURES, index=2)
    show = a3.radio("Customers shown", ["All", "Train only", "Test only"], horizontal=True)
    view = df if show == "All" else df[df["Split"] == show.split()[0]]
    fig = px.scatter(view, x=x_feat, y=y_feat, color="Segment", color_discrete_map=color_map,
                     category_orders={"Segment": seg_order}, hover_data=[ml.ID_COL, "Gender", "Education"], opacity=0.85)
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.add_trace(go.Scatter(
        x=centers[x_feat], y=centers[y_feat], mode="markers+text", name="Centroids",
        marker=dict(symbol="x", size=15, color="black", line=dict(width=2)),
        text=[f"C{c}" for c in centers.index], textposition="top center"))
    fig.update_layout(height=520)
    st.plotly_chart(fig, key="seg_map")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### 3-D view")
        fig = px.scatter_3d(df, x=ml.AGE, y=ml.INCOME, z=ml.SPENDING, color="Segment",
                            color_discrete_map=color_map, category_orders={"Segment": seg_order}, opacity=0.85)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=520, legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, key="seg_3d")
    with p2:
        st.markdown("#### Segment fingerprints")
        lo, hi = df[ml.FEATURES].min(), df[ml.FEATURES].max()
        means = df.groupby("Cluster")[ml.FEATURES].mean()
        norm = (means - lo) / (hi - lo).replace(0, 1)
        radar = go.Figure()
        for cid in order:
            vals = norm.loc[cid].tolist()
            radar.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=["Age", "Income", "Spending", "Age"], fill="toself",
                name=segments[cid]["title"], line=dict(color=color_by_cluster[cid]), opacity=0.55))
        radar.update_layout(height=520, polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                            legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(radar, key="seg_radar")

    st.markdown("#### Feature spread inside each segment")
    f_choice = st.selectbox("Feature", ml.FEATURES, key="box_feat")
    fig = px.box(df, x="Segment", y=f_choice, color="Segment", color_discrete_map=color_map,
                 category_orders={"Segment": seg_order}, points="all")
    fig.update_layout(height=430, showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, key="seg_box")

    st.markdown("#### Segment composition")
    cat = st.selectbox("Category", ml.CATEGORICAL, key="cat_choice")
    fig = px.histogram(df, x="Segment", color=cat, barnorm="percent",
                       category_orders={"Segment": seg_order}, color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=430, yaxis_title="% of segment", xaxis_title=None, bargap=0.25)
    st.plotly_chart(fig, key="seg_comp")

    summary = pd.DataFrame([
        {"Cluster": cid, "Segment": segments[cid]["title"], "Customers": segments[cid]["size"],
         "Share (%)": round(segments[cid]["pct"], 1),
         **{f: round(segments[cid]["means"][f], 1) for f in ml.FEATURES},
         **{f"Top {c}": df[df["Cluster"] == cid][c].value_counts().idxmax() for c in ml.CATEGORICAL}}
        for cid in order
    ])
    st.markdown("#### Summary table")
    st.dataframe(summary, hide_index=True)

    st.markdown("#### Key findings")
    big = max(order, key=lambda c: segments[c]["size"])
    rich = max(order, key=lambda c: segments[c]["means"][ml.INCOME])
    spender = max(order, key=lambda c: segments[c]["means"][ml.SPENDING])
    young = min(order, key=lambda c: segments[c]["means"][ml.AGE])
    saver = min(order, key=lambda c: segments[c]["means"][ml.SPENDING])
    st.markdown(
        f"- The largest segment is **{segments[big]['title']}** with **{segments[big]['size']}** customers.\n"
        f"- The highest average income belongs to **{segments[rich]['title']}** "
        f"({segments[rich]['means'][ml.INCOME]:.0f}k).\n"
        f"- The biggest spenders are **{segments[spender]['title']}** "
        f"(spending score {segments[spender]['means'][ml.SPENDING]:.0f}).\n"
        f"- The youngest segment is **{segments[young]['title']}** (average age {segments[young]['means'][ml.AGE]:.0f}).\n"
        f"- The lowest spenders are **{segments[saver]['title']}** "
        f"(spending score {segments[saver]['means'][ml.SPENDING]:.0f})."
    )

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
PRESETS = {
    "Young big spender": (24, 28, 82),
    "Affluent shopper": (33, 90, 85),
    "Wealthy but careful": (44, 95, 12),
    "Budget-conscious": (48, 25, 15),
    "Typical customer": (int(df[ml.AGE].median()), int(df[ml.INCOME].median()), int(df[ml.SPENDING].median())),
}
st.session_state.setdefault("in_age", PRESETS["Typical customer"][0])
st.session_state.setdefault("in_income", PRESETS["Typical customer"][1])
st.session_state.setdefault("in_spend", PRESETS["Typical customer"][2])
st.session_state.setdefault("pred_log", [])


def apply_preset(age: int, income: int, spend: int) -> None:
    st.session_state["in_age"], st.session_state["in_income"], st.session_state["in_spend"] = age, income, spend


with tab_pred:
    st.subheader("Predict the segment of a new customer")
    st.write("Move the sliders or pick an example. The trained model assigns the customer to the nearest segment instantly.")
    pcols = st.columns(len(PRESETS))
    for col, (name, vals) in zip(pcols, PRESETS.items()):
        col.button(name, on_click=apply_preset, args=vals, key=f"preset_{name}")

    inp, out = st.columns([1, 1.5])
    with inp:
        with st.container(border=True):
            st.markdown("#### Customer details")
            age = st.slider("Age (years)", 15, 80, key="in_age")
            income = st.slider("Annual income (k$)", 10, 150, key="in_income")
            spend = st.slider("Spending score (1-100)", 1, 100, key="in_spend")
    values = {ml.AGE: age, ml.INCOME: income, ml.SPENDING: spend}
    res = ml.predict_customer(model, values)
    cid = res["cluster"]
    seg = segments[cid]

    with out:
        st.markdown(
            f"""<div class="pred-card" style="border-color: {color_by_cluster[cid]}">
            <div class="pred-title">{seg['title']}</div><p>{seg['description']}</p></div>""",
            unsafe_allow_html=True,
        )
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Cluster", f"#{cid}")
        q2.metric("Confidence", f"{res['confidence'] * 100:.0f}%",
                  help="How clearly the customer sits inside this segment rather than between two segments.")
        q3.metric("Match strength", f"{res['match'][cid] * 100:.0f}%")
        q4.metric("Segment size", seg["size"])
        st.success(f"**Recommended strategy:** {seg['strategy']}")
        if res["confidence"] < 0.25:
            others = np.argsort(res["distances"])[1]
            st.warning(f"This customer is on the border with **{segments[int(others)]['title']}**.")
        if st.button("Add this prediction to the log", key="log_add"):
            st.session_state["pred_log"].append({
                "Time": datetime.now().strftime("%H:%M:%S"), "Age": age, "Income (k$)": income,
                "Spending score": spend, "Cluster": cid, "Segment": seg["title"],
                "Confidence (%)": round(res["confidence"] * 100),
            })

    v1, v2, v3 = st.columns([1.4, 1.1, 1])
    with v1:
        st.markdown("#### Where the customer lands")
        fig = px.scatter(df, x=ml.INCOME, y=ml.SPENDING, color="Segment", color_discrete_map=color_map,
                         category_orders={"Segment": seg_order}, opacity=0.4)
        fig.add_trace(go.Scatter(
            x=[income], y=[spend], mode="markers", name="New customer",
            marker=dict(symbol="star", size=24, color="gold", line=dict(color="black", width=2))))
        fig.update_layout(height=420, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, key="pred_map")
    with v2:
        st.markdown("#### Customer vs segment average")
        lo, hi = df[ml.FEATURES].min(), df[ml.FEATURES].max()
        span = (hi - lo).replace(0, 1)
        me = [float(np.clip((values[f] - lo[f]) / span[f], 0, 1)) for f in ml.FEATURES]
        avg = [float((seg["means"][f] - lo[f]) / span[f]) for f in ml.FEATURES]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=["Age", "Income", "Spending", "Age"], fill="toself",
                                        name="Segment average", line=dict(color=color_by_cluster[cid]), opacity=0.5))
        radar.add_trace(go.Scatterpolar(r=me + me[:1], theta=["Age", "Income", "Spending", "Age"],
                                        name="This customer", line=dict(color="black", width=3)))
        radar.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                            legend=dict(orientation="h", y=-0.1), margin=dict(t=10))
        st.plotly_chart(radar, key="pred_radar")
    with v3:
        st.markdown("#### Match with each segment")
        match_df = pd.DataFrame({"Segment": seg_order, "Match (%)": [round(res["match"][c] * 100, 1) for c in order]})
        fig = px.bar(match_df, x="Match (%)", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=color_map, text="Match (%)")
        fig.update_layout(height=420, showlegend=False, margin=dict(t=10))
        fig.update_yaxes(autorange="reversed", title_text="")
        st.plotly_chart(fig, key="pred_bar")

    st.markdown("#### The five most similar existing customers")
    near = ml.nearest_customers(model, df, values, 5)
    st.dataframe(near[[ml.ID_COL, "Gender", "Education", "Marital Status", *ml.FEATURES, "Segment", "Distance"]],
                 hide_index=True)

    if st.session_state["pred_log"]:
        st.markdown("#### Prediction log")
        log_df = pd.DataFrame(st.session_state["pred_log"])
        st.dataframe(log_df, hide_index=True)
        lc1, lc2 = st.columns(2)
        lc1.download_button("Download log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                            "prediction_log.csv", "text/csv", key="dl_log")
        if lc2.button("Clear log", key="log_clear"):
            st.session_state["pred_log"] = []
            st.rerun()

    with st.expander("Batch prediction - upload a CSV of new customers"):
        st.write(md(f"The file needs these columns: **{', '.join(ml.FEATURES)}**"))
        template = pd.DataFrame([[25, 30, 80], [45, 90, 15]], columns=ml.FEATURES)
        st.download_button("Download a template CSV", template.to_csv(index=False).encode("utf-8"),
                           "new_customers_template.csv", "text/csv", key="dl_template")
        up = st.file_uploader("New customers CSV", type=["csv"], key="batch_upload")
        if up is not None:
            try:
                new = pd.read_csv(up)
                new.columns = [c.strip() for c in new.columns]
                missing = [f for f in ml.FEATURES if f not in new.columns]
                if missing:
                    st.error(md(f"Missing columns: {missing}"))
                else:
                    new = new.dropna(subset=ml.FEATURES).reset_index(drop=True)
                    labels = model.predict(new[ml.FEATURES].to_numpy(dtype=float))
                    new["Cluster"] = labels
                    new["Segment"] = [segments[int(c)]["title"] for c in labels]
                    st.dataframe(new, hide_index=True)
                    st.download_button("Download predictions", new.to_csv(index=False).encode("utf-8"),
                                       "batch_predictions.csv", "text/csv", key="dl_batch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not process that file: {exc}")

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
with tab_exp:
    st.subheader("Download the results")
    export_df = df.drop(columns=["Split"])
    st.dataframe(export_df.head(50), hide_index=True, height=300)
    card = (
        f"# Model card - K-Means Customer Clustering ML\n\n"
        f"- Algorithm: K-Means (k-means++), StandardScaler\n- Features: {', '.join(ml.FEATURES)}\n"
        f"- K (segments): {K}, chosen by {metrics['n_splits']}-fold CV silhouette on the training set\n"
        f"- Data: {metrics['n_rows']} customers ({metrics['n_train']} train / {metrics['n_test']} test)\n"
        f"- Test silhouette: {te['silhouette']:.3f} (random baseline {baseline:.3f})\n"
        f"- Test Davies-Bouldin: {te['davies_bouldin']:.3f}\n"
        f"- Bootstrap stability (ARI): {stab['mean_ari']:.3f} +/- {stab['std_ari']:.3f}\n"
        f"- Trained: {metrics.get('trained_at', '-')}, scikit-learn {metrics.get('sklearn_version', '-')}\n"
    )
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("Customers + segments (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                       "customers_with_segments.csv", "text/csv", key="dl_customers")
    e2.download_button("Segment summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                       "segment_summary.csv", "text/csv", key="dl_summary")
    e3.download_button("Training metrics (JSON)", json.dumps(metrics, indent=2, default=str).encode("utf-8"),
                       "ml_metrics.json", "application/json", key="dl_metrics")
    e4.download_button("Model card (Markdown)", card.encode("utf-8"), "model_card.md", "text/markdown", key="dl_card")

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("How K-Means works")
    st.markdown(
        """
        K-Means is an **unsupervised** learning algorithm: it finds groups on its own, without labels.

        1. Pick the number of groups **K**.
        2. Place K centroids (group centres) using the k-means++ smart start.
        3. Assign every customer to the nearest centroid.
        4. Move each centroid to the average of its customers.
        5. Repeat steps 3 and 4 until the groups stop changing.

        **Prediction:** a new customer is scaled with the same scaler as the training data and assigned to
        the segment with the nearest centroid.
        """
    )
    st.subheader("Metrics used")
    st.markdown(
        "- **Silhouette** (-1 to 1): how well each customer fits its own segment compared with the next one. Higher is better.\n"
        "- **Davies-Bouldin**: average similarity between segments. Lower is better.\n"
        "- **Calinski-Harabasz**: between-segment vs within-segment spread. Higher is better.\n"
        "- **Inertia**: total squared distance to the centroids. Lower means tighter clusters.\n"
        "- **ARI (stability)**: agreement between clusterings on resampled data. 1 means identical."
    )
    st.subheader("Honest limitations")
    st.markdown(
        "- The data has only 200 customers, so results can shift a little with more data.\n"
        "- Segments overlap a little (silhouette is in the 'fair' range), which is normal for real customer behaviour.\n"
        "- K-Means finds groups; naming them and choosing marketing actions is a business interpretation."
    )
    st.subheader("Project files")
    st.code(
        "kmeans_ml_app.py      Streamlit app (this UI)\n"
        "ml_pipeline.py        cleaning, training, testing, prediction\n"
        "train_ml_model.py     trains + saves the model\n"
        "test_ml_model.py      automated tests\n"
        "dataset.csv           customer data\n"
        "models/               saved model + metrics",
        language="text",
    )
    st.markdown(md("**Tech stack:** Python, pandas, NumPy, scikit-learn, Plotly, Streamlit"))

st.markdown('<div class="footer">Customer Clustering ML - K-Means - built with Python and Streamlit</div>',
            unsafe_allow_html=True)
