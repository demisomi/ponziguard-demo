"""
PonziGuard Dashboard — Tabbed Layout
Clean, focused views for each stage of the three-stage cascade.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import os
from transformers import pipeline as hf_pipeline

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PonziGuard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom styling for cleaner look ──────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    [data-testid="stMetricValue"] { font-size: 2rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] {
        padding: 12px 24px; font-size: 1rem; font-weight: 500;
        border-radius: 8px 8px 0 0;
    }
    .stAlert { border-radius: 8px; }
    h1 { font-size: 2rem; margin-bottom: 0.2rem; }
    h2 { font-size: 1.4rem; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.title("🛡️ PonziGuard")
st.caption("Three-Stage Ponzi Scheme Detection — Pilot Demo")
st.markdown("")

# ── Red-flag lexicon (kept as a set for speed) ───────────────────────
RED_FLAGS = set([
    "invest", "investment", "investor", "investing", "profit", "profits",
    "profitable", "return", "returns", "roi", "earn", "earning", "earnings",
    "income", "passive income", "money", "cash", "funds", "capital",
    "payout", "pay out", "cashout", "cash out", "interest", "dividend",
    "yield", "ponzi", "pyramid", "scheme", "referral", "refer a friend",
    "recruit", "downline", "upline", "network", "commission", "bonus",
    "reward", "incentive", "withdraw", "withdrawal", "deposit", "membership",
    "upgrade", "package", "plan", "tier", "level", "rank", "guaranteed",
    "guarantee", "risk free", "risk-free", "no risk", "double", "triple",
    "multiply", "high return", "huge return", "massive return",
    "daily profit", "weekly profit", "monthly profit", "daily income",
    "weekly income", "monthly income", "limited offer", "limited slots",
    "limited time", "act now", "join now", "register now", "sign up now",
    "don't miss", "do not miss", "hurry", "100%", "200%", "300%", "500%",
    "per day", "per week", "per month", "forex", "crypto", "bitcoin",
    "btc", "cryptocurrency", "trading", "trade", "signal", "signals",
    "stocks", "shares", "market", "loan", "lending", "borrowing",
    "transfer", "send money", "bank account", "account number", "wallet",
    "naira", "ngn", "dollar", "usd", "opay", "palmpay", "kuda", "mmm",
    "loom", "racksterli", "cbex", "join us", "join our", "become a member",
    "register", "sign up", "click here", "whatsapp", "contact us", "dm us",
    "telegram group", "invite", "link below", "exclusive", "vip", "premium",
    "opportunity", "once in a lifetime", "financial freedom",
    "financial independence", "retire", "quit your job", "work from home",
    "make money", "make cash", "extra money", "side hustle", "side income",
    "scam", "scammer", "fraud", "payment", "sell", "offer", "free",
    "unlimited", "fast",
])

def get_flags(text):
    t = str(text).lower()
    return [kw for kw in RED_FLAGS if kw in t]

# ── Data loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("telegram_data.csv")
    df["text"] = df["text"].fillna("")
    df["red_flags"] = df["text"].apply(get_flags)
    df["flag_count"] = df["red_flags"].apply(len)
    df["stage1_flagged"] = df["flag_count"] > 0
    df["red_flags_str"] = df["red_flags"].apply(lambda x: ", ".join(x) if x else "—")
    df["is_ponzi"] = df["label"] == "ponzi_related"
    return df

@st.cache_data
def load_stage2_results():
    try:
        return pd.read_csv("stage2_finetuned_results.csv"), True
    except: return None, False

@st.cache_data
def load_stage3_results():
    try:
        return pd.read_csv("stage3_alerts.csv"), True
    except: return None, False

df = load_data()
stage2_df, s2_loaded = load_stage2_results()
stage3_df, s3_loaded = load_stage3_results()

# ── Compute metrics once (used across tabs) ───────────────────────────
# Stage 1
s1_tp = int((df["stage1_flagged"] & df["is_ponzi"]).sum())
s1_fp = int((df["stage1_flagged"] & ~df["is_ponzi"]).sum())
s1_fn = int((~df["stage1_flagged"] & df["is_ponzi"]).sum())
s1_tn = int((~df["stage1_flagged"] & ~df["is_ponzi"]).sum())
s1_prec   = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0
s1_recall = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0
s1_f1     = 2 * s1_prec * s1_recall / (s1_prec + s1_recall) if (s1_prec + s1_recall) > 0 else 0

# Stage 2
s2_prec = s2_recall = s2_f1 = s2_acc = 0
s2_tp = s2_fp = s2_fn = s2_tn = 0
if s2_loaded:
    s2 = stage2_df.copy()
    s2["is_ponzi"] = s2["label"] == "ponzi_related"
    s2["pred_ponzi"] = s2["stage2_pred_label"] == "ponzi_related"
    s2_tp = int((s2["pred_ponzi"] & s2["is_ponzi"]).sum())
    s2_fp = int((s2["pred_ponzi"] & ~s2["is_ponzi"]).sum())
    s2_fn = int((~s2["pred_ponzi"] & s2["is_ponzi"]).sum())
    s2_tn = int((~s2["pred_ponzi"] & ~s2["is_ponzi"]).sum())
    s2_prec   = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0
    s2_recall = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0
    s2_f1     = 2 * s2_prec * s2_recall / (s2_prec + s2_recall) if (s2_prec + s2_recall) > 0 else 0
    s2_acc    = (s2_tp + s2_tn) / len(s2) if len(s2) > 0 else 0

# Stage 3
total_s3 = alerts_s3 = critical_n = high_n = medium_n = low_n = 0
if s3_loaded:
    total_s3   = len(stage3_df)
    alerts_s3  = int(stage3_df["stage3_alert"].sum())
    critical_n = int((stage3_df["risk_level"] == "CRITICAL").sum())
    high_n     = int((stage3_df["risk_level"] == "HIGH").sum())
    medium_n   = int((stage3_df["risk_level"] == "MEDIUM").sum())
    low_n      = int((stage3_df["risk_level"] == "LOW").sum())

# ══ TABS ═════════════════════════════════════════════════════════════
tab_overview, tab_s1, tab_s2, tab_s3, tab_alerts, tab_check, tab_ml = st.tabs([
    "📊 Overview", "🔍 Stage 1", "🧠 Stage 2", "⚖️ Stage 3", "🚨 Alerts", "🛡️ Check a Message", "📈 ML Performance"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Pipeline at a Glance")
    st.caption(f"Evaluated on {len(df):,} Telegram messages after financial content filtering")
    st.markdown("")

    # Top row: headline numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages Analysed", f"{len(df):,}")
    c2.metric("Ponzi-Related",     f"{df['is_ponzi'].sum():,}")
    c3.metric("Stage 2 F1 Score",  f"{s2_f1:.1%}" if s2_loaded else "—")
    c4.metric("Stage 3 Alerts",    f"{alerts_s3:,}" if s3_loaded else "—")

    st.markdown("---")

    # Pipeline flow visualisation
    st.markdown("**Message Flow Through Pipeline**")

    flow_data = {
        "Stage": ["Total Messages", "Stage 1 Flagged", "Stage 2 Confirmed", "Stage 3 Alerts"],
        "Count": [
            len(df),
            int(df["stage1_flagged"].sum()),
            total_s3 if s3_loaded else (s2["pred_ponzi"].sum() if s2_loaded else 0),
            alerts_s3 if s3_loaded else 0,
        ],
    }
    flow_df = pd.DataFrame(flow_data)

    fig = go.Figure(go.Funnel(
        y=flow_df["Stage"],
        x=flow_df["Count"],
        textinfo="value+percent initial",
        marker={"color": ["#457b9d", "#f4a261", "#e76f51", "#d62828"]},
    ))
    fig.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Stage comparison table - simple and clean
    st.markdown("**Stage Performance Summary**")

    summary_rows = []
    summary_rows.append({
        "Stage": "Stage 1 — Keyword Filter",
        "Role": "Recall-optimised gate",
        "Precision": f"{s1_prec:.2%}",
        "Recall": f"{s1_recall:.2%}",
        "F1 Score": f"{s1_f1:.2%}",
    })
    if s2_loaded:
        summary_rows.append({
            "Stage": "Stage 2 — Fine-tuned XLM-RoBERTa-base",
            "Role": "Multilingual classifier",
            "Precision": f"{s2_prec:.2%}",
            "Recall": f"{s2_recall:.2%}",
            "F1 Score": f"{s2_f1:.2%}",
        })
    if s3_loaded:
        summary_rows.append({
            "Stage": "Stage 3 — Rule Reasoner",
            "Role": "Explainability + risk scoring",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1 Score": "N/A",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: STAGE 1
# ═══════════════════════════════════════════════════════════════════════
with tab_s1:
    st.subheader("Stage 1 — UltraFast Keyword Filter")
    st.caption("A lightweight gate that catches everything suspicious. Optimised for recall; precision refined later by Stage 2.")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{s1_prec:.2%}")
    c2.metric("Recall",    f"{s1_recall:.2%}")
    c3.metric("F1 Score",  f"{s1_f1:.2%}")

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("**Most-Triggered Red-Flag Keywords**")
        top_flags = pd.Series(
            [kw for flags in df["red_flags"] for kw in flags]
        ).value_counts().head(8).reset_index()
        top_flags.columns = ["Keyword", "Messages"]
        fig = px.bar(
            top_flags, x="Messages", y="Keyword", orientation="h",
            color="Messages", color_continuous_scale="Reds"
        )
        fig.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Classification Outcomes**")
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[s1_tp, s1_fn], [s1_fp, s1_tn]],
            x=["Flagged", "Not Flagged"],
            y=["Ponzi", "Legitimate"],
            colorscale="RdBu_r",
            text=[[f"{s1_tp:,}", f"{s1_fn:,}"], [f"{s1_fp:,}", f"{s1_tn:,}"]],
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False
        ))
        fig_cm.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: STAGE 2
# ═══════════════════════════════════════════════════════════════════════
with tab_s2:
    st.subheader("Stage 2 — Multilingual Deep Analyzer")
    st.caption("Fine-tuned XLM-RoBERTa-base multilingual model. Classifies messages passed by Stage 1.")
    st.markdown("")

    if not s2_loaded:
        st.info("Stage 2 results not available. Run `python stage2_finetune.py` first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{s2_prec:.2%}",
                  delta=f"+{(s2_prec - s1_prec)*100:.1f}pp vs Stage 1")
        c2.metric("Recall",    f"{s2_recall:.2%}",
                  delta=f"{(s2_recall - s1_recall)*100:.1f}pp vs Stage 1")
        c3.metric("F1 Score",  f"{s2_f1:.2%}",
                  delta=f"+{(s2_f1 - s1_f1)*100:.1f}pp vs Stage 1")
        c4.metric("Accuracy",  f"{s2_acc:.2%}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Classification Outcomes**")
            fig_cm = go.Figure(data=go.Heatmap(
                z=[[s2_tp, s2_fn], [s2_fp, s2_tn]],
                x=["Predicted Ponzi", "Predicted Legit"],
                y=["Actually Ponzi", "Actually Legit"],
                colorscale="RdBu_r",
                text=[[f"{s2_tp:,}", f"{s2_fn:,}"], [f"{s2_fp:,}", f"{s2_tn:,}"]],
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=False
            ))
            fig_cm.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_b:
            st.markdown("**Stage 1 vs Stage 2 Comparison**")
            comp = pd.DataFrame({
                "Metric":   ["Precision", "Recall", "F1"],
                "Stage 1":  [s1_prec * 100, s1_recall * 100, s1_f1 * 100],
                "Stage 2":  [s2_prec * 100, s2_recall * 100, s2_f1 * 100],
            })
            comp_melted = comp.melt(id_vars="Metric", var_name="Stage", value_name="Score (%)")
            fig = px.bar(
                comp_melted, x="Metric", y="Score (%)", color="Stage",
                barmode="group",
                color_discrete_sequence=["#457b9d", "#e63946"],
                text_auto=".1f"
            )
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Training diagnostics (loss curves, PR, ROC) ──────────────
        st.markdown("**Training Diagnostics**")

        history_path = "training_history.csv"
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)

            # Build per-epoch summary
            train_h = history_df[history_df["loss"].notna()].copy()
            train_h["epoch_int"] = train_h["epoch"].round().astype(int)
            train_loss = train_h.groupby("epoch_int")["loss"].mean().reset_index()
            train_loss.columns = ["epoch", "train_loss"]

            val_h = history_df[history_df["eval_loss"].notna()].copy()
            val_h["epoch_int"] = val_h["epoch"].round().astype(int)
            val_h = val_h[["epoch_int", "eval_loss", "eval_precision",
                           "eval_recall", "eval_f1"]].rename(columns={"epoch_int": "epoch"})

            merged_h = pd.merge(train_loss, val_h, on="epoch", how="outer").sort_values("epoch")

            col_l, col_r = st.columns(2)

            with col_l:
                st.caption("Training and Validation Loss per Epoch")
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["train_loss"],
                    mode="lines+markers", name="Training Loss",
                    line=dict(color="#2E86AB", width=3),
                    marker=dict(size=10)
                ))
                fig_loss.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_loss"],
                    mode="lines+markers", name="Validation Loss",
                    line=dict(color="#E63946", width=3),
                    marker=dict(size=10, symbol="square")
                ))
                fig_loss.update_layout(
                    height=340, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Epoch", yaxis_title="Loss",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_r:
                st.caption("Validation Metrics per Epoch")
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_precision"],
                    mode="lines+markers", name="Precision",
                    line=dict(color="#3B82F6", width=2.5), marker=dict(size=9)
                ))
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_recall"],
                    mode="lines+markers", name="Recall",
                    line=dict(color="#10B981", width=2.5), marker=dict(size=9, symbol="square")
                ))
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_f1"],
                    mode="lines+markers", name="F1 Score",
                    line=dict(color="#F59E0B", width=2.5), marker=dict(size=10, symbol="triangle-up")
                ))
                fig_metrics.update_layout(
                    height=340, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Epoch", yaxis_title="Score",
                    yaxis=dict(range=[0, 1.05]),
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.info("Training history not available. Run `python training_visualizations.py` to generate training diagnostics.")

        # ── Precision-Recall and ROC curves (computed from test results) ──
        if "stage2_prob_ponzi" in s2.columns:
            from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc

            y_true_arr = s2["is_ponzi"].astype(int).values
            y_prob_arr = s2["stage2_prob_ponzi"].values

            prec_arr, rec_arr, _ = precision_recall_curve(y_true_arr, y_prob_arr)
            fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
            pr_auc = sk_auc(rec_arr, prec_arr)
            roc_auc = sk_auc(fpr, tpr)

            col_pr, col_roc = st.columns(2)

            with col_pr:
                st.caption(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rec_arr, y=prec_arr, mode="lines",
                    fill="tozeroy", line=dict(color="#D04A2F", width=3),
                    fillcolor="rgba(208,74,47,0.12)", name="PR Curve"
                ))
                fig_pr.update_layout(
                    height=340, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Recall", yaxis_title="Precision",
                    xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                    showlegend=False
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            with col_roc:
                st.caption(f"ROC Curve (AUC = {roc_auc:.3f})")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    fill="tozeroy", line=dict(color="#1D9E75", width=3),
                    fillcolor="rgba(29,158,117,0.12)", name="ROC Curve"
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    name="Random Classifier"
                ))
                fig_roc.update_layout(
                    height=340, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                    showlegend=False
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---")

        # Sample predictions in expanders (hide complexity unless needed)
        st.markdown("**Sample Predictions**")
        tab1, tab2, tab3 = st.tabs(["✅ Correctly Flagged", "❌ False Positives", "⚠️ Missed Ponzi"])
        with tab1:
            samples = s2[s2["pred_ponzi"] & s2["is_ponzi"]][["text"]].head(5)
            samples.columns = ["Message"]
            st.dataframe(samples, use_container_width=True, hide_index=True, height=200)
        with tab2:
            samples = s2[s2["pred_ponzi"] & ~s2["is_ponzi"]][["text"]].head(5)
            samples.columns = ["Message"]
            st.dataframe(samples, use_container_width=True, hide_index=True, height=200)
        with tab3:
            samples = s2[~s2["pred_ponzi"] & s2["is_ponzi"]][["text"]].head(5)
            samples.columns = ["Message"]
            st.dataframe(samples, use_container_width=True, hide_index=True, height=200)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: STAGE 3
# ═══════════════════════════════════════════════════════════════════════
with tab_s3:
    st.subheader("Stage 3 — Structured Red-Flag Reasoner")
    st.caption("Rule-based engine enumerating specific fraud indicators and assigning structured risk scores.")
    st.markdown("")

    if not s3_loaded:
        st.info("Stage 3 results not available. Run `python stage3.py` first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages Analysed", f"{total_s3:,}")
        c2.metric("Alerts Raised",     f"{alerts_s3:,}")
        c3.metric("Critical Risk",     f"{critical_n:,}")
        c4.metric("High Risk",         f"{high_n:,}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Risk Level Distribution**")
            risk_df = pd.DataFrame({
                "Risk": ["Critical", "High", "Medium", "Low"],
                "Count": [critical_n, high_n, medium_n, low_n]
            })
            fig = px.bar(
                risk_df, x="Risk", y="Count",
                color="Risk",
                color_discrete_map={
                    "Critical": "#d00000", "High": "#e85d04",
                    "Medium": "#f4a261", "Low": "#457b9d"
                },
                text="Count"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=340, margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Risk Score Distribution**")
            fig = px.histogram(
                stage3_df, x="risk_score", nbins=15,
                color_discrete_sequence=["#e63946"]
            )
            fig.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Messages",
                height=340, margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: ALERTS
# ═══════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.subheader("Alert Queue")
    st.caption("Stage 3 alerts sorted by risk score. Use the filter to drill down by risk level.")
    st.markdown("")

    if not s3_loaded:
        st.info("No alerts to display. Run `python stage3.py` first.")
    else:
        # Filter control
        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            label_visibility="collapsed"
        )

        filtered = stage3_df if risk_filter == "All" else stage3_df[stage3_df["risk_level"] == risk_filter]
        filtered = filtered.sort_values("risk_score", ascending=False).reset_index(drop=True)

        st.caption(f"Showing {len(filtered):,} alerts")

        def style_risk(val):
            colours = {
                "CRITICAL": "background-color: #d00000; color: white; font-weight: bold;",
                "HIGH":     "background-color: #e85d04; color: white; font-weight: bold;",
                "MEDIUM":   "background-color: #f4a261; color: white;",
                "LOW":      "background-color: #457b9d; color: white;",
            }
            return colours.get(val, "")

        display_df = filtered[[
            "risk_level", "risk_score", "flags_triggered", "text"
        ]].rename(columns={
            "risk_level": "Risk",
            "risk_score": "Score",
            "flags_triggered": "Flags",
            "text": "Message"
        })

        st.dataframe(
            display_df.style.map(style_risk, subset=["Risk"]),
            use_container_width=True,
            height=500,
            hide_index=True,
            column_config={
                "Risk":    st.column_config.TextColumn(width="small"),
                "Score":   st.column_config.NumberColumn(width="small"),
                "Flags":   st.column_config.NumberColumn(width="small"),
                "Message": st.column_config.TextColumn(width="large"),
            }
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 6: CHECK A MESSAGE (Public-facing user tool)
# ═══════════════════════════════════════════════════════════════════════
with tab_check:
    st.subheader("Is this message a Ponzi scheme?")
    st.caption("Paste any suspicious message you received and PonziGuard will check it for known Ponzi patterns.")
    st.markdown("")

    # ── Investment-fraud sub-filter vocabulary (Stage 2.5) ───────────
    INVESTMENT_VOCAB = [
        "invest", "investment", "investing", "investor", "portfolio",
        "return", "returns", "roi", "yield", "profit", "profits",
        "earnings", "income", "passive income", "compound", "capital",
        "trading", "trader", "trade", "forex", "crypto", "bitcoin",
        "deposit", "withdraw", "withdrawal", "payout", "package", "plan",
        "tier", "membership", "daily profit", "weekly profit",
        "guaranteed return", "double your", "triple your", "ponzi",
        "pyramid", "mlm", "downline", "upline", "mmm", "cbex", "racksterli",
        "mining", "staking", "high yield", "trading bot", "mba forex",
    ]

    # ── Rule set (matches Stage 3 v2) ─────────────────────────────────
    PUBLIC_RULES = [
        {"name": "Guaranteed Returns",      "weight": 5, "plain": "Promises money is 100% safe — real investments cannot guarantee returns",
         "patterns": ["guaranteed return", "guaranteed profit", "guaranteed payout", "100% guaranteed", "100% safe", "risk free investment", "risk-free investment", "no risk", "zero risk", "assured returns"]},
        {"name": "Unrealistic Promises",    "weight": 5, "plain": "Promises returns that are impossible (200%, doubled money, 10x)",
         "patterns": ["200%", "300%", "400%", "500%", "1000%", "double your money", "triple your money", "multiply your", "10x returns", "100x"]},
        {"name": "Daily/Weekly Profits",    "weight": 5, "plain": "Promises daily or weekly payouts — classic Ponzi signature",
         "patterns": ["daily profit", "daily income", "daily return", "weekly profit", "weekly income", "weekly return", "earn daily", "paid daily"]},
        {"name": "Pyramid Structure",       "weight": 5, "plain": "Uses pyramid or MLM language — money comes from new members, not real returns",
         "patterns": ["pyramid", "ponzi", "mlm", "matrix", "downline", "upline", "multi-level"]},
        {"name": "Known Ponzi Schemes",     "weight": 5, "plain": "Mentions a known fraudulent scheme",
         "patterns": ["mmm", "loom money", "cbex", "racksterli", "racksterly", "mba forex", "twinkas", "onecoin", "bitconnect"]},
        {"name": "Recruitment Pay",         "weight": 5, "plain": "Pays you to recruit others — Ponzi mechanic, not real business",
         "patterns": ["referral commission", "referral bonus", "refer and earn", "recruit and earn", "downline commission", "binary plan"]},
        {"name": "VIP / Tiered Packages",   "weight": 4, "plain": "Sells investment packages by tier (Gold, Diamond, VIP)",
         "patterns": ["starter package", "bronze package", "gold package", "platinum package", "diamond package", "vip package", "premium package"]},
        {"name": "Passive Income Hype",     "weight": 4, "plain": "Sells the dream of money without work",
         "patterns": ["passive income", "earn while you sleep", "no work required", "financial freedom", "quit your job", "retire early"]},
        {"name": "Urgency / Limited Slots", "weight": 2, "plain": "Pressures you to act fast — fraud loves urgency",
         "patterns": ["limited slots", "limited offer", "act now", "join now", "hurry", "closing soon", "last chance"]},
        {"name": "Trading Bots",            "weight": 2, "plain": "Mentions automated trading bots — common cover for Ponzi",
         "patterns": ["trading bot", "ai trading", "forex signals", "crypto signals", "mining bot"]},
        {"name": "Wallet Transfers",        "weight": 1, "plain": "Asks you to send money to a wallet or account",
         "patterns": ["wallet address", "send to wallet", "send funds", "pay to", "transfer to"]},
        {"name": "Fake Proof",              "weight": 1, "plain": "Shows testimonials or proof of payment — easily fabricated",
         "patterns": ["proof of payment", "screenshot", "testimonial", "i just received", "i just withdrew", "100% legit"]},
    ]

    def is_investment_message(text):
        t = str(text).lower()
        return any(v in t for v in INVESTMENT_VOCAB)

    def check_message(text):
        """Returns (risk_score, risk_level, triggered_rules, plain_explanations, is_investment)."""
        t = str(text).lower()

        # Stage 2.5 check — is it investment-related at all?
        if not is_investment_message(t):
            return 0, "NOT INVESTMENT", [], [], False

        # Stage 3 — match weighted rules
        triggered = []
        explanations = []
        score = 0
        for rule in PUBLIC_RULES:
            matched = [p for p in rule["patterns"] if p in t]
            if matched:
                triggered.append({"name": rule["name"], "weight": rule["weight"], "matched": matched})
                explanations.append(rule["plain"])
                score += rule["weight"]

        # Map to risk level
        if score >= 15:
            level = "CRITICAL"
        elif score >= 10:
            level = "HIGH"
        elif score >= 5:
            level = "MEDIUM"
        elif score >= 1:
            level = "LOW"
        else:
            level = "SAFE"

        return score, level, triggered, explanations, True

    # ── Example messages users can try ────────────────────────────────
    examples = {
        "Pick an example to try…": "",
        "Example 1: CBEX-style pitch": (
            "🚨 CBEX AI Trading is back! Deposit ₦150,000 and watch our bot earn you "
            "₦15,000 daily for 30 days. Withdrawals go straight to your Opay account. "
            "Limited slots — DM admin to register."
        ),
        "Example 2: MMM-style recruitment": (
            "MMM Nigeria Revival! Provide Help ₦50,000, Get Help ₦100,000 in 14 days. "
            "Verified by community admins. 30% bonus for first 50 participants this week. "
            "Join the WhatsApp group via link in bio."
        ),
        "Example 3: Legitimate bank notice": (
            "GTBank has announced new savings rates effective Monday. Visit your nearest "
            "branch or use the GTWorld app for more information."
        ),
        "Example 4: Phishing (not Ponzi)": (
            "Congrats: $25,758. Dear applicant, after further review of your "
            "application, your funds have been released. Click below to claim now."
        ),
    }

    # ── Input area ────────────────────────────────────────────────────
    selected_example = st.selectbox("Try an example or paste your own:", list(examples.keys()))
    default_text = examples[selected_example] if selected_example != "Pick an example to try…" else ""

    user_text = st.text_area(
        "Paste the suspicious message here:",
        value=default_text,
        height=160,
        placeholder="Example: 'Deposit ₦50,000 and earn ₦10,000 daily, guaranteed 100% safe...'",
        key=f"user_input_{selected_example}",
    )

    check_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)

    st.markdown("")

    # ── Results ───────────────────────────────────────────────────────
    if check_clicked and user_text.strip():
        score, level, triggered, explanations, is_investment = check_message(user_text)

        # Big verdict banner
        verdicts = {
            "CRITICAL":       ("🚨 CRITICAL RISK",   "#d32f2f", "white"),
            "HIGH":           ("⚠️ HIGH RISK",       "#f57c00", "white"),
            "MEDIUM":         ("⚡ MEDIUM RISK",     "#fbc02d", "black"),
            "LOW":            ("🔵 LOW RISK",        "#1976d2", "white"),
            "SAFE":           ("✅ NO PONZI MARKERS","#388e3c", "white"),
            "NOT INVESTMENT": ("ℹ️ NOT INVESTMENT-RELATED","#616161","white"),
        }
        label, bg, fg = verdicts[level]
        st.markdown(
            f"<div style='background:{bg};color:{fg};padding:24px;border-radius:12px;"
            f"text-align:center;font-size:1.6rem;font-weight:600;margin-bottom:16px'>"
            f"{label}<br><span style='font-size:1rem;font-weight:400'>Risk score: {score}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Verdict explanation
        if level == "NOT INVESTMENT":
            st.info(
                "This message does not appear to be about an investment offer. "
                "PonziGuard checks specifically for Ponzi scheme promotion. "
                "If you suspect another type of fraud (phishing, romance scam, lottery scam), "
                "report it to the appropriate authority."
            )
        elif level == "CRITICAL":
            st.error(
                "**Do not invest. Do not send any money.** This message shows multiple "
                "patterns specific to Ponzi schemes. Block the sender. Report to your "
                "country's financial regulator (EFCC in Nigeria, SEBI in India, OJK in Indonesia)."
            )
        elif level == "HIGH":
            st.error(
                "**Treat as a likely Ponzi scheme.** Do not send money. Do not recruit "
                "friends or family. Verify the platform with your financial regulator before "
                "taking any action."
            )
        elif level == "MEDIUM":
            st.warning(
                "**Be cautious.** Some Ponzi indicators are present. Research the platform "
                "independently. Search for it on regulator websites and recent news. "
                "If anyone pressures you to invest quickly, that is itself a warning sign."
            )
        elif level == "LOW":
            st.warning(
                "**Minor red flags detected.** The message uses some language associated with "
                "investment fraud but lacks the strong markers of a confirmed Ponzi scheme. "
                "Stay cautious and verify independently."
            )
        elif level == "SAFE":
            st.success(
                "**No Ponzi-specific patterns detected.** The message appears to be "
                "investment-related but contains none of the documented Ponzi scheme markers. "
                "Note: this does not guarantee the offer is legitimate. Always verify with "
                "your financial regulator before investing."
            )

        # Triggered rules detail
        if triggered:
            st.markdown("---")
            st.markdown("### 🚩 Red flags detected")
            for rule in triggered:
                with st.expander(f"**{rule['name']}**  (weight: {rule['weight']})"):
                    st.markdown(f"*{[r['plain'] for r in PUBLIC_RULES if r['name'] == rule['name']][0]}*")
                    st.markdown(f"**Matched phrases:** `{', '.join(rule['matched'])}`")

        # Always show educational footer
        st.markdown("---")
        st.markdown("### 📚 How to protect yourself")
        st.markdown(
            "- **No investment is risk-free.** Anyone who guarantees returns is lying.\n"
            "- **Real returns are slow.** 5%, 10%, even 15% per year is realistic. "
            "Daily or weekly profits are not.\n"
            "- **You cannot earn by recruiting.** If a platform pays you to bring in friends, "
            "the money is coming from those friends, not from real investment.\n"
            "- **Check regulators first.** EFCC (Nigeria), SEC Nigeria, SEBI (India), "
            "OJK (Indonesia) all publish lists of unregistered platforms.\n"
            "- **If unsure, walk away.** A legitimate opportunity will still be there tomorrow."
        )

    elif check_clicked:
        st.warning("Please paste a message to check.")

    else:
        st.info(
            "👆 Paste any suspicious message above and click **Check Message** "
            "to see if it shows known Ponzi scheme patterns. You can also pick one of the "
            "example messages from the dropdown to see how it works."
        )

    # Disclaimer footer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** PonziGuard is a research prototype. It checks messages against "
        "documented Ponzi scheme patterns but cannot guarantee detection of all fraud. "
        "Always verify investment offers with your country's financial regulator before "
        "sending any money. This tool does not provide financial or legal advice."
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 7: ML PERFORMANCE (comprehensive supervisor-facing diagnostics)
# ═══════════════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("Stage 2 Model Performance — Full Diagnostics")
    st.caption("Comprehensive ML evaluation showing how the fine-tuned XLM-RoBERTa-base model learned and how it performs on held-out data.")
    st.markdown("")

    # ── Configuration panel ──────────────────────────────────────────
    with st.expander("ℹ️ Training Configuration", expanded=False):
        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        cfg_c1.markdown("""
**Model**
- Architecture: XLM-RoBERTa-base
- Parameters: ~270 million
- Transformer layers: 12
        """)
        cfg_c2.markdown("""
**Training**
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Weight decay: 0.01
- Max length: 128 tokens
        """)
        cfg_c3.markdown("""
**Data**
- Total messages: 15,010
- Train: 10,508 (70%)
- Validation: 1,500 (10%)
- Test: 3,002 (20%)
- Stratified split
        """)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 1: TRAINING DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### 📉 Training Diagnostics")
    st.caption("How the model learned over the course of training. Loss should decrease; validation metrics should rise and stabilise.")

    history_path = "training_history.csv"
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)

        # Build per-epoch summary
        train_h = history_df[history_df["loss"].notna()].copy()
        if not train_h.empty:
            train_h["epoch_int"] = train_h["epoch"].round().astype(int)
            train_loss_epoch = train_h.groupby("epoch_int")["loss"].mean().reset_index()
            train_loss_epoch.columns = ["epoch", "train_loss"]
        else:
            train_loss_epoch = pd.DataFrame(columns=["epoch", "train_loss"])

        val_h = history_df[history_df["eval_loss"].notna()].copy()
        if not val_h.empty:
            val_h["epoch_int"] = val_h["epoch"].round().astype(int)
            val_metrics = val_h.groupby("epoch_int").agg({
                "eval_loss": "mean", "eval_precision": "mean",
                "eval_recall": "mean", "eval_f1": "mean",
                "eval_accuracy": "mean" if "eval_accuracy" in val_h.columns else "mean"
            }).reset_index().rename(columns={"epoch_int": "epoch"})
        else:
            val_metrics = pd.DataFrame()

        merged_h = pd.merge(train_loss_epoch, val_metrics, on="epoch", how="outer").sort_values("epoch")

        # Row 1: Loss and Metrics curves
        row1_c1, row1_c2 = st.columns(2)

        with row1_c1:
            st.markdown("**1. Training and Validation Loss**")
            st.caption("Both curves trending down means the model is learning. If validation rises while training falls, that's overfitting.")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=merged_h["epoch"], y=merged_h["train_loss"],
                mode="lines+markers", name="Training Loss",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=11)
            ))
            if "eval_loss" in merged_h.columns:
                fig_loss.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_loss"],
                    mode="lines+markers", name="Validation Loss",
                    line=dict(color="#E63946", width=3),
                    marker=dict(size=11, symbol="square")
                ))
            fig_loss.update_layout(
                height=380, margin=dict(t=10, b=40, l=10, r=10),
                xaxis_title="Epoch", yaxis_title="Loss",
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with row1_c2:
            st.markdown("**2. Validation Metrics per Epoch**")
            st.caption("All metrics should rise toward 1.0 and stabilise as training progresses.")
            fig_metrics = go.Figure()
            if "eval_precision" in merged_h.columns:
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_precision"],
                    mode="lines+markers", name="Precision",
                    line=dict(color="#3B82F6", width=2.5), marker=dict(size=10)
                ))
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_recall"],
                    mode="lines+markers", name="Recall",
                    line=dict(color="#10B981", width=2.5), marker=dict(size=10, symbol="square")
                ))
                fig_metrics.add_trace(go.Scatter(
                    x=merged_h["epoch"], y=merged_h["eval_f1"],
                    mode="lines+markers", name="F1 Score",
                    line=dict(color="#F59E0B", width=2.5), marker=dict(size=11, symbol="triangle-up")
                ))
                if "eval_accuracy" in merged_h.columns:
                    fig_metrics.add_trace(go.Scatter(
                        x=merged_h["epoch"], y=merged_h["eval_accuracy"],
                        mode="lines+markers", name="Accuracy",
                        line=dict(color="#8B5CF6", width=2.5), marker=dict(size=10, symbol="diamond")
                    ))
            fig_metrics.update_layout(
                height=380, margin=dict(t=10, b=40, l=10, r=10),
                xaxis_title="Epoch", yaxis_title="Score",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified",
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        # Row 2: Per-step loss and learning rate
        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            st.markdown("**3. Training Loss per Step (Granular)**")
            st.caption("Step-level loss showing the noisy descent during training. Smoother = better learning.")
            if "loss" in history_df.columns:
                step_loss = history_df[history_df["loss"].notna()].copy()
                fig_step = go.Figure()
                fig_step.add_trace(go.Scatter(
                    x=step_loss["step"], y=step_loss["loss"],
                    mode="lines", name="Training Loss",
                    line=dict(color="#2E86AB", width=1.5),
                ))
                # Rolling mean for smoother trend
                if len(step_loss) > 10:
                    step_loss["rolling"] = step_loss["loss"].rolling(window=10, min_periods=1).mean()
                    fig_step.add_trace(go.Scatter(
                        x=step_loss["step"], y=step_loss["rolling"],
                        mode="lines", name="Smoothed (10-step rolling)",
                        line=dict(color="#E63946", width=2.5),
                    ))
                fig_step.update_layout(
                    height=380, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Training Step", yaxis_title="Loss",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_step, use_container_width=True)

        with row2_c2:
            st.markdown("**4. Learning Rate Schedule**")
            st.caption("Shows how the learning rate decayed during training. Standard linear-decay scheduler.")
            if "learning_rate" in history_df.columns:
                lr_data = history_df[history_df["learning_rate"].notna()]
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=lr_data["step"], y=lr_data["learning_rate"],
                    mode="lines", name="Learning Rate",
                    line=dict(color="#7C3AED", width=2.5),
                    fill="tozeroy", fillcolor="rgba(124,58,237,0.12)",
                ))
                fig_lr.update_layout(
                    height=380, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Training Step", yaxis_title="Learning Rate",
                    yaxis=dict(tickformat=".0e"),
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            else:
                st.info("Learning rate data not available in training history.")
    else:
        st.warning("⚠️ Training history not available. Run `python stage2_finetune.py` to generate `training_history.csv` and unlock these charts.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 2: CLASSIFICATION PERFORMANCE
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Classification Performance (Held-Out Test Set)")
    st.caption("How the trained model performs on data it never saw during training. 3,002 messages, stratified split.")

    if os.path.exists("stage2_finetuned_results.csv"):
        s2_full = pd.read_csv("stage2_finetuned_results.csv")
        s2_full["is_ponzi"] = (s2_full["label"] == "ponzi_related").astype(int)
        s2_full["pred_ponzi"] = (s2_full["stage2_pred_label"] == "ponzi_related").astype(int)

        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score,
            confusion_matrix as sk_cm, precision_recall_curve, roc_curve, auc as sk_auc
        )

        y_true = s2_full["is_ponzi"].values
        y_pred = s2_full["pred_ponzi"].values

        # Headline metrics
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
        m_c2.metric("Recall",    f"{recall_score(y_true, y_pred):.2%}")
        m_c3.metric("F1 Score",  f"{f1_score(y_true, y_pred):.2%}")
        m_c4.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.2%}")

        st.markdown("")

        # Row 3: Confusion matrix + Per-class metrics
        row3_c1, row3_c2 = st.columns(2)

        with row3_c1:
            st.markdown("**5. Confusion Matrix Heatmap**")
            st.caption("Diagonal = correct predictions. Off-diagonal = errors. Most weight should be on the diagonal.")
            cm = sk_cm(y_true, y_pred)
            tn, fp = int(cm[0, 0]), int(cm[0, 1])
            fn, tp = int(cm[1, 0]), int(cm[1, 1])

            cm_text = [
                [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
                [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]
            ]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted: Legitimate", "Predicted: Ponzi"],
                y=["Actual: Legitimate", "Actual: Ponzi"],
                colorscale="Blues",
                text=cm_text,
                texttemplate="%{text}",
                textfont={"size": 18, "color": "white"},
                showscale=True,
            ))
            fig_cm.update_layout(
                height=400, margin=dict(t=10, b=40, l=10, r=10),
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with row3_c2:
            st.markdown("**6. Per-Class Metrics**")
            st.caption("Precision, recall, F1 broken down by class. Both classes should perform well, not just one.")
            ponzi_p = precision_score(y_true, y_pred, pos_label=1)
            ponzi_r = recall_score(y_true, y_pred, pos_label=1)
            ponzi_f = f1_score(y_true, y_pred, pos_label=1)
            legit_p = precision_score(y_true, y_pred, pos_label=0)
            legit_r = recall_score(y_true, y_pred, pos_label=0)
            legit_f = f1_score(y_true, y_pred, pos_label=0)

            fig_pc = go.Figure()
            fig_pc.add_trace(go.Bar(
                name="Ponzi", x=["Precision", "Recall", "F1"],
                y=[ponzi_p, ponzi_r, ponzi_f],
                marker_color="#D04A2F",
                text=[f"{v:.1%}" for v in [ponzi_p, ponzi_r, ponzi_f]],
                textposition="outside",
            ))
            fig_pc.add_trace(go.Bar(
                name="Legitimate", x=["Precision", "Recall", "F1"],
                y=[legit_p, legit_r, legit_f],
                marker_color="#1D9E75",
                text=[f"{v:.1%}" for v in [legit_p, legit_r, legit_f]],
                textposition="outside",
            ))
            fig_pc.update_layout(
                barmode="group", height=400,
                margin=dict(t=10, b=40, l=10, r=10),
                yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_pc, use_container_width=True)

        # Row 4: PR and ROC curves
        if "stage2_prob_ponzi" in s2_full.columns:
            row4_c1, row4_c2 = st.columns(2)
            y_prob = s2_full["stage2_prob_ponzi"].values

            with row4_c1:
                st.markdown("**7. Precision-Recall Curve**")
                st.caption("Trade-off between precision and recall across all decision thresholds. Higher AUC = better.")
                prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = sk_auc(rec_arr, prec_arr)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rec_arr, y=prec_arr, mode="lines",
                    fill="tozeroy", line=dict(color="#D04A2F", width=3),
                    fillcolor="rgba(208,74,47,0.12)",
                    name=f"PR Curve (AUC = {pr_auc:.4f})",
                ))
                # Baseline: % positive class
                baseline = y_true.mean()
                fig_pr.add_hline(y=baseline, line_dash="dash", line_color="gray",
                                 annotation_text=f"Random baseline ({baseline:.2%})")
                fig_pr.update_layout(
                    height=400, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Recall", yaxis_title="Precision",
                    xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.05]),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_pr, use_container_width=True)
                st.metric("PR AUC", f"{pr_auc:.4f}")

            with row4_c2:
                st.markdown("**8. ROC Curve**")
                st.caption("True positive rate vs false positive rate. The closer to the top-left, the better. AUC = 1.0 is perfect.")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = sk_auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    fill="tozeroy", line=dict(color="#1D9E75", width=3),
                    fillcolor="rgba(29,158,117,0.12)",
                    name=f"ROC (AUC = {roc_auc:.4f})",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    name="Random classifier",
                ))
                fig_roc.update_layout(
                    height=400, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.metric("ROC AUC", f"{roc_auc:.4f}")

            st.markdown("---")

            # ─────────────────────────────────────────────────────────
            # SECTION 3: PREDICTION ANALYSIS
            # ─────────────────────────────────────────────────────────
            st.markdown("### 🔬 Prediction Analysis")
            st.caption("How confident the model is in its predictions and how confidence relates to correctness.")

            row5_c1, row5_c2 = st.columns(2)

            with row5_c1:
                st.markdown("**9. Predicted Probability Distribution**")
                st.caption("Distribution of model confidence across all test predictions. A well-calibrated model has high density near 0 and 1.")
                s2_full["pred_class"] = s2_full["pred_ponzi"].map({0: "Predicted Legitimate", 1: "Predicted Ponzi"})
                fig_hist = px.histogram(
                    s2_full, x="stage2_prob_ponzi", color="pred_class",
                    nbins=50, opacity=0.8,
                    color_discrete_map={"Predicted Legitimate": "#1D9E75", "Predicted Ponzi": "#D04A2F"},
                )
                fig_hist.update_layout(
                    height=400, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Predicted Probability of Ponzi",
                    yaxis_title="Number of Messages",
                    legend=dict(orientation="h", y=-0.2, title=None),
                    bargap=0.02,
                )
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Decision threshold (0.5)")
                st.plotly_chart(fig_hist, use_container_width=True)

            with row5_c2:
                st.markdown("**10. Confidence by Prediction Outcome**")
                st.caption("Are the model's mistakes confident or uncertain? Confident wrong answers (high TP/TN confidence + visible FP/FN spread) reveal where the model is overconfident.")

                # Categorise each prediction
                def categorise(row):
                    if row["is_ponzi"] == 1 and row["pred_ponzi"] == 1: return "TP"
                    if row["is_ponzi"] == 0 and row["pred_ponzi"] == 0: return "TN"
                    if row["is_ponzi"] == 0 and row["pred_ponzi"] == 1: return "FP"
                    if row["is_ponzi"] == 1 and row["pred_ponzi"] == 0: return "FN"

                s2_full["outcome"] = s2_full.apply(categorise, axis=1)
                order = ["TP", "TN", "FP", "FN"]
                color_map = {"TP": "#1D9E75", "TN": "#3B82F6", "FP": "#F59E0B", "FN": "#D04A2F"}

                fig_box = go.Figure()
                for outcome in order:
                    subset = s2_full[s2_full["outcome"] == outcome]
                    if len(subset) > 0:
                        fig_box.add_trace(go.Box(
                            y=subset["stage2_prob_ponzi"],
                            name=f"{outcome} (n={len(subset)})",
                            marker_color=color_map[outcome],
                            boxmean=True,
                        ))
                fig_box.update_layout(
                    height=400, margin=dict(t=10, b=40, l=10, r=10),
                    yaxis_title="Predicted Probability of Ponzi",
                    yaxis=dict(range=[0, 1.05]),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("⚠️ Probability scores missing. Run `python stage2_finetune.py` to add the `stage2_prob_ponzi` column.")

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────
        # SECTION 4: TEXT LENGTH ANALYSIS
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 📏 Performance by Message Length")
        st.caption("Does the model handle short messages as well as long ones? Length-stratified accuracy reveals input-distribution edge cases.")

        s2_full["msg_length"] = s2_full["text"].astype(str).str.split().str.len()
        s2_full["length_bucket"] = pd.cut(
            s2_full["msg_length"],
            bins=[0, 10, 25, 50, 100, 1000],
            labels=["0-10", "11-25", "26-50", "51-100", "100+"]
        )
        s2_full["correct"] = (s2_full["is_ponzi"] == s2_full["pred_ponzi"]).astype(int)
        length_stats = s2_full.groupby("length_bucket").agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
        ).reset_index()
        length_stats["accuracy_pct"] = length_stats["accuracy"] * 100

        fig_len = go.Figure()
        fig_len.add_trace(go.Bar(
            x=length_stats["length_bucket"], y=length_stats["accuracy_pct"],
            marker_color="#3B82F6",
            text=[f"{v:.1f}%<br>(n={int(c):,})" for v, c in zip(length_stats["accuracy_pct"], length_stats["count"])],
            textposition="outside",
        ))
        fig_len.update_layout(
            height=380, margin=dict(t=10, b=40, l=10, r=10),
            xaxis_title="Message Length (words)",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 110]),
        )
        st.plotly_chart(fig_len, use_container_width=True)

    else:
        st.warning("⚠️ Stage 2 test results not available. Run `python stage2_finetune.py` to produce `stage2_finetuned_results.csv`.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 5: PER-LANGUAGE PERFORMANCE
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### 🌍 Per-Language Performance (Curated Multilingual Set)")
    st.caption("How the model and full cascade perform across English, Nigerian Pidgin, Hindi, and Indonesian — tests cross-lingual generalisation.")

    pl_path = "per_language_confusion.csv"
    if os.path.exists(pl_path):
        pl_df = pd.read_csv(pl_path)

        # Stage 2 alone vs cascade comparison
        s2_only = pl_df[pl_df["stage"] == "Stage 2 alone"].copy()
        cascade = pl_df[pl_df["stage"] == "Cascade"].copy()

        # ── Row: F1 comparison and Per-language Recall ─────────────────
        pl_c1, pl_c2 = st.columns(2)

        with pl_c1:
            st.markdown("**11. F1 Score by Language (Stage 2 alone vs Cascade)**")
            st.caption("How adding Stage 3 changes the F1 score per language. Reveals where the cascade helps vs hurts.")
            fig_lang_f1 = go.Figure()
            fig_lang_f1.add_trace(go.Bar(
                name="Stage 2 alone",
                x=s2_only["language"].str.title(),
                y=s2_only["f1"] * 100,
                marker_color="#3B82F6",
                text=[f"{v*100:.1f}%" for v in s2_only["f1"]],
                textposition="outside",
            ))
            fig_lang_f1.add_trace(go.Bar(
                name="Cascade (S2 + S3)",
                x=cascade["language"].str.title(),
                y=cascade["f1"] * 100,
                marker_color="#F59E0B",
                text=[f"{v*100:.1f}%" for v in cascade["f1"]],
                textposition="outside",
            ))
            fig_lang_f1.update_layout(
                barmode="group", height=400,
                margin=dict(t=10, b=40, l=10, r=10),
                yaxis_title="F1 Score (%)",
                yaxis=dict(range=[0, 110]),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_lang_f1, use_container_width=True)

        with pl_c2:
            st.markdown("**12. Precision and Recall per Language (Cascade)**")
            st.caption("Per-language precision-recall trade-off. Balanced bars indicate the cascade works equally well in both directions.")
            fig_lang_pr = go.Figure()
            fig_lang_pr.add_trace(go.Bar(
                name="Precision",
                x=cascade["language"].str.title(),
                y=cascade["precision"] * 100,
                marker_color="#1976d2",
                text=[f"{v*100:.1f}%" for v in cascade["precision"]],
                textposition="outside",
            ))
            fig_lang_pr.add_trace(go.Bar(
                name="Recall",
                x=cascade["language"].str.title(),
                y=cascade["recall"] * 100,
                marker_color="#388e3c",
                text=[f"{v*100:.1f}%" for v in cascade["recall"]],
                textposition="outside",
            ))
            fig_lang_pr.update_layout(
                barmode="group", height=400,
                margin=dict(t=10, b=40, l=10, r=10),
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 115]),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_lang_pr, use_container_width=True)

        # ── Confusion matrix table ─────────────────────────────────────
        st.markdown("**13. Per-Language Confusion Matrices**")
        st.caption("Exact TP/FP/FN/TN counts for each language. The Pidgin row reveals the rule-based Stage 3 limitation directly.")
        display_table = cascade[["language", "n", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]].copy()
        display_table["language"] = display_table["language"].str.title()
        display_table["precision"] = (display_table["precision"] * 100).round(2).astype(str) + "%"
        display_table["recall"]    = (display_table["recall"] * 100).round(2).astype(str) + "%"
        display_table["f1"]        = (display_table["f1"] * 100).round(2).astype(str) + "%"
        display_table.columns = ["Language", "n", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]
        st.dataframe(display_table, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Per-language data not available. Run `python per_language_confusion.py` to generate `per_language_confusion.csv`.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 6: LATENCY BENCHMARK
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### ⚡ Inference Latency Benchmark")
    st.caption("How fast each stage processes a message, and the full cascade end-to-end. Chapter 3 specified a sub-100ms target on GPU; the prototype meets it on CPU.")

    lat_path = "latency_benchmark.csv"
    if os.path.exists(lat_path):
        lat_df = pd.read_csv(lat_path)

        # Headline latency metrics
        e2e_row = lat_df[lat_df["stage"].str.contains("End-to-End", case=False, na=False)]
        if not e2e_row.empty:
            e2e = e2e_row.iloc[0]
            lat_c1, lat_c2, lat_c3, lat_c4 = st.columns(4)
            lat_c1.metric("Mean latency",   f"{e2e['mean']:.2f} ms")
            lat_c2.metric("Median latency", f"{e2e['median']:.2f} ms")
            lat_c3.metric("p95 latency",    f"{e2e['p95']:.2f} ms")
            lat_c4.metric("p99 latency",    f"{e2e['p99']:.2f} ms")
            st.markdown("")

        # ── Row: Per-stage mean + p95 ──────────────────────────────────
        lat_c_left, lat_c_right = st.columns(2)

        with lat_c_left:
            st.markdown("**14. Mean vs p95 Latency per Stage**")
            st.caption("Mean = average case. p95 = 95th percentile (the slow tail). Both should be well below the 100ms production target.")
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                name="Mean (ms)",
                x=lat_df["stage"], y=lat_df["mean"],
                marker_color="#1976d2",
                text=[f"{v:.2f}" for v in lat_df["mean"]],
                textposition="outside",
            ))
            fig_lat.add_trace(go.Bar(
                name="p95 (ms)",
                x=lat_df["stage"], y=lat_df["p95"],
                marker_color="#f57c00",
                text=[f"{v:.2f}" for v in lat_df["p95"]],
                textposition="outside",
            ))
            fig_lat.add_hline(y=100, line_dash="dash", line_color="red",
                              annotation_text="100ms target (Chapter 3)")
            fig_lat.update_layout(
                barmode="group", height=420,
                margin=dict(t=10, b=80, l=10, r=10),
                yaxis_title="Latency (ms)",
                xaxis=dict(tickangle=-25),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with lat_c_right:
            st.markdown("**15. Latency Percentiles (Tail Behaviour)**")
            st.caption("Mean, p95, p99 side by side. The gap between p95 and p99 reveals tail behaviour — important for real-time guarantees.")
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Bar(
                name="Median", x=lat_df["stage"], y=lat_df["median"],
                marker_color="#10B981",
            ))
            fig_pct.add_trace(go.Bar(
                name="p95", x=lat_df["stage"], y=lat_df["p95"],
                marker_color="#F59E0B",
            ))
            fig_pct.add_trace(go.Bar(
                name="p99", x=lat_df["stage"], y=lat_df["p99"],
                marker_color="#D04A2F",
            ))
            fig_pct.add_hline(y=100, line_dash="dash", line_color="red",
                              annotation_text="100ms target")
            fig_pct.update_layout(
                barmode="group", height=420,
                margin=dict(t=10, b=80, l=10, r=10),
                yaxis_title="Latency (ms)",
                xaxis=dict(tickangle=-25),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        # ── Insight callout ────────────────────────────────────────────
        st.info(
            "💡 **Key finding:** The end-to-end cascade is **faster than Stage 2 alone** "
            "(28 ms vs 51 ms). This is because Stage 1 dismisses non-financial messages "
            "in 0.11 ms — most input never reaches the expensive XLM-RoBERTa-base classifier. "
            "The cascade architecture saves expensive computation for messages that need it."
        )
    else:
        st.info("ℹ️ Latency data not available. Run `python benchmark_latency.py` to generate `latency_benchmark.csv`.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 7: ERROR ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Error Analysis")
    st.caption("Where the model fails and why. All errors are categorised into failure modes for honest evaluation.")

    err_path = "error_analysis.csv"
    if os.path.exists(err_path):
        err_df = pd.read_csv(err_path)

        # Headline numbers
        total_test = 60  # curated test set size
        total_errors = len(err_df)
        false_pos = len(err_df[err_df["label"] == "legitimate"])
        false_neg = len(err_df[err_df["label"] == "ponzi_related"])

        err_c1, err_c2, err_c3, err_c4 = st.columns(4)
        err_c1.metric("Total test examples", f"{total_test}")
        err_c2.metric("Misclassifications",  f"{total_errors}")
        err_c3.metric("False positives",     f"{false_pos}")
        err_c4.metric("False negatives",     f"{false_neg}",
                      delta="Zero missed Ponzi" if false_neg == 0 else None,
                      delta_color="normal")
        st.markdown("")

        # ── Row: error categories + per-language errors ────────────────
        err_left, err_right = st.columns(2)

        with err_left:
            st.markdown("**16. Error Categories**")
            st.caption("Failure modes grouped into named categories. Reveals systematic weaknesses.")
            cat_counts = err_df["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            cat_counts["pct"] = (cat_counts["count"] / cat_counts["count"].sum() * 100).round(1)

            category_descriptions = {
                "FINANCIAL_LEGIT":  "Bank notifications and fintech updates misread as Ponzi",
                "WARNING_MISREAD":  "Regulator advisories (SEC, RBI, OJK) misread as Ponzi",
                "OTHER":            "Utility bills, free webinars, e-commerce notifications",
                "CODE_MIXED":       "Code-switched messages mixing two languages",
                "SHORT_MESSAGE":    "Messages too short for context (<20 words)",
                "NUMERIC_HEAVY":    "Messages dominated by digits and symbols",
                "CULTURAL_MARKER":  "Language-specific scheme names not in keywords",
                "LEXICAL_DRIFT":    "Non-standard or rare scheme vocabulary",
            }

            fig_cat = go.Figure(data=[go.Bar(
                x=cat_counts["count"],
                y=cat_counts["category"],
                orientation="h",
                marker_color=["#D04A2F", "#F59E0B", "#3B82F6", "#1D9E75",
                              "#8B5CF6", "#10B981", "#EC4899", "#6366F1"][:len(cat_counts)],
                text=[f"{c} ({p:.1f}%)" for c, p in zip(cat_counts["count"], cat_counts["pct"])],
                textposition="outside",
            )])
            fig_cat.update_layout(
                height=400, margin=dict(t=10, b=40, l=10, r=80),
                xaxis_title="Number of Errors",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with err_right:
            st.markdown("**17. Errors by Language**")
            st.caption("Where errors cluster by language. Different languages may show different failure patterns.")
            if "language" in err_df.columns:
                lang_err = err_df.groupby("language").size().reset_index(name="errors")
                lang_err["language"] = lang_err["language"].str.title()
                fig_lerr = go.Figure(data=[go.Bar(
                    x=lang_err["language"], y=lang_err["errors"],
                    marker_color="#7C3AED",
                    text=lang_err["errors"],
                    textposition="outside",
                )])
                fig_lerr.update_layout(
                    height=400, margin=dict(t=10, b=40, l=10, r=10),
                    xaxis_title="Language",
                    yaxis_title="Number of Errors",
                )
                st.plotly_chart(fig_lerr, use_container_width=True)

        # ── Category descriptions and sample errors ────────────────────
        st.markdown("**Category Descriptions**")
        desc_md = "\n".join([
            f"- **{cat}** — {category_descriptions.get(cat, 'Other failure mode')}"
            for cat in cat_counts["category"]
        ])
        st.markdown(desc_md)

        # Sample errors per category
        st.markdown("**Sample Misclassifications**")
        for cat in cat_counts["category"]:
            cat_samples = err_df[err_df["category"] == cat].head(2)
            with st.expander(f"**{cat}** ({len(err_df[err_df['category'] == cat])} errors)"):
                for _, row in cat_samples.iterrows():
                    lang = row.get("language", "unknown")
                    true_l = row.get("label", "unknown")
                    pred_l = row.get("stage2_pred", "unknown")
                    text = str(row.get("text", ""))[:200]
                    st.markdown(f"- *[{lang}]* **True:** {true_l} → **Predicted:** {pred_l}")
                    st.markdown(f"  > {text}...")

        # ── Honest insight callout ────────────────────────────────────
        if false_neg == 0:
            st.success(
                "✅ **Zero false negatives.** Stage 2 missed no Ponzi messages in any language. "
                "All errors are false positives (legitimate messages flagged as Ponzi). "
                "This is the safer failure mode — Stage 3 then filters most false positives "
                "via structured rule matching."
            )
    else:
        st.info("ℹ️ Error analysis not available. Run `python error_analysis.py` to generate `error_analysis.csv`.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 8: THRESHOLD SENSITIVITY (Stage 3 risk-score sweep)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Stage 3 Threshold Sensitivity")
    st.caption("How precision, recall, and F1 change as the Stage 3 risk-score threshold moves. Demonstrates the framework supports multiple operating points by adjusting a single parameter.")

    thr_path = "threshold_sensitivity.csv"
    if os.path.exists(thr_path):
        thr_df = pd.read_csv(thr_path)

        # ── Row: PR-F1 curve + alert volume ───────────────────────────
        thr_c1, thr_c2 = st.columns(2)

        with thr_c1:
            st.markdown("**18. Precision / Recall / F1 across Thresholds**")
            st.caption("As the threshold rises, precision climbs but recall falls. The crossover region holds the F1-optimal operating point.")
            fig_thr = go.Figure()
            fig_thr.add_trace(go.Scatter(
                x=thr_df["threshold"], y=thr_df["precision"] * 100,
                mode="lines+markers", name="Precision",
                line=dict(color="#1976d2", width=3), marker=dict(size=8),
            ))
            fig_thr.add_trace(go.Scatter(
                x=thr_df["threshold"], y=thr_df["recall"] * 100,
                mode="lines+markers", name="Recall",
                line=dict(color="#388e3c", width=3), marker=dict(size=8, symbol="square"),
            ))
            fig_thr.add_trace(go.Scatter(
                x=thr_df["threshold"], y=thr_df["f1"] * 100,
                mode="lines+markers", name="F1 Score",
                line=dict(color="#f57c00", width=3), marker=dict(size=9, symbol="triangle-up"),
            ))
            # Mark F1-optimal point
            best_f1_row = thr_df.loc[thr_df["f1"].idxmax()]
            fig_thr.add_vline(
                x=best_f1_row["threshold"], line_dash="dash", line_color="#7C3AED",
                annotation_text=f"F1-optimal (t={int(best_f1_row['threshold'])})",
                annotation_position="top right",
            )
            fig_thr.update_layout(
                height=420, margin=dict(t=10, b=40, l=10, r=10),
                xaxis_title="Stage 3 Risk Score Threshold",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 110]),
                legend=dict(orientation="h", y=-0.15),
                hovermode="x unified",
            )
            st.plotly_chart(fig_thr, use_container_width=True)

        with thr_c2:
            st.markdown("**19. Alert Volume by Threshold**")
            st.caption("How many messages trigger alerts at each threshold. Lower thresholds = more alerts (more recall); higher thresholds = fewer alerts (more precision).")
            fig_vol = go.Figure(data=[go.Bar(
                x=thr_df["threshold"], y=thr_df["alerts"],
                marker_color="#8B5CF6",
                text=thr_df["alerts"],
                textposition="outside",
            )])
            fig_vol.update_layout(
                height=420, margin=dict(t=10, b=40, l=10, r=10),
                xaxis_title="Stage 3 Risk Score Threshold",
                yaxis_title="Number of Alerts Raised",
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # ── Operating points table ─────────────────────────────────────
        st.markdown("**20. Recommended Operating Points**")
        st.caption("Different deployment scenarios call for different thresholds. The cascade supports all of them by adjusting a single parameter.")

        op_points = []
        # Recall-optimised (regulatory screening)
        r_max = thr_df.loc[thr_df["recall"].idxmax()]
        op_points.append({
            "Operating Point":  "Recall-optimised",
            "Use Case":         "Regulatory first-pass screening",
            "Threshold":        int(r_max["threshold"]),
            "Precision":        f"{r_max['precision']*100:.2f}%",
            "Recall":           f"{r_max['recall']*100:.2f}%",
            "F1":               f"{r_max['f1']*100:.2f}%",
        })
        # F1-optimal
        op_points.append({
            "Operating Point":  "F1-optimal (balanced)",
            "Use Case":         "Default operational alerting",
            "Threshold":        int(best_f1_row["threshold"]),
            "Precision":        f"{best_f1_row['precision']*100:.2f}%",
            "Recall":           f"{best_f1_row['recall']*100:.2f}%",
            "F1":               f"{best_f1_row['f1']*100:.2f}%",
        })
        # Precision-optimised
        precision_filter = thr_df[thr_df["recall"] > 0.3]
        if not precision_filter.empty:
            p_max = precision_filter.loc[precision_filter["precision"].idxmax()]
            op_points.append({
                "Operating Point":  "Precision-optimised",
                "Use Case":         "Consumer-facing alerts",
                "Threshold":        int(p_max["threshold"]),
                "Precision":        f"{p_max['precision']*100:.2f}%",
                "Recall":           f"{p_max['recall']*100:.2f}%",
                "F1":               f"{p_max['f1']*100:.2f}%",
            })
        # Strict precision
        strict = thr_df[thr_df["precision"] == 1.0]
        if not strict.empty:
            s_row = strict.iloc[0]  # lowest threshold achieving 100% precision
            op_points.append({
                "Operating Point":  "Strict precision (100%)",
                "Use Case":         "High-stakes regulatory action",
                "Threshold":        int(s_row["threshold"]),
                "Precision":        f"{s_row['precision']*100:.2f}%",
                "Recall":           f"{s_row['recall']*100:.2f}%",
                "F1":               f"{s_row['f1']*100:.2f}%",
            })

        st.dataframe(pd.DataFrame(op_points), use_container_width=True, hide_index=True)

        # ── Honest finding callout ────────────────────────────────────
        default_threshold = 3
        default_row = thr_df[thr_df["threshold"] == default_threshold]
        if not default_row.empty:
            default_f1 = default_row.iloc[0]["f1"]
            best_f1_val = best_f1_row["f1"]
            best_t = int(best_f1_row["threshold"])
            if best_t != default_threshold and best_f1_val > default_f1:
                st.warning(
                    f"💡 **Empirical finding:** The default cascade threshold of {default_threshold} (F1: {default_f1*100:.2f}%) "
                    f"is not the F1-optimal choice. Threshold {best_t} produces a higher F1 of {best_f1_val*100:.2f}% on the curated test set. "
                    "This is a documented finding to be addressed in production by either re-tuning the default or expanding the test set."
                )
    else:
        st.info("ℹ️ Threshold sensitivity data not available. Run `python threshold_analysis.py` to generate `threshold_sensitivity.csv`.")

    st.markdown("---")

    st.caption(
        "**Methodology note:** Primary classification metrics computed on a stratified held-out test set (n=3,002) "
        "that the model never saw during training. Per-language results, error analysis, and threshold sensitivity computed on a hand-crafted "
        "curated multilingual test set (n=60) covering English, Nigerian Pidgin, Hindi, and Indonesian. "
        "Latency measured on 200 sampled messages on CPU. All evaluations are documented in Chapter 4 of the thesis."
    )
