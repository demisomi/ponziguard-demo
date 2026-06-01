"""
PonziGuard — Public Demo (Streamlit Cloud deployment)
Lightweight version showing the Check-a-Message tool and key results.
Does not require the trained DistilBERT model.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="PonziGuard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
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

st.title("🛡️ PonziGuard")
st.caption("A generalizable, real-time detector of social-media Ponzi schemes — Prototype Demo")
st.markdown("")

tab_check, tab_results, tab_about = st.tabs([
    "🛡️ Check a Message", "📊 Evaluation Results", "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: CHECK A MESSAGE
# ═══════════════════════════════════════════════════════════════════════
with tab_check:
    st.subheader("Is this message a Ponzi scheme?")
    st.caption("Paste any suspicious message you received and PonziGuard will check it for known Ponzi patterns.")
    st.markdown("")

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

    PUBLIC_RULES = [
        {"name": "Guaranteed Returns", "weight": 5, "plain": "Promises money is 100% safe — real investments cannot guarantee returns",
         "patterns": ["guaranteed return", "guaranteed profit", "guaranteed payout", "100% guaranteed", "100% safe", "risk free investment", "risk-free investment", "no risk", "zero risk", "assured returns"]},
        {"name": "Unrealistic Promises", "weight": 5, "plain": "Promises returns that are impossible (200%, doubled money, 10x)",
         "patterns": ["200%", "300%", "400%", "500%", "1000%", "double your money", "triple your money", "multiply your", "10x returns", "100x"]},
        {"name": "Daily/Weekly Profits", "weight": 5, "plain": "Promises daily or weekly payouts — classic Ponzi signature",
         "patterns": ["daily profit", "daily income", "daily return", "weekly profit", "weekly income", "weekly return", "earn daily", "paid daily"]},
        {"name": "Pyramid Structure", "weight": 5, "plain": "Uses pyramid or MLM language — money comes from new members, not real returns",
         "patterns": ["pyramid", "ponzi", "mlm", "matrix", "downline", "upline", "multi-level"]},
        {"name": "Known Ponzi Schemes", "weight": 5, "plain": "Mentions a known fraudulent scheme",
         "patterns": ["mmm", "loom money", "cbex", "racksterli", "racksterly", "mba forex", "twinkas", "onecoin", "bitconnect"]},
        {"name": "Recruitment Pay", "weight": 5, "plain": "Pays you to recruit others — Ponzi mechanic, not real business",
         "patterns": ["referral commission", "referral bonus", "refer and earn", "recruit and earn", "downline commission", "binary plan"]},
        {"name": "VIP / Tiered Packages", "weight": 4, "plain": "Sells investment packages by tier (Gold, Diamond, VIP)",
         "patterns": ["starter package", "bronze package", "gold package", "platinum package", "diamond package", "vip package", "premium package"]},
        {"name": "Passive Income Hype", "weight": 4, "plain": "Sells the dream of money without work",
         "patterns": ["passive income", "earn while you sleep", "no work required", "financial freedom", "quit your job", "retire early"]},
        {"name": "Urgency / Limited Slots", "weight": 2, "plain": "Pressures you to act fast — fraud loves urgency",
         "patterns": ["limited slots", "limited offer", "act now", "join now", "hurry", "closing soon", "last chance"]},
        {"name": "Trading Bots", "weight": 2, "plain": "Mentions automated trading bots — common cover for Ponzi",
         "patterns": ["trading bot", "ai trading", "forex signals", "crypto signals", "mining bot"]},
        {"name": "Wallet Transfers", "weight": 1, "plain": "Asks you to send money to a wallet or account",
         "patterns": ["wallet address", "send to wallet", "send funds", "pay to", "transfer to"]},
        {"name": "Fake Proof", "weight": 1, "plain": "Shows testimonials or proof of payment — easily fabricated",
         "patterns": ["proof of payment", "screenshot", "testimonial", "i just received", "i just withdrew", "100% legit"]},
    ]

    def is_investment_message(text):
        t = str(text).lower()
        return any(v in t for v in INVESTMENT_VOCAB)

    def check_message(text):
        t = str(text).lower()
        if not is_investment_message(t):
            return 0, "NOT INVESTMENT", [], False

        triggered = []
        score = 0
        for rule in PUBLIC_RULES:
            matched = [p for p in rule["patterns"] if p in t]
            if matched:
                triggered.append({"name": rule["name"], "weight": rule["weight"], "matched": matched, "plain": rule["plain"]})
                score += rule["weight"]

        if score >= 15: level = "CRITICAL"
        elif score >= 10: level = "HIGH"
        elif score >= 5: level = "MEDIUM"
        elif score >= 1: level = "LOW"
        else: level = "SAFE"
        return score, level, triggered, True

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

    selected_example = st.selectbox("Try an example or paste your own:", list(examples.keys()))
    default_text = examples[selected_example] if selected_example != "Pick an example to try…" else ""

    user_text = st.text_area(
        "Paste the suspicious message here:",
        value=default_text, height=160,
        placeholder="Example: 'Deposit ₦50,000 and earn ₦10,000 daily, guaranteed 100% safe...'",
        key=f"user_input_{selected_example}",
    )

    check_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)
    st.markdown("")

    if check_clicked and user_text.strip():
        score, level, triggered, is_investment = check_message(user_text)

        verdicts = {
            "CRITICAL": ("🚨 CRITICAL RISK", "#d32f2f", "white"),
            "HIGH": ("⚠️ HIGH RISK", "#f57c00", "white"),
            "MEDIUM": ("⚡ MEDIUM RISK", "#fbc02d", "black"),
            "LOW": ("🔵 LOW RISK", "#1976d2", "white"),
            "SAFE": ("✅ NO PONZI MARKERS", "#388e3c", "white"),
            "NOT INVESTMENT": ("ℹ️ NOT INVESTMENT-RELATED", "#616161", "white"),
        }
        label, bg, fg = verdicts[level]
        st.markdown(
            f"<div style='background:{bg};color:{fg};padding:24px;border-radius:12px;"
            f"text-align:center;font-size:1.6rem;font-weight:600;margin-bottom:16px'>"
            f"{label}<br><span style='font-size:1rem;font-weight:400'>Risk score: {score}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if level == "NOT INVESTMENT":
            st.info("This message does not appear to be about an investment offer. PonziGuard checks specifically for Ponzi scheme promotion. If you suspect another type of fraud (phishing, romance scam, lottery scam), report it to the appropriate authority.")
        elif level == "CRITICAL":
            st.error("**Do not invest. Do not send any money.** This message shows multiple patterns specific to Ponzi schemes. Block the sender. Report to your country's financial regulator (EFCC in Nigeria, SEBI in India, OJK in Indonesia).")
        elif level == "HIGH":
            st.error("**Treat as a likely Ponzi scheme.** Do not send money. Do not recruit friends or family. Verify the platform with your financial regulator before taking any action.")
        elif level == "MEDIUM":
            st.warning("**Be cautious.** Some Ponzi indicators are present. Research the platform independently. Search for it on regulator websites and recent news. If anyone pressures you to invest quickly, that is itself a warning sign.")
        elif level == "LOW":
            st.warning("**Minor red flags detected.** The message uses some language associated with investment fraud but lacks the strong markers of a confirmed Ponzi scheme. Stay cautious and verify independently.")
        elif level == "SAFE":
            st.success("**No Ponzi-specific patterns detected.** The message appears to be investment-related but contains none of the documented Ponzi scheme markers. Note: this does not guarantee the offer is legitimate. Always verify with your financial regulator before investing.")

        if triggered:
            st.markdown("---")
            st.markdown("### 🚩 Red flags detected")
            for rule in triggered:
                with st.expander(f"**{rule['name']}**  (weight: {rule['weight']})"):
                    st.markdown(f"*{rule['plain']}*")
                    st.markdown(f"**Matched phrases:** `{', '.join(rule['matched'])}`")

        st.markdown("---")
        st.markdown("### 📚 How to protect yourself")
        st.markdown(
            "- **No investment is risk-free.** Anyone who guarantees returns is lying.\n"
            "- **Real returns are slow.** 5%, 10%, even 15% per year is realistic. Daily or weekly profits are not.\n"
            "- **You cannot earn by recruiting.** If a platform pays you to bring in friends, the money is coming from those friends, not from real investment.\n"
            "- **Check regulators first.** EFCC (Nigeria), SEC Nigeria, SEBI (India), OJK (Indonesia) all publish lists of unregistered platforms.\n"
            "- **If unsure, walk away.** A legitimate opportunity will still be there tomorrow."
        )

    elif check_clicked:
        st.warning("Please paste a message to check.")
    else:
        st.info("👆 Paste any suspicious message above and click **Check Message** to see if it shows known Ponzi scheme patterns. You can also pick one of the example messages from the dropdown to see how it works.")

    st.markdown("---")
    st.caption("**Disclaimer:** PonziGuard is a research prototype. It checks messages against documented Ponzi scheme patterns but cannot guarantee detection of all fraud. Always verify investment offers with your country's financial regulator before sending any money. This tool does not provide financial or legal advice.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION RESULTS
# ═══════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Empirical Evaluation Results")
    st.caption("Results from the prototype implementation on a 15,010-message multilingual dataset.")

    st.markdown("### Headline Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage 2 F1", "97.96%", "+71.29 pp over zero-shot")
    c2.metric("Test Set Size", "3,002", "messages")
    c3.metric("Cascade Precision", "90.48%", "multilingual curated set")
    c4.metric("End-to-End Latency", "28.14 ms", "CPU")

    st.markdown("---")
    st.markdown("### Stage 2 Held-Out Test Set")
    cm_data = pd.DataFrame({
        "Predicted": ["Ponzi", "Ponzi", "Legitimate", "Legitimate"],
        "Actual": ["Ponzi", "Legitimate", "Ponzi", "Legitimate"],
        "Count": [1416, 31, 28, 1527],
        "Type": ["TP", "FP", "FN", "TN"],
    })
    fig_cm = px.bar(cm_data, x="Type", y="Count", color="Type",
                    color_discrete_map={"TP": "#388e3c", "TN": "#1976d2", "FP": "#f57c00", "FN": "#d32f2f"},
                    title="Stage 2 Confusion Matrix (n = 3,002)")
    fig_cm.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_cm, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Precision", "97.86%")
    col1.metric("Recall", "98.06%")
    col2.metric("F1 Score", "97.96%")
    col2.metric("Accuracy", "98.03%")

    st.markdown("---")
    st.markdown("### Per-Language Performance (Cascade)")
    lang_data = pd.DataFrame({
        "Language": ["English", "Pidgin", "Hindi", "Indonesian"],
        "Precision": [87.50, 100.00, 83.33, 100.00],
        "Recall": [63.64, 16.67, 83.33, 85.71],
        "F1": [73.68, 28.57, 83.33, 92.31],
    })
    fig_lang = go.Figure()
    fig_lang.add_trace(go.Bar(name="Precision", x=lang_data["Language"], y=lang_data["Precision"], marker_color="#1976d2"))
    fig_lang.add_trace(go.Bar(name="Recall", x=lang_data["Language"], y=lang_data["Recall"], marker_color="#388e3c"))
    fig_lang.add_trace(go.Bar(name="F1", x=lang_data["Language"], y=lang_data["F1"], marker_color="#f57c00"))
    fig_lang.update_layout(barmode="group", height=380, yaxis_title="Score (%)", title="Cascade Per-Language Metrics")
    st.plotly_chart(fig_lang, use_container_width=True)

    st.markdown("---")
    st.markdown("### Latency Per Stage")
    lat_data = pd.DataFrame({
        "Stage": ["Stage 1\n(Keyword)", "Stage 2\n(DistilBERT)", "Stage 2.5\n(Sub-Filter)", "Stage 3\n(Rules)", "End-to-End\nCascade"],
        "Mean (ms)": [0.11, 51.10, 0.03, 0.09, 28.14],
        "p95 (ms)": [0.33, 58.03, 0.10, 0.24, 64.45],
    })
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Bar(name="Mean (ms)", x=lat_data["Stage"], y=lat_data["Mean (ms)"], marker_color="#1976d2"))
    fig_lat.add_trace(go.Bar(name="p95 (ms)", x=lat_data["Stage"], y=lat_data["p95 (ms)"], marker_color="#f57c00"))
    fig_lat.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100ms target (Chapter 3)")
    fig_lat.update_layout(barmode="group", height=400, yaxis_title="Latency (ms)", title="Inference Latency by Stage (n=200, CPU)")
    st.plotly_chart(fig_lat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ═══════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About PonziGuard")

    st.markdown("""
    **PonziGuard** is a Final Year Project at Pan-Atlantic University, Department of Computer
    and Information Sciences. It is a generalizable, real-time detector for social-media
    Ponzi schemes, designed for the Nigerian context and validated across four languages
    (English, Nigerian Pidgin, Hindi, Indonesian).

    ### Architecture

    The system uses a **three-stage cascade**:

    1. **Stage 1 — Keyword Filter:** Catches anything suspicious using a 100-term lexicon.
    2. **Stage 2 — DistilBERT Classifier:** Fine-tuned multilingual transformer separating
       Ponzi promotion from legitimate financial content.
    3. **Stage 2.5 — Investment Sub-Filter:** Dismisses non-investment fraud (phishing,
       lottery, romance scams) before scoring.
    4. **Stage 3 — Rule Reasoner:** 12 weighted rule categories produce an explainable
       Ponzi-specific risk score.

    ### Key Numbers

    - **15,010 messages** in combined dataset (3 HuggingFace sources + 401 hand-crafted synthetic)
    - **97.96% F1** on held-out test set of 3,002 messages
    - **90.48% precision** on curated multilingual test set
    - **28.14 ms** mean end-to-end latency on CPU
    - **47.3%** of non-investment fraud dismissed by Stage 2.5 sub-filter

    """)