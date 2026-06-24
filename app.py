import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main { background: #F5F7FA; }
.block-container { padding: 3rem 2rem 2rem 2rem !important; max-width: 1440px; }

/* HEADER */
.hero {
    background: linear-gradient(135deg, #0F2167 0%, #1A3A8F 40%, #0E6BA8 100%);
    border-radius: 20px;
    padding: 30px 40px 26px 40px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(15,33,103,0.22);
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -30px;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; right: 120px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 30px; font-weight: 800;
    color: #fff; margin: 0 0 5px 0;
}
.hero-sub { font-size: 13px; color: rgba(255,255,255,0.72); margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px; padding: 3px 13px;
    font-size: 11px; color: #fff; font-weight: 600;
    margin-right: 7px; margin-top: 11px; letter-spacing: 0.3px;
}

/* SECTION LABELS */
.sec-label {
    font-size: 11px; font-weight: 700; color: #0F2167;
    letter-spacing: 1.2px; text-transform: uppercase;
    border-left: 3px solid #0E6BA8;
    padding-left: 9px; margin: 16px 0 11px 0;
}

/* INFO BOXES */
.info-box {
    background: #EEF4FB; border-radius: 10px;
    padding: 12px 15px; border-left: 3px solid #0E6BA8;
    font-size: 13px; color: #1e3a5f; margin: 10px 0; line-height: 1.6;
}
.warn-box {
    background: #FFF8E8; border-radius: 10px;
    padding: 12px 15px; border-left: 3px solid #D97706;
    font-size: 13px; color: #78350F; margin: 10px 0;
}

/* RESULT BANNERS */
.result-churn {
    background: linear-gradient(135deg, #B91C1C, #DC2626);
    border-radius: 14px; padding: 16px 22px;
    text-align: center; font-size: 17px; font-weight: 700;
    color: #fff; margin: 12px 0;
    box-shadow: 0 4px 18px rgba(185,28,28,0.28);
}
.result-stay {
    background: linear-gradient(135deg, #065F46, #059669);
    border-radius: 14px; padding: 16px 22px;
    text-align: center; font-size: 17px; font-weight: 700;
    color: #fff; margin: 12px 0;
    box-shadow: 0 4px 18px rgba(6,95,70,0.28);
}

/* SHAP ITEMS */
.shap-row {
    display: flex; align-items: center;
    background: #F8FAFC; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 7px;
    border: 1px solid #E2E8F0;
}
.shap-rank { font-size: 13px; font-weight: 700; color: #0F2167; width: 22px; }
.shap-name { font-size: 13px; font-weight: 600; color: #1E293B; flex: 1; }
.shap-val  { font-size: 11.5px; color: #64748B; font-family: monospace; }

/* EMPTY */
.empty-state {
    background: #F8FAFC; border-radius: 16px;
    padding: 48px 28px; text-align: center;
    border: 2px dashed #CBD5E1;
}

/* INSIGHT CARDS */
.insight-card {
    border-radius: 13px; padding: 16px 18px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    height: 100%;
}

/* STREAMLIT OVERRIDES */
div[data-testid="stMetric"] {
    background: #fff; border-radius: 12px;
    padding: 14px 16px; border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
div[data-testid="stMetricLabel"] p  { color: #64748B !important; font-size: 12px !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"]    { color: #0F2167 !important; font-size: 24px !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"]    { font-size: 12px !important; }

.stButton > button {
    background: linear-gradient(135deg, #0F2167, #0E6BA8) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 14px !important; padding: 12px 20px !important;
    box-shadow: 0 4px 14px rgba(14,107,168,0.30) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

.stSelectbox label, .stNumberInput label, .stSlider label {
    font-size: 12.5px !important; font-weight: 600 !important; color: #374151 !important;
}
div[data-baseweb="select"] > div {
    border-radius: 9px !important; border-color: #E2E8F0 !important; background: #fff !important;
}
div[data-testid="stNumberInput"] input { border-radius: 9px !important; border-color: #E2E8F0 !important; }

button[data-baseweb="tab"] { font-size: 13px !important; font-weight: 600 !important; color: #64748B !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0F2167 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Resources ──
@st.cache_resource
def load_model():
    return joblib.load("lgb_smote.pkl")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
        return df
    except:
        return None

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model    = load_model()
df_raw   = load_data()
explainer = get_explainer(model)

FEATURE_COLS = [
    'gender','SeniorCitizen','Partner','Dependents','tenure',
    'PhoneService','MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_DSL','InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No','OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No','OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No','DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No','TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No','StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No','StreamingMovies_No internet service','StreamingMovies_Yes',
    'Contract_Month-to-month','Contract_One year','Contract_Two year',
    'PaperlessBilling','PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check','MonthlyCharges','TotalCharges'
]

def build_row(gender, senior, partner, dependents, tenure, phone,
              multiple_lines, internet, online_sec, online_backup,
              device_prot, tech_support, streaming_tv, streaming_movies,
              contract, paperless, payment, monthly, total):
    row = {col: 0 for col in FEATURE_COLS}
    row['gender']          = 1 if gender == "Male" else 0
    row['SeniorCitizen']   = 1 if senior == "Yes" else 0
    row['Partner']         = 1 if partner == "Yes" else 0
    row['Dependents']      = 1 if dependents == "Yes" else 0
    row['tenure']          = tenure
    row['PhoneService']    = 1 if phone == "Yes" else 0
    row['PaperlessBilling']= 1 if paperless == "Yes" else 0
    row['MonthlyCharges']  = monthly
    row['TotalCharges']    = total
    row[f'MultipleLines_{multiple_lines}']    = 1
    row[f'InternetService_{internet}']        = 1
    row[f'OnlineSecurity_{online_sec}']       = 1
    row[f'OnlineBackup_{online_backup}']      = 1
    row[f'DeviceProtection_{device_prot}']    = 1
    row[f'TechSupport_{tech_support}']        = 1
    row[f'StreamingTV_{streaming_tv}']        = 1
    row[f'StreamingMovies_{streaming_movies}']= 1
    row[f'Contract_{contract}']               = 1
    row[f'PaymentMethod_{payment}']           = 1
    return pd.DataFrame([row])

def make_gauge(val):
    if val >= 60:   color, label = "#DC2626", "High Risk"
    elif val >= 35: color, label = "#D97706", "Medium Risk"
    else:           color, label = "#059669", "Low Risk"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix":"%","font":{"size":40,"color":"#0F2167","family":"Sora"}},
        title={"text":f"<b>{label}</b>","font":{"size":13,"color":color,"family":"Plus Jakarta Sans"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#94A3B8","tickfont":{"color":"#64748B","size":10}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"#F8FAFC","borderwidth":0,
            "steps":[
                {"range":[0,35],"color":"#DCFCE7"},
                {"range":[35,60],"color":"#FEF3C7"},
                {"range":[60,100],"color":"#FEE2E2"},
            ],
            "threshold":{"line":{"color":"#0F2167","width":3},"thickness":0.85,"value":val}
        }
    ))
    fig.update_layout(
        height=210, margin=dict(t=30,b=0,l=35,r=35),
        paper_bgcolor="#fff", plot_bgcolor="#fff"
    )
    return fig

# CHART HELPERS
CHART_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Plus Jakarta Sans", color="#1E293B"),
    margin=dict(t=15, b=15, l=15, r=15)
)

# ── HEADER ──
st.markdown("""
<div class="hero">
  <p class="hero-title">📡 Telco Customer Churn Prediction</p>
  <p class="hero-sub">LightGBM + SMOTE · Explainable ML · SHAP Analysis · Telco Dataset</p>
  <span class="hero-badge">🎓 Final Project ML</span>
  <span class="hero-badge">📊 Benchmark 12 Models</span>
  <span class="hero-badge">🔬 SHAP Explainability</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Churn Prediction",
    "📊  Model Comparison",
    "📈  Data Insights",
    "💡  Business Insights",
    "ℹ️  About",
])

# ══════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1.05, 1], gap="large")

    with col_in:
        st.markdown('<div class="sec-label">Customer Profile</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            gender      = st.selectbox("Gender", ["Male","Female"])
            senior      = st.selectbox("Senior Citizen", ["No","Yes"])
            partner     = st.selectbox("Partner", ["Yes","No"])
            dependents  = st.selectbox("Dependents", ["No","Yes"])
            tenure      = st.slider("Tenure (months)", 0, 72, 12)
        with r1c2:
            contract    = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            payment     = st.selectbox("Payment Method", [
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ])
            paperless   = st.selectbox("Paperless Billing", ["Yes","No"])
            monthly     = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total       = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly*tenure) if tenure > 0 else 65.0, step=1.0)

        st.markdown('<div class="sec-label">Services</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            phone       = st.selectbox("Phone Service", ["Yes","No"])
            multiple    = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
            internet    = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        with s2:
            online_sec  = st.selectbox("Online Security", ["No","Yes","No internet service"])
            online_bak  = st.selectbox("Online Backup", ["No","Yes","No internet service"])
            device_prot = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        with s3:
            tech_sup    = st.selectbox("Tech Support", ["No","Yes","No internet service"])
            stream_tv   = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
            stream_mv   = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

        predict_btn = st.button("🔍  Predict Churn Risk", use_container_width=True)

    with col_out:
        st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            df_pred    = build_row(gender, senior, partner, dependents, tenure, phone,
                                   multiple, internet, online_sec, online_bak,
                                   device_prot, tech_sup, stream_tv, stream_mv,
                                   contract, paperless, payment, monthly, total)
            prob       = model.predict_proba(df_pred)[0]
            stay_prob  = prob[0] * 100
            churn_prob = prob[1] * 100
            prediction = model.predict(df_pred)[0]

            st.plotly_chart(make_gauge(churn_prob), use_container_width=True, key="gauge")

            m1, m2 = st.columns(2)
            m1.metric("✅ Stay Probability",  f"{stay_prob:.1f}%")
            m2.metric("⚠️ Churn Probability", f"{churn_prob:.1f}%")

            if prediction == 1:
                risk = "High" if churn_prob >= 60 else "Medium"
                st.markdown(f'<div class="result-churn">⚠️ Customer Likely to Churn &nbsp;·&nbsp; {risk} Risk ({churn_prob:.1f}%)</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-stay">✅ Customer Likely to Stay &nbsp;·&nbsp; {stay_prob:.1f}% Confidence</div>',
                            unsafe_allow_html=True)

            # SHAP
            st.markdown('<div class="sec-label">Why This Prediction? (SHAP)</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">SHAP menjelaskan faktor mana yang mendorong atau menurunkan risiko churn pelanggan ini. <b>Merah</b> = meningkatkan risiko churn · <b>Biru</b> = menurunkan risiko.</div>',
                        unsafe_allow_html=True)

            shap_out = explainer(df_pred)
            sv = shap_out.values
            if sv.ndim == 3:
                sv = sv[:,:,1]
            sv = sv[0]
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

            explanation = shap.Explanation(
                values=sv, base_values=base_val,
                data=df_pred.iloc[0].values,
                feature_names=FEATURE_COLS
            )
            fig_shap, ax = plt.subplots(figsize=(7, 4.2))
            fig_shap.patch.set_facecolor("#fff")
            ax.set_facecolor("#fff")
            shap.waterfall_plot(explanation, max_display=10, show=False)
            for txt in ax.texts: txt.set_color("#1E293B")
            ax.tick_params(colors="#1E293B")
            ax.xaxis.label.set_color("#1E293B")
            for sp in ax.spines.values(): sp.set_edgecolor("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

            # Top 3 factors
            st.markdown('<div class="sec-label">Top Risk Factors</div>', unsafe_allow_html=True)
            top_idx = np.argsort(np.abs(sv))[::-1][:3]
            for i, idx in enumerate(top_idx):
                direction = "🔴 Increases churn risk" if sv[idx] > 0 else "🟢 Reduces churn risk"
                dir_color = "#DC2626" if sv[idx] > 0 else "#059669"
                st.markdown(f"""
                <div class="shap-row">
                  <span class="shap-rank">{i+1}.</span>
                  <span class="shap-name">{FEATURE_COLS[idx]}</span>
                  <span class="shap-val" style="color:{dir_color};font-weight:600;">{direction}</span>
                  <span class="shap-val">&nbsp;({sv[idx]:+.3f})</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:50px;margin-bottom:12px;">📡</div>
              <div style="font-size:15px;font-weight:700;color:#1E293B;margin-bottom:5px;">Ready to Analyze</div>
              <div style="font-size:13px;color:#64748B;">Isi profil pelanggan di sebelah kiri,<br>lalu klik <b>Predict Churn Risk</b></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">Benchmark Results — 12 Model Combinations</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Hasil benchmark <b>4 model × 3 teknik imbalance</b> (None, SMOTE, class_weight) pada test set 20% (1.409 sampel).
    <b>PR-AUC</b> adalah metrik utama. Baseline random = 0.2654.
    </div>""", unsafe_allow_html=True)

    bench_data = {
        "Model": ["LGB — SMOTE ⭐","LGB — cw","LGB — None","XGB — None","LR — None",
                  "LR — cw","LR — SMOTE","XGB — cw","XGB — SMOTE","RF — cw","RF — SMOTE","RF — None"],
        "PR-AUC":    [0.6421,0.6393,0.6365,0.6313,0.6311,0.6302,0.6281,0.6214,0.6162,0.6098,0.6061,0.6017],
        "ROC-AUC":   [0.8312,0.8329,0.8304,0.8254,0.8416,0.8411,0.8402,0.8187,0.8216,0.8229,0.8208,0.8179],
        "Accuracy":  [0.7821,0.7573,0.7857,0.7850,0.8070,0.7395,0.7353,0.7523,0.7821,0.7835,0.7764,0.7850],
        "F1":        [0.5834,0.6192,0.5610,0.5726,0.6092,0.6149,0.6151,0.5889,0.5834,0.5455,0.5714,0.5628],
        "Precision": [0.5923,0.5305,0.6146,0.6060,0.6584,0.5060,0.5008,0.5263,0.5923,0.6162,0.5817,0.6113],
        "Recall":    [0.5749,0.7433,0.5160,0.5428,0.5668,0.7834,0.7968,0.6684,0.5749,0.4893,0.5615,0.5214],
    }
    df_bench = pd.DataFrame(bench_data)

    def highlight_best(row):
        if "⭐" in str(row["Model"]):
            return ["background-color:#DBEAFE;font-weight:700;color:#1E3A8A"] * len(row)
        return ["color:#1E293B"] * len(row)

    styled = (df_bench.style
        .apply(highlight_best, axis=1)
        .format({k:"{:.4f}" for k in ["PR-AUC","ROC-AUC","Accuracy","F1","Precision","Recall"]})
        .bar(subset=["PR-AUC"], color=["#FEE2E2","#DBEAFE"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)
    st.caption("⭐ = Model terbaik yang digunakan untuk prediksi.")

    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="sec-label">PR-AUC per Model</div>', unsafe_allow_html=True)
        colors = ["#0F2167" if "⭐" in m else "#93C5FD" for m in df_bench["Model"]]
        fig_bar = go.Figure(go.Bar(
            x=df_bench["PR-AUC"],
            y=[m.replace(" ⭐","") for m in df_bench["Model"]],
            orientation="h", marker_color=colors, marker_line_width=0,
            text=[f"{v:.4f}" for v in df_bench["PR-AUC"]],
            textposition="outside",
            textfont=dict(color="#1E293B", size=10.5, family="Plus Jakarta Sans")
        ))
        fig_bar.add_vline(x=0.2654, line_dash="dash", line_color="#94A3B8",
                          annotation_text="Baseline", annotation_font_color="#64748B",
                          annotation_font_size=10)
        fig_bar.update_layout(
            height=420, xaxis=dict(color="#64748B", range=[0.55, 0.68], gridcolor="#F1F5F9"),
            yaxis=dict(color="#374151", autorange="reversed"),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_bench")

    with c2:
        st.markdown('<div class="sec-label">PR-AUC vs ROC-AUC Scatter</div>', unsafe_allow_html=True)
        model_base = [m.split(" — ")[0].replace(" ⭐","") for m in df_bench["Model"]]
        colors_map = {"LGB":"#0F2167","XGB":"#0E6BA8","LR":"#10B981","RF":"#F59E0B"}
        dot_colors = [colors_map.get(m, "#94A3B8") for m in model_base]

        fig_sc = go.Figure()
        for base, color in colors_map.items():
            mask = [m == base for m in model_base]
            fig_sc.add_trace(go.Scatter(
                x=[v for v, m in zip(df_bench["ROC-AUC"], mask) if m],
                y=[v for v, m in zip(df_bench["PR-AUC"], mask) if m],
                mode="markers+text",
                name=base,
                marker=dict(color=color, size=11, line=dict(color="#fff", width=1.5)),
                text=[m.split("—")[1].strip().replace(" ⭐","") for m, mm in zip(df_bench["Model"], mask) if mm],
                textposition="top center",
                textfont=dict(size=9, color=color)
            ))
        fig_sc.update_layout(
            height=420,
            xaxis=dict(color="#64748B", title="ROC-AUC", gridcolor="#F1F5F9"),
            yaxis=dict(color="#64748B", title="PR-AUC", gridcolor="#F1F5F9"),
            legend=dict(font=dict(size=11, color="#374151"), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#E2E8F0", borderwidth=1),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="scatter_bench")

    # Kenapa PR-AUC
    st.divider()
    st.markdown('<div class="sec-label">Kenapa PR-AUC Sebagai Metrik Utama?</div>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    for col, title, body, col_txt, bg in [
        (ec1,"⚠️ Accuracy","Menyesatkan di data imbalanced 73:27. Model yang selalu prediksi 'tidak churn' bisa dapat accuracy 73% tanpa berguna.","#B45309","#FFFBEB"),
        (ec2,"⚠️ ROC-AUC","True Negative ikut dihitung. Skor bisa tampak bagus (0.83+) meski deteksi churn masih lemah.","#B45309","#FFFBEB"),
        (ec3,"✅ PR-AUC","Hanya Precision & Recall — fokus murni pada kemampuan mendeteksi pelanggan yang benar-benar akan churn.","#065F46","#F0FDF4"),
    ]:
        col.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:15px 17px;border:1px solid #E2E8F0;">
          <div style="font-size:13px;font-weight:700;color:{col_txt};margin-bottom:7px;">{title}</div>
          <div style="font-size:12.5px;color:#374151;line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">Dataset Overview</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.markdown('<div class="warn-box">⚠️ File <code>WA_Fn-UseC_-Telco-Customer-Churn.csv</code> tidak ditemukan.</div>',
                    unsafe_allow_html=True)
    else:
        total  = len(df_raw)
        n_churn = df_raw["Churn_bin"].sum()
        n_stay  = total - n_churn

        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Total Customers", f"{total:,}")
        s2.metric("Churn", f"{n_churn:,} ({n_churn/total*100:.1f}%)")
        s3.metric("Stay", f"{n_stay:,} ({n_stay/total*100:.1f}%)")
        s4.metric("Avg Tenure", f"{df_raw['tenure'].mean():.0f} months")
        s5.metric("Avg Monthly", f"${df_raw['MonthlyCharges'].mean():.0f}")

        st.divider()

        # Row 1
        r1, r2 = st.columns(2, gap="large")
        with r1:
            st.markdown('<div class="sec-label">Churn Distribution</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=["Stay","Churn"],
                values=[n_stay, n_churn], hole=0.58,
                marker_colors=["#0E6BA8","#DC2626"],
                textinfo="label+percent",
                textfont=dict(color="#1E293B", size=12, family="Plus Jakarta Sans"),
            ))
            fig_pie.update_layout(height=250, showlegend=True,
                                  legend=dict(font=dict(color="#374151")), **CHART_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_churn")

        with r2:
            st.markdown('<div class="sec-label">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
            fig_ten = go.Figure()
            for label, color, name in [(0,"#0E6BA8","Stay"),(1,"#DC2626","Churn")]:
                fig_ten.add_trace(go.Histogram(
                    x=df_raw[df_raw["Churn_bin"]==label]["tenure"],
                    name=name, marker_color=color, opacity=0.72, nbinsx=24
                ))
            fig_ten.update_layout(
                barmode="overlay", height=250,
                xaxis=dict(color="#374151", title="Tenure (months)", gridcolor="#F1F5F9"),
                yaxis=dict(color="#374151", gridcolor="#F1F5F9"),
                legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)"),
                **CHART_LAYOUT
            )
            st.plotly_chart(fig_ten, use_container_width=True, key="tenure_hist")

        st.divider()

        # Numerik boxplot
        st.markdown('<div class="sec-label">Distribusi Fitur Numerik per Kelas</div>', unsafe_allow_html=True)
        num_feat = st.selectbox("Pilih fitur:", ["tenure","MonthlyCharges","TotalCharges"])
        fig_box = go.Figure()
        for label, color, name in [(0,"#0E6BA8","Stay"),(1,"#DC2626","Churn")]:
            fig_box.add_trace(go.Box(
                y=df_raw[df_raw["Churn_bin"]==label][num_feat],
                name=name, marker_color=color, boxmean=True, line_color=color
            ))
        fig_box.update_layout(
            height=300,
            yaxis=dict(color="#374151", title=num_feat, gridcolor="#F1F5F9"),
            xaxis=dict(color="#374151"),
            legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)"),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_box, use_container_width=True, key="boxplot")

        st.divider()

        # Kategorikal
        st.markdown('<div class="sec-label">Churn Rate per Kategori</div>', unsafe_allow_html=True)
        cat_feat = st.selectbox("Pilih fitur:", ["Contract","InternetService","PaymentMethod",
                                                  "gender","SeniorCitizen","Partner","Dependents",
                                                  "PhoneService","PaperlessBilling"])
        ct = pd.crosstab(df_raw[cat_feat], df_raw["Churn"], normalize="index") * 100
        ct = ct.reset_index()
        fig_cat = go.Figure()
        for col_name, color in [("No","#0E6BA8"),("Yes","#DC2626")]:
            if col_name in ct.columns:
                fig_cat.add_trace(go.Bar(
                    x=ct[cat_feat].astype(str), y=ct[col_name],
                    name="Stay" if col_name=="No" else "Churn",
                    marker_color=color, marker_line_width=0
                ))
        fig_cat.update_layout(
            barmode="group", height=300,
            yaxis=dict(color="#374151", title="%", gridcolor="#F1F5F9"),
            xaxis=dict(color="#374151"),
            legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)"),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="cat_bar")

        st.divider()

        # Monthly charges vs tenure scatter
        st.markdown('<div class="sec-label">Monthly Charges vs Tenure</div>', unsafe_allow_html=True)
        fig_sc2 = go.Figure()
        for label, color, name in [(0,"#0E6BA8","Stay"),(1,"#DC2626","Churn")]:
            sub = df_raw[df_raw["Churn_bin"]==label].sample(min(500, len(df_raw[df_raw["Churn_bin"]==label])), random_state=42)
            fig_sc2.add_trace(go.Scatter(
                x=sub["tenure"], y=sub["MonthlyCharges"],
                mode="markers", name=name,
                marker=dict(color=color, size=5, opacity=0.55, line=dict(color="#fff", width=0.3))
            ))
        fig_sc2.update_layout(
            height=320,
            xaxis=dict(color="#374151", title="Tenure (months)", gridcolor="#F1F5F9"),
            yaxis=dict(color="#374151", title="Monthly Charges ($)", gridcolor="#F1F5F9"),
            legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)"),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_sc2, use_container_width=True, key="scatter_eda")

# ══════════════════════════════════════════════════════
# TAB 4 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════
with tab4:

    if df_raw is None:
        st.markdown('<div class="warn-box">⚠️ Data tidak ditemukan.</div>', unsafe_allow_html=True)
    else:
        # Churn rate per contract
        cr_contract = df_raw.groupby("Contract")["Churn_bin"].mean() * 100
        cr_internet = df_raw.groupby("InternetService")["Churn_bin"].mean() * 100
        cr_tenure   = df_raw.groupby(pd.cut(df_raw["tenure"], bins=[0,12,24,48,72]))["Churn_bin"].mean() * 100

        b1, b2 = st.columns(2, gap="large")

        with b1:
            st.markdown('<div class="sec-label">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
            fig_ct = go.Figure(go.Bar(
                x=cr_contract.index, y=cr_contract.values,
                marker_color=["#DC2626","#F59E0B","#059669"],
                marker_line_width=0, text=[f"{v:.1f}%" for v in cr_contract.values],
                textposition="outside", textfont=dict(size=12, color="#1E293B", family="Plus Jakarta Sans")
            ))
            fig_ct.update_layout(
                height=280,
                yaxis=dict(color="#374151", title="Churn Rate (%)", gridcolor="#F1F5F9", range=[0,55]),
                xaxis=dict(color="#374151"),
                **CHART_LAYOUT
            )
            st.plotly_chart(fig_ct, use_container_width=True, key="biz_contract")
            st.markdown("""<div class="info-box">
            💡 Pelanggan <b>Month-to-month</b> churn ~42% — 3x lebih tinggi dari kontrak tahunan.
            Strategi retensi harus diprioritaskan untuk segmen ini.
            </div>""", unsafe_allow_html=True)

        with b2:
            st.markdown('<div class="sec-label">Churn Rate by Internet Service</div>', unsafe_allow_html=True)
            fig_int = go.Figure(go.Bar(
                x=cr_internet.index, y=cr_internet.values,
                marker_color=["#0E6BA8","#DC2626","#059669"],
                marker_line_width=0, text=[f"{v:.1f}%" for v in cr_internet.values],
                textposition="outside", textfont=dict(size=12, color="#1E293B", family="Plus Jakarta Sans")
            ))
            fig_int.update_layout(
                height=280,
                yaxis=dict(color="#374151", title="Churn Rate (%)", gridcolor="#F1F5F9", range=[0,50]),
                xaxis=dict(color="#374151"),
                **CHART_LAYOUT
            )
            st.plotly_chart(fig_int, use_container_width=True, key="biz_internet")
            st.markdown("""<div class="info-box">
            💡 Pengguna <b>Fiber Optic</b> churn lebih tinggi meski layanan premium — mengindikasikan
            ketidakpuasan pada kualitas atau harga layanan tersebut.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Churn rate by tenure group
        st.markdown('<div class="sec-label">Churn Rate by Tenure Group</div>', unsafe_allow_html=True)
        tenure_labels = ["0-12 bulan","13-24 bulan","25-48 bulan","49-72 bulan"]
        fig_ten2 = go.Figure(go.Bar(
            x=tenure_labels[:len(cr_tenure)], y=cr_tenure.values,
            marker_color=["#DC2626","#F59E0B","#0E6BA8","#059669"][:len(cr_tenure)],
            marker_line_width=0, text=[f"{v:.1f}%" for v in cr_tenure.values],
            textposition="outside", textfont=dict(size=12, color="#1E293B", family="Plus Jakarta Sans")
        ))
        fig_ten2.update_layout(
            height=280,
            yaxis=dict(color="#374151", title="Churn Rate (%)", gridcolor="#F1F5F9", range=[0,55]),
            xaxis=dict(color="#374151"),
            **CHART_LAYOUT
        )
        st.plotly_chart(fig_ten2, use_container_width=True, key="biz_tenure")
        st.markdown("""<div class="info-box">
        💡 Pelanggan dengan tenure <b>0-12 bulan pertama</b> memiliki churn rate tertinggi.
        Program onboarding dan retensi awal sangat kritis untuk mengurangi early churn.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Revenue at risk
        st.markdown('<div class="sec-label">Estimated Revenue at Risk</div>', unsafe_allow_html=True)
        churn_customers    = df_raw[df_raw["Churn_bin"]==1]
        monthly_at_risk    = churn_customers["MonthlyCharges"].sum()
        annual_at_risk     = monthly_at_risk * 12
        avg_churn_monthly  = churn_customers["MonthlyCharges"].mean()

        rv1, rv2, rv3 = st.columns(3)
        rv1.metric("Jumlah Pelanggan Churn", f"{len(churn_customers):,}")
        rv2.metric("Monthly Revenue at Risk", f"${monthly_at_risk:,.0f}")
        rv3.metric("Annual Revenue at Risk", f"${annual_at_risk:,.0f}")

# ══════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-label">Tentang Penelitian</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="large")

with a1:
    # Judul + Metodologi (keduanya di kiri)
    st.markdown("""
    <div style="background:#fff;border-radius:14px;padding:20px 22px;
                border:1px solid #E2E8F0;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:14px;">
      <div style="font-size:11px;font-weight:700;color:#64748B;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:10px;">Judul</div>
      <div style="font-size:14px;font-weight:700;color:#0F2167;line-height:1.6;">
        Prediksi Customer Churn pada Perusahaan Telekomunikasi Menggunakan
        Pendekatan Explainable Machine Learning: Studi Komparatif LightGBM
      </div>
    </div>
    <div style="background:#fff;border-radius:14px;padding:20px 22px;
                border:1px solid #E2E8F0;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
      <div style="font-size:11px;font-weight:700;color:#64748B;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:10px;">Metodologi</div>
      <div style="font-size:13.5px;color:#374151;line-height:2.0;">
        ✅ &nbsp;Benchmark 4 model × 3 teknik imbalance<br>
        ✅ &nbsp;Model final: LightGBM + SMOTE<br>
        ✅ &nbsp;Metrik: PR-AUC · ROC-AUC · F1 · Accuracy<br>
        ✅ &nbsp;Explainability: SHAP TreeExplainer
      </div>
    </div>
    """, unsafe_allow_html=True)

with a2:
    # Dataset + Model Terbaik (keduanya di kanan)
    st.markdown("""
    <div style="background:#fff;border-radius:14px;padding:20px 22px;
                border:1px solid #E2E8F0;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:14px;">
      <div style="font-size:11px;font-weight:700;color:#64748B;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:10px;">Dataset</div>
      <div style="font-size:13.5px;color:#374151;line-height:2.0;">
        📌 Telco Customer Churn Dataset (IBM/Kaggle)<br>
        📌 7.043 pelanggan · 20 fitur · 26.5% churn rate<br>
        📌 Domain: Telekomunikasi
      </div>
    </div>
    <div style="background:linear-gradient(135deg,#DBEAFE,#EFF6FF);border-radius:14px;
                padding:20px 22px;border:1px solid #BFDBFE;">
      <div style="font-size:11px;font-weight:700;color:#1E40AF;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:6px;">Model Terbaik</div>
      <div style="font-size:24px;font-weight:800;color:#0F2167;font-family:Sora,sans-serif;">
        ⭐ LightGBM + SMOTE
      </div>
      <div style="font-size:12.5px;color:#1E40AF;margin-top:5px;">
        PR-AUC: 0.6421 &nbsp;·&nbsp; ROC-AUC: 0.8312 &nbsp;·&nbsp; Accuracy: 78.2%
      </div>
    </div>
    """, unsafe_allow_html=True)
