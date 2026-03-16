import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Biomarker Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #f8f9fa; }
  .metric-card {
      background: white;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
      border: 1px solid #e0e0e0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
  .metric-label { font-size: 0.85rem; color: #6b6b78; text-transform: uppercase; letter-spacing: 0.05em; }
  .predict-benign {
      background: linear-gradient(135deg, #d4edda, #c3e6cb);
      border-radius: 12px; padding: 20px; text-align: center;
      border: 2px solid #28a745;
  }
  .predict-malignant {
      background: linear-gradient(135deg, #f8d7da, #f5c6cb);
      border-radius: 12px; padding: 20px; text-align: center;
      border: 2px solid #dc3545;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      border-radius: 8px 8px 0 0;
      padding: 8px 20px;
      font-weight: 500;
  }
  h1 { color: #1a1a2e !important; }
  .sidebar-info {
      background: #eef2ff;
      border-radius: 8px;
      padding: 12px;
      font-size: 0.85rem;
      color: #3c3489;
      margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── DATA & MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({1: 'Benign', 0: 'Malignant'})
    return df, data

@st.cache_resource
def train_models(df, data):
    X = df[data.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        trained[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_test': y_test,
            'cm': confusion_matrix(y_test, y_pred)
        }
    return trained, scaler, X_train_sc, X_test_sc, X_train, X_test, y_train, y_test

df, data = load_data()
trained_models, scaler, X_train_sc, X_test_sc, X_train, X_test, y_train, y_test = train_models(df, data)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Breast Cancer Biomarker Classifier")
st.markdown(
    "**Bhavya Sevak** · M.S. Biomedical Informatics, Arizona State University  \n"
    "Wisconsin Breast Cancer Dataset · 569 samples · 30 features"
)
st.divider()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="sidebar-info">Select a model and explore predictions, evaluation metrics, and SHAP explanations.</div>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Model",
        list(trained_models.keys()),
        index=1
    )

    st.divider()
    st.markdown("### 📊 Quick Stats")
    res = trained_models[selected_model]
    st.metric("Accuracy", f"{res['accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{res['auc']:.4f}")
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"🟢 Benign: {(df.target==1).sum()} samples")
    st.write(f"🔴 Malignant: {(df.target==0).sum()} samples")
    st.write(f"📐 Features: {len(data.feature_names)}")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔬 EDA",
    "📈 Model Evaluation",
    "🤖 SHAP Explainability",
    "🔍 Predict"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Model Performance Summary")

    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Logistic Regression", col1),
        ("Random Forest", col2),
        ("SVM", col3)
    ]
    for name, col in metrics:
        r = trained_models[name]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{r['accuracy']*100:.1f}%</div>
                <div class="metric-label">AUC: {r['auc']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### All Models — ROC Curves")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for (name, res), color in zip(trained_models.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontweight='bold')
    ax.legend()
    sns.despine()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['diagnosis'].value_counts()
        ax.pie(counts, labels=counts.index,
               colors=['#2ecc71', '#e74c3c'],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Benign vs Malignant', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Top Features by Mean Difference**")
        mean_diff = abs(
            df[df.target==1][data.feature_names].mean() -
            df[df.target==0][data.feature_names].mean()
        )
        top10 = mean_diff.nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top10.index, top10.values, color='#3498db')
        ax.set_title('Top 10 Discriminative Features', fontweight='bold')
        sns.despine()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Feature Boxplots — Select a feature:**")
    selected_feat = st.selectbox("Feature", list(data.feature_names))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='diagnosis', y=selected_feat,
                palette={'Benign': '#2ecc71', 'Malignant': '#e74c3c'}, ax=ax)
    ax.set_title(f'{selected_feat} by Diagnosis', fontweight='bold')
    sns.despine()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Correlation Heatmap (mean features)**")
    mean_features = [f for f in data.feature_names if 'mean' in f]
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[mean_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"### {selected_model} — Evaluation")
    res = trained_models[selected_model]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Malignant','Benign'],
                    yticklabels=['Malignant','Benign'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Classification Report**")
        report = classification_report(
            res['y_test'], res['y_pred'],
            target_names=['Malignant', 'Benign'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    if selected_model == 'Random Forest':
        st.markdown("**Feature Importance (Random Forest)**")
        rf = trained_models['Random Forest']['model']
        importances = pd.Series(rf.feature_importances_, index=data.feature_names)
        top15 = importances.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        colors_bar = ['#e74c3c' if i >= 10 else '#3498db' for i in range(len(top15))]
        ax.barh(top15.index, top15.values, color=colors_bar)
        ax.set_title('Top 15 Biomarker Features', fontweight='bold')
        ax.set_xlabel('Gini Importance')
        sns.despine()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SHAP
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🤖 SHAP Explainability")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows **why** the model made each prediction — "
        "which features pushed the result toward malignant or benign."
    )

    shap_model_name = st.radio(
        "Select model for SHAP analysis:",
        ["Logistic Regression", "Random Forest"],
        horizontal=True
    )

    with st.spinner("Computing SHAP values..."):
        model_for_shap = trained_models[shap_model_name]['model']
        X_test_sc_df = pd.DataFrame(X_test_sc, columns=data.feature_names)

        if shap_model_name == "Random Forest":
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(X_test_sc_df)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer = shap.LinearExplainer(model_for_shap, X_train_sc)
            shap_values = explainer.shap_values(X_test_sc_df)
            sv = shap_values

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SHAP Summary Plot — Feature Impact**")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(sv, X_test_sc_df, show=False, plot_size=None)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**SHAP Bar Plot — Mean |SHAP| per Feature**")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(sv, X_test_sc_df, plot_type='bar', show=False, plot_size=None)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Individual Prediction Explanation**")
    sample_idx = st.slider("Select a test sample to explain:", 0, len(X_test_sc)-1, 0)

    sample = X_test_sc_df.iloc[[sample_idx]]
    true_label = "Benign" if y_test.iloc[sample_idx] == 1 else "Malignant"
    pred_label = "Benign" if trained_models[shap_model_name]['y_pred'][sample_idx] == 1 else "Malignant"

    col1, col2 = st.columns(2)
    col1.metric("True Diagnosis", true_label)
    col2.metric("Model Prediction", pred_label,
                delta="✓ Correct" if true_label == pred_label else "✗ Incorrect")

    st.markdown("**SHAP Waterfall — Why this prediction?**")
    if shap_model_name == "Random Forest":
        exp = shap.Explanation(
            values=sv[sample_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=sample.values[0],
            feature_names=list(data.feature_names)
        )
    else:
        exp = shap.Explanation(
            values=sv[sample_idx],
            base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
            data=sample.values[0],
            feature_names=list(data.feature_names)
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(exp, show=False, max_display=12)
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔍 Live Prediction")
    st.markdown("Adjust the sliders to input tumor measurements and get a real-time prediction.")

    # Use mean values as defaults
    benign_means = df[df.target==1][data.feature_names].mean()
    malignant_means = df[df.target==0][data.feature_names].mean()
    overall_means = df[data.feature_names].mean()
    overall_stds = df[data.feature_names].std()

    st.markdown("**Quick preset:**")
    preset = st.radio("", ["Average Benign", "Average Malignant", "Custom"], horizontal=True)

    if preset == "Average Benign":
        defaults = benign_means
    elif preset == "Average Malignant":
        defaults = malignant_means
    else:
        defaults = overall_means

    # Show top 10 most important features only (cleaner UX)
    mean_diff = abs(benign_means - malignant_means)
    top10_feats = mean_diff.nlargest(10).index.tolist()

    st.markdown("**Adjust top 10 discriminative features:**")
    cols = st.columns(2)
    user_input = dict(zip(data.feature_names, defaults.values))

    for i, feat in enumerate(top10_feats):
        col = cols[i % 2]
        min_val = float(df[feat].min())
        max_val = float(df[feat].max())
        default_val = float(defaults[feat])
        user_input[feat] = col.slider(
            feat, min_val, max_val, default_val,
            step=float((max_val - min_val) / 100)
        )

    if st.button("🧬 Run Prediction", type="primary", use_container_width=True):
        input_array = np.array([[user_input[f] for f in data.feature_names]])
        input_scaled = scaler.transform(input_array)

        model = trained_models[selected_model]['model']
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="predict-benign">
                <h2>🟢 BENIGN</h2>
                <p style="font-size:1.1rem">Confidence: <strong>{prob[1]*100:.1f}%</strong></p>
                <p style="color:#155724">The model predicts this tumor is <strong>benign</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-malignant">
                <h2>🔴 MALIGNANT</h2>
                <p style="font-size:1.1rem">Confidence: <strong>{prob[0]*100:.1f}%</strong></p>
                <p style="color:#721c24">The model predicts this tumor is <strong>malignant</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("P(Benign)", f"{prob[1]*100:.1f}%")
        col2.metric("P(Malignant)", f"{prob[0]*100:.1f}%")

        st.info("⚠️ This tool is for educational purposes only and is not a medical diagnostic tool.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#6b6b78; font-size:0.85rem'>"
    "Built by Bhavya Sevak · M.S. Biomedical Informatics, ASU · "
    "<a href='https://bsevakstark.github.io' target='_blank'>Portfolio</a> · "
    "<a href='https://www.linkedin.com/in/bhavya-sevak-48515b215' target='_blank'>LinkedIn</a>"
    "</center>",
    unsafe_allow_html=True
)
