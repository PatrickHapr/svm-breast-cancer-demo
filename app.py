import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

REPO = "Hapyr/svm-breast-cancer-wisconsin"
LOCAL_PKL = os.path.join("..", "svm-breast-cancer-wisconsin", "svm_breast_cancer.pkl")


@st.cache_resource
def carregar_modelo():
    try:
        caminho = hf_hub_download(REPO, "svm_breast_cancer.pkl")
    except Exception:
        caminho = os.path.join(os.path.dirname(__file__), LOCAL_PKL)
    return joblib.load(caminho)


data = load_breast_cancer(as_frame=True)
FEATURES = list(data.feature_names)
model = carregar_modelo()

st.set_page_config(page_title="SVM Wisconsin Breast Cancer", page_icon="🩺", layout="wide")
st.title("🩺 SVM - Wisconsin Breast Cancer")
st.markdown(
    "Classifica **Maligno/Benigno** a partir das 30 features do dataset Wisconsin. "
    "Modelo: [Hapyr/svm-breast-cancer-wisconsin](https://huggingface.co/Hapyr/svm-breast-cancer-wisconsin)."
)

if "exemplo" not in st.session_state:
    st.session_state.exemplo = {f: float(data.frame[f].iloc[0]) for f in FEATURES}

st.subheader("30 features")
vals = {}
cols = st.columns(3)
for i, nome in enumerate(FEATURES):
    with cols[i % 3]:
        vals[nome] = st.number_input(
            nome, value=st.session_state.exemplo[nome], format="%.4f", key=nome
        )

c1, c2 = st.columns(2)
with c1:
    if st.button("📋 Preencher exemplo aleatório"):
        i = int(np.random.randint(0, len(data.frame)))
        st.session_state.exemplo = {f: float(data.frame[f].iloc[i]) for f in FEATURES}
        st.rerun()
with c2:
    classificar = st.button("🧠 Classificar", type="primary")

if classificar:
    X = pd.DataFrame([vals], columns=FEATURES)
    with st.spinner("Classificando..."):
        p = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])
    nome = "Benigno" if pred == 1 else "Maligno"
    st.progress(int(round(p[pred] * 100)), text=f"{nome}: {p[pred] * 100:.1f}%")
    st.markdown(f"### {'🟢' if pred == 1 else '🔴'} {nome} ({p[pred] * 100:.1f}%)")
    st.markdown(f"- Maligno: {p[0] * 100:.1f}%\n- Benigno: {p[1] * 100:.1f}%")
    st.info("⚠️ Demonstração educacional — não é diagnóstico médico.")
