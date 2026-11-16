# Deploy no Streamlit Community Cloud (grátis)

## 1) Testar localmente

```
pip install streamlit scikit-learn joblib huggingface_hub pandas numpy
streamlit run app.py
```

Abrir http://localhost:8501

## 2) Subir para o GitHub

- Crie um repositório **público** (ex: `svm-breast-cancer-demo`).
- Adicione **apenas** `app.py` + `requirements.txt` (o `.pkl` não vai: o app baixa do HF em runtime).
- Opção A (CLI):

```
git init
git add app.py requirements.txt
git commit -m "SVM Wisconsin demo"
git branch -M main
git remote add origin <URL_DO_REPO>
git push -u origin main
```

- Opção B (gh): `gh repo create svm-breast-cancer-demo --public --source=. --push`

## 3) Deploy

1. Entre em https://share.streamlit.app com sua conta GitHub.
2. "Deploy a new app" → escolha o repo/branch.
3. Main file path: `app.py` → Deploy.
4. URL pública `*.streamlit.app` aparece na tela.

## Notas

- O modelo vem de `huggingface.co/Hapyr/svm-breast-cancer-wisconsin` (público), então não precisa de segredo.
- Se o download do HF falhar, o app tenta o caminho local `../svm-breast-cancer-wisconsin/svm_breast_cancer.pkl`.
