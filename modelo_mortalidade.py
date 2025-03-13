import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 📌 Carregar os dados
admissions_df = pd.read_csv("admissions.csv")
icustays_df = pd.read_csv("icustays.csv")
labevents_df = pd.read_csv("labevents.csv")
diagnoses_icd_df = pd.read_csv("diagnoses_icd.csv")
chartevents_df = pd.read_csv("chartevents.csv", usecols=['subject_id', 'itemid', 'value'])
procedureevents_df = pd.read_csv("procedureevents.csv")

# 📌 Calcular tempo de internação hospitalar
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])
admissions_df['hospital_los'] = (admissions_df['dischtime'] - admissions_df['admittime']).dt.total_seconds() / (24 * 3600)

# 📌 Ajustar nome da coluna do tempo de UTI
icustays_df.rename(columns={'los': 'icu_los'}, inplace=True)

# 📌 Selecionar variáveis importantes
dados_modelo = admissions_df[['subject_id', 'hospital_los', 'hospital_expire_flag']]
dados_modelo = dados_modelo.merge(icustays_df[['subject_id', 'icu_los']], on="subject_id", how="left")

# 📌 Processar exames laboratoriais (Lactato e Creatinina)
exames_interesse = { "lactato": [50813], "creatinina": [50912] }
labevents_df = labevents_df[labevents_df['itemid'].isin(sum(exames_interesse.values(), []))].copy()
labevents_df['exame_tipo'] = labevents_df['itemid'].map({v: k for k, vals in exames_interesse.items() for v in vals})
labevents_filtrados = labevents_df.groupby(["subject_id", "exame_tipo"])["valuenum"].mean().reset_index()
exames_pivot = labevents_filtrados.pivot(index="subject_id", columns="exame_tipo", values="valuenum").reset_index()
dados_modelo = dados_modelo.merge(exames_pivot, on="subject_id", how="left")

# 📌 Processar comorbidades (Fibrilação Atrial e Hipertensão)
comorbidades_interesse = ["42731", "4019"]
diagnoses_filtro = diagnoses_icd_df[diagnoses_icd_df['icd_code'].isin(comorbidades_interesse)].copy()
diagnoses_filtro = diagnoses_filtro.assign(
    fibrilacao_atrial = diagnoses_filtro['icd_code'] == "42731",
    hipertensao = diagnoses_filtro['icd_code'] == "4019"
)
pacientes_comorbidades = diagnoses_filtro.groupby("subject_id")[['fibrilacao_atrial', 'hipertensao']].any().reset_index()
dados_modelo = dados_modelo.merge(pacientes_comorbidades[['subject_id', 'fibrilacao_atrial', 'hipertensao']], on="subject_id", how="left")

# 📌 Identificar Pacientes com Ventilação Mecânica
ventilacao_itens = [220339, 224684, 224685, 224686, 224687, 224697, 224695, 224696, 224690]
ventilacao_eventos = chartevents_df[chartevents_df['itemid'].isin(ventilacao_itens)].copy()
pacientes_ventilados = ventilacao_eventos.groupby("subject_id")['itemid'].count().reset_index()
pacientes_ventilados['ventilacao_mecanica'] = 1  # Criar variável binária
dados_modelo = dados_modelo.merge(pacientes_ventilados[['subject_id', 'ventilacao_mecanica']], on="subject_id", how="left")
dados_modelo = dados_modelo.assign(ventilacao_mecanica=dados_modelo['ventilacao_mecanica'].fillna(0))

# 📌 Identificar Pacientes com TSR (Terapia de Substituição Renal)
tsr_keywords = ["Hemodialysis", "Dialysis", "Renal Replacement", "Hemofiltration"]
tsr_procedures = procedureevents_df[procedureevents_df['ordercategoryname'].str.contains('|'.join(tsr_keywords), case=False, na=False)].copy()
pacientes_tsr = tsr_procedures.groupby("subject_id")['ordercategoryname'].count().reset_index()
pacientes_tsr['tsr'] = 1  # Criar variável binária
dados_modelo = dados_modelo.merge(pacientes_tsr[['subject_id', 'tsr']], on="subject_id", how="left")
dados_modelo = dados_modelo.assign(tsr=dados_modelo['tsr'].fillna(0))

# 📌 Variável alvo e features
y = dados_modelo["hospital_expire_flag"]
X = dados_modelo.drop(columns=["subject_id", "hospital_expire_flag"])
X.fillna(X.mean(), inplace=True)

# 📌 Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Treinamento do Modelo (Random Forest Atualizado)
modelo_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, class_weight="balanced", random_state=42
)
modelo_rf.fit(X_train, y_train)

# 📌 Avaliação do Modelo Atualizado
y_pred = modelo_rf.predict(X_test)
y_pred_proba = modelo_rf.predict_proba(X_test)[:, 1]

# 📌 Exibir métricas
print("📊 Desempenho do Modelo Atualizado:")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))

# 📌 Salvar modelo atualizado
joblib.dump(modelo_rf, "modelo_mortalidade_atualizado.pkl")
print("✅ Modelo salvo com sucesso: modelo_mortalidade_atualizado.pkl")

# Converter X_test para float antes de passar ao SHAP
X_test_numeric = X_test.astype(float)

# Criar objeto SHAP explainer usando Permutation Explainer (CPU-friendly)
explainer = shap.Explainer(modelo_rf.predict, X_test_numeric)

# Calcular valores SHAP para o conjunto de teste
shap_values = explainer(X_test_numeric)

# Visualização da importância das variáveis com SHAP
shap.summary_plot(shap_values, X_test_numeric, plot_type="bar")