import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import shap
import joblib

# 📌 Carregar dataset
dataset_uti = pd.read_csv("dataset_UTI_completo.csv")

# 📌 Converter 'gender' para numérico
label_encoder = LabelEncoder()
dataset_uti['gender'] = label_encoder.fit_transform(dataset_uti['gender'])

# 📌 Definir colunas a excluir
colunas_excluir = ['subject_id', 'status_morte', 'morte_hospitalar', 'morte_1ano']

# 📌 Definir variável-alvo e features
y = dataset_uti['morte_na_uti']
X = dataset_uti.drop(columns=colunas_excluir + ['morte_na_uti'])

# 📌 Preencher valores ausentes
X.fillna(X.mean(), inplace=True)

# 📌 Normalizar os dados
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 📌 Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# 📌 Balancear com RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# 📌 Modelo Random Forest com class_weight
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 📌 Avaliação com cross-validation
cv_scores = cross_val_score(rf, X_resampled, y_resampled, cv=5, scoring='f1')
print(f"\n🎯 F1-score médio na validação cruzada: {np.mean(cv_scores):.4f}")

# 📌 Treinar modelo
rf.fit(X_resampled, y_resampled)

# 📌 Previsões
y_pred = rf.predict(X_test)

# 📌 Avaliação
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n📊 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 📌 Salvar modelo
joblib.dump(rf, "modelo_rf_morte_uti.pkl")

# 📌 SHAP
print("\n🔍 Gerando análise SHAP...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Converter para DataFrame
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Se shap_values for lista (binário), usa o índice 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Calcular impacto médio
impacto_medio = np.mean(np.abs(shap_values), axis=0)
shap_df = pd.DataFrame({
    'variavel': X_test_df.columns,
    'impacto_medio': impacto_medio
}).sort_values(by='impacto_medio', ascending=False)

print("\n🔬 Variáveis ordenadas por impacto médio:")
print(shap_df)

# 📌 Plot SHAP
shap.summary_plot(shap_values, X_test_df, show=False)
plt.tight_layout()
plt.savefig("grafico_shap_uti_morte.png", dpi=300)
plt.close()