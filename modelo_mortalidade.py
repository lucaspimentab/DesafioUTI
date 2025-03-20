import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
import shap
import joblib

dataset_uti = pd.read_csv("dataset_UTI_completo.csv")

# Converter 'gender' para numérico
label_encoder = LabelEncoder()
dataset_uti['gender'] = label_encoder.fit_transform(dataset_uti['gender'])

# Agrupar medições por paciente
dataset_uti_static = dataset_uti.groupby('subject_id').mean().reset_index()

# Definir variável-alvo e features
y = dataset_uti_static['mortalidade_intra_uti']
X = dataset_uti_static.drop(columns=['subject_id', 'mortalidade_intra_uti'])

# Preencher valores ausentes com a média
X.fillna(X.mean(), inplace=True)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Aplicar SMOTEENN para balanceamento dos dados
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Definir e treinar modelo Random Forest otimizado
rf_optimized = RandomForestClassifier(
    bootstrap=False,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# Avaliação com validação cruzada
cv_scores = cross_val_score(rf_optimized, X_resampled, y_resampled, cv=5, scoring='f1')
print(f"\nMédia do F1-score na Validação Cruzada: {np.mean(cv_scores):.4f}")

# Treinar modelo final
rf_optimized.fit(X_resampled, y_resampled)

# Fazer previsões no conjunto de teste
y_pred_final = rf_optimized.predict(X_test)

# Relatório de desempenho do modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_final))

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_final))

# Salvar modelo treinado
joblib.dump(rf_optimized, "modelo_rf_uti_otimizado_final.pkl")

# Explicabilidade com SHAP
print("\nGerando análise SHAP...")
explainer = shap.TreeExplainer(rf_optimized)
shap_values = explainer.shap_values(X_test)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

if isinstance(shap_values, list):
    shap_values_correct = np.array(shap_values[1])
else:
    shap_values_correct = np.array(shap_values)

if len(shap_values_correct.shape) == 3:
    shap_values_correct = shap_values_correct.sum(axis=2)

impacto_medio = np.mean(np.abs(shap_values_correct), axis=0)
shap_df = pd.DataFrame({
    'variavel': X_test_df.columns,
    'impacto_medio': impacto_medio
})

shap_df = shap_df.sort_values(by='impacto_medio', ascending=False)
print("\nVariáveis ordenadas por impacto médio no modelo:")
print(shap_df)

# Gerar e salvar gráfico SHAP
shap.summary_plot(shap_values_correct, X_test_df, show=False)
plt.tight_layout()
plt.savefig("grafico_shap.png", dpi=300)
plt.close()