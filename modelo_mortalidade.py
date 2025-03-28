import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Carregar dataset tratado
df = pd.read_csv("dataset_UTI.csv")

# Separar variáveis independentes e dependente
X = df.drop(columns=['subject_id', 'target'])
y = df['target']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Pipeline do modelo otimizado
logistic_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(
        C=0.1, 
        class_weight='balanced', 
        penalty='l2', 
        solver='liblinear', 
        random_state=42))
])

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(logistic_model, X, y, cv=cv, scoring=['precision', 'recall', 'f1', 'accuracy'])

print("\n📌 Resultados da Validação Cruzada:")
for metric in ['test_precision', 'test_recall', 'test_f1', 'test_accuracy']:
    print(f"{metric[5:].capitalize()}: {np.mean(scores[metric]):.3f}")

# Treinamento final
logistic_model.fit(X_train, y_train)

# Avaliação no teste
print("\n📌 Avaliação Final no Teste:")
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Salvar o modelo
joblib.dump(logistic_model, "modelo_logistico_otimizado.pkl")
print("✅ Modelo Logístico salvo como 'modelo_logistico_otimizado.pkl'")

# Explicabilidade com SHAP
# Aplicar transformação completa dos dados
X_train_transformed = logistic_model.named_steps['scaler'].transform(
    logistic_model.named_steps['imputer'].transform(X_train)
)
X_test_transformed = logistic_model.named_steps['scaler'].transform(
    logistic_model.named_steps['imputer'].transform(X_test)
)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=X_test.columns)

# SHAP explainer e valores
explainer = shap.LinearExplainer(logistic_model.named_steps['logistic'], X_train_transformed)
shap_values = explainer(X_test_transformed)

# Plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_logistic_importancia.png", dpi=300)
print("📈 Gráfico SHAP salvo como 'shap_logistic_importancia.png'")

# Importância média
importances = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Variável': X_test.columns,
    'Importância Média (SHAP)': importances
}).sort_values(by='Importância Média (SHAP)', ascending=False)

print("\n📊 Importância das Variáveis (SHAP):")
print(feature_importance.to_string(index=False))
