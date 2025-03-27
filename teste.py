import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb

# Carregar dados
df = pd.read_csv("dataset_UTI_completo.csv")
df.columns = df.columns.str.strip()

# Codificação do gênero
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Variável alvo
target = 'morte_na_uti'
drop_cols = ['morte_1ano', 'morte_hospitalar', 'status_morte']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Agrupamento
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [col for col in num_cols if col not in ['subject_id', target]]
agg = {col: 'mean' for col in num_cols}
agg[target] = 'mean'
df_static = df.groupby('subject_id').agg(agg).reset_index()

y = df_static[target].round().astype(int)
X = df_static.drop(columns=['subject_id', target])
X.fillna(X.mean(), inplace=True)

# Normalização
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Técnicas de balanceamento
bal_methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

# Hiperparâmetros fixos do XGBoost
xgb_params = {
    'n_estimators': 463,
    'max_depth': 6,
    'learning_rate': 0.019,
    'min_child_weight': 3,
    'gamma': 0.4,
    'subsample': 0.88,
    'colsample_bytree': 0.86,
    'scale_pos_weight': 10,
    'eval_metric': 'logloss',
    'random_state': 42
}

# Armazenar previsões do ensemble
ensemble_probas = []

# Treinar para cada técnica de balanceamento
for name, sampler in bal_methods.items():
    print(f"\n📌 Técnica de Balanceamento: {name}")
    X_res, y_res = sampler.fit_resample(X_scaled, y)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_res, y_res)

    # Seleção de variáveis após balanceamento
    selector = SelectFromModel(model, threshold="median", prefit=True)
    X_res_sel = selector.transform(X_res)
    X_scaled_sel = selector.transform(X_scaled)

    # Ensemble com RandomForest e LogisticRegression
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_res_sel, y_res)
    lr.fit(X_res_sel, y_res)

    # Média das probabilidades
    probas = (
        model.predict_proba(X_scaled_sel)[:, 1] +
        rf.predict_proba(X_scaled_sel)[:, 1] +
        lr.predict_proba(X_scaled_sel)[:, 1]
    ) / 3

    ensemble_probas.append(probas)

# Média final das probabilidades (voto por média)
final_proba = np.mean(ensemble_probas, axis=0)

# Ajuste de threshold (valor entre 0.2 e 0.35 para otimizar f1 e precisão)
threshold = 0.3
final_preds = (final_proba >= threshold).astype(int)

# Avaliação final
print("\n🧪 Ensemble Final com Seleção de Variáveis e Threshold Ajustado")
print(classification_report(y, final_preds, target_names=['Sobrevive na UTI', 'Morre na UTI']))
print("\n📩 Matriz de Confusão:")
print(confusion_matrix(y, final_preds))
