import pandas as pd
import numpy as np
import os

# 📌 Definir a pasta onde os arquivos CSV estão armazenados
pasta_dados = "dados_UTI"
arquivos = {
    arquivo.split('.')[0]: pd.read_csv(os.path.join(pasta_dados, arquivo))
    for arquivo in os.listdir(pasta_dados) if arquivo.endswith(".csv")
}

# 📌 Atribuir os DataFrames
admissions_df = arquivos.get("admissions")
patients_df = arquivos.get("patients")
prescriptions_df = arquivos.get("prescriptions")
procedureevents_df = arquivos.get("procedureevents")
icustays_df = arquivos.get("icustays")
diagnoses_df = arquivos.get("diagnoses_icd")
icd_descriptions_df = arquivos.get("d_icd_diagnoses")
labevents_df = arquivos.get("labevents")
chartevents_df = arquivos.get("chartevents")

# 📌 Converter colunas de tempo para datetime
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'], errors='coerce')
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'], errors='coerce')
patients_df['dod'] = pd.to_datetime(patients_df['dod'], errors='coerce')
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'], errors='coerce')
icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'], errors='coerce')

# 📌 Calcular duração na UTI
icustays_df['icu_los'] = (icustays_df['outtime'] - icustays_df['intime']).dt.total_seconds() / (24 * 3600)

# 📌 Base demográfica
dataset_uti = patients_df[['subject_id', 'gender', 'anchor_age']].copy()

# 📌 Variáveis clínicas
num_admissoes_uti = icustays_df.groupby('subject_id')['stay_id'].nunique().reset_index(name='num_admissoes_uti')
total_icu_los = icustays_df.groupby('subject_id')['icu_los'].sum().reset_index(name='total_icu_los')
num_diagnosticos = diagnoses_df.groupby('subject_id')['icd_code'].nunique().reset_index(name='num_diagnosticos')

# 📌 Tempo antes da UTI
admissions_uti = admissions_df.merge(icustays_df[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='left')
admissions_uti['tempo_antes_uti'] = (admissions_uti['intime'] - admissions_uti['admittime']).dt.total_seconds() / (24 * 3600)
tempo_antes_uti = admissions_uti[['subject_id', 'tempo_antes_uti']].drop_duplicates()
mediana_tempo = tempo_antes_uti['tempo_antes_uti'].median()
tempo_antes_uti['tempo_antes_uti'] = tempo_antes_uti['tempo_antes_uti'].fillna(mediana_tempo)

# 📌 Exames laboratoriais
exames_variancia = labevents_df.groupby('subject_id')['valuenum'].var().reset_index(name='variancia_exames_lab')
exames_first_last = labevents_df.groupby('subject_id')['valuenum'].agg(['first', 'last']).reset_index()
exames_first_last['dif_exames'] = exames_first_last['last'] - exames_first_last['first']
exames_first_last = exames_first_last[['subject_id', 'dif_exames']]

# 📌 Procedimentos
num_procedimentos = procedureevents_df.groupby('subject_id')['ordercategoryname'].nunique().reset_index(name='num_procedimentos')

# 📌 Tratamentos
ventilacao_df = procedureevents_df[procedureevents_df['ordercategoryname'].str.contains("Ventilation", case=False, na=False)]
pacientes_ventilacao = ventilacao_df[['subject_id']].drop_duplicates()
pacientes_ventilacao['ventilacao'] = 1

vasopressores = ["dopamine", "epinephrine", "norepinephrine", "phenylephrine", "vasopressin"]
pacientes_vasopressores = prescriptions_df[prescriptions_df['drug'].str.contains('|'.join(vasopressores), case=False, na=False)]
pacientes_vasopressores = pacientes_vasopressores[['subject_id']].drop_duplicates()
pacientes_vasopressores['vasopressor'] = 1

tsr_procedures = ["Dialysis", "CRRT Filter Change"]
pacientes_tsr = procedureevents_df[procedureevents_df['ordercategoryname'].isin(tsr_procedures)][['subject_id']].drop_duplicates()
pacientes_tsr['TSR'] = 1

# 📌 Mortalidade
mortalidade_intra_uti = icustays_df.merge(
    admissions_df[['subject_id', 'hadm_id', 'deathtime']],
    on=['subject_id', 'hadm_id'],
    how='left'
)
mortalidade_intra_uti = mortalidade_intra_uti[
    (mortalidade_intra_uti['deathtime'].notna()) &
    (mortalidade_intra_uti['deathtime'] >= mortalidade_intra_uti['intime']) &
    (mortalidade_intra_uti['deathtime'] <= mortalidade_intra_uti['outtime'])
][['subject_id']].drop_duplicates()

mortalidade_hospitalar = admissions_df[
    (admissions_df['hospital_expire_flag'] == 1) &
    (~admissions_df['subject_id'].isin(mortalidade_intra_uti['subject_id'])) &
    (admissions_df['subject_id'].isin(icustays_df['subject_id']))
][['subject_id']].drop_duplicates()

# 📌 Mortalidade em até 1 ano
ultima_admissao = admissions_df.sort_values(['subject_id', 'dischtime'], ascending=[True, False]).drop_duplicates(subset='subject_id', keep='first')
ultima_uti = icustays_df.sort_values(['subject_id', 'outtime'], ascending=[True, False]).drop_duplicates(subset='subject_id', keep='first')

mortalidade_1_ano = ultima_admissao.merge(
    patients_df[['subject_id', 'dod']], on='subject_id', how='left'
).merge(
    ultima_uti[['subject_id', 'outtime']], on='subject_id', how='left'
)

mortalidade_1_ano = mortalidade_1_ano[
    (mortalidade_1_ano['hospital_expire_flag'] == 0) &
    (mortalidade_1_ano['dod'].notna()) &
    (mortalidade_1_ano['dischtime'].notna()) &
    (mortalidade_1_ano['outtime'].notna()) &
    (mortalidade_1_ano['dod'] > mortalidade_1_ano['outtime']) &
    ((mortalidade_1_ano['dod'] - mortalidade_1_ano['dischtime']).dt.days <= 365)
][['subject_id']].drop_duplicates()

# 📌 Flags de mortalidade e target
dataset_uti['morte_na_uti'] = dataset_uti['subject_id'].isin(mortalidade_intra_uti['subject_id']).astype(int)
dataset_uti['morte_hospitalar'] = dataset_uti['subject_id'].isin(mortalidade_hospitalar['subject_id']).astype(int)
dataset_uti['morte_1ano'] = dataset_uti['subject_id'].isin(mortalidade_1_ano['subject_id']).astype(int)
dataset_uti['mortalidade_total'] = (
    (dataset_uti['morte_na_uti'] + dataset_uti['morte_hospitalar'] + dataset_uti['morte_1ano']) > 0
).astype(int)
dataset_uti['target'] = dataset_uti['mortalidade_total']

# 📌 Variáveis adicionais
dataset_uti = dataset_uti.merge(num_admissoes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(total_icu_los, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_diagnosticos, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(tempo_antes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_variancia, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_first_last, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_procedimentos, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(pacientes_ventilacao, on='subject_id', how='left').fillna({'ventilacao': 0})
dataset_uti = dataset_uti.merge(pacientes_vasopressores, on='subject_id', how='left').fillna({'vasopressor': 0})
dataset_uti = dataset_uti.merge(pacientes_tsr, on='subject_id', how='left').fillna({'TSR': 0})

dataset_uti[['ventilacao', 'vasopressor', 'TSR']] = dataset_uti[['ventilacao', 'vasopressor', 'TSR']].astype(int)

# 📌 Internações, admissões e exames
admissions_df['duracao_internacao'] = (admissions_df['dischtime'] - admissions_df['admittime']).dt.total_seconds() / (24 * 3600)
duracao_internacao = admissions_df.groupby('subject_id')['duracao_internacao'].max().reset_index()
num_admissoes_hospital = admissions_df.groupby('subject_id')['hadm_id'].nunique().reset_index(name='num_admissoes_hospital')
num_exames_lab = labevents_df.groupby('subject_id')['itemid'].count().reset_index(name='num_exames_lab')

dataset_uti = dataset_uti.merge(duracao_internacao, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_admissoes_hospital, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_exames_lab, on='subject_id', how='left')

df_modelo = dataset_uti.drop(columns=['morte_na_uti', 'morte_hospitalar', 'morte_1ano', 'mortalidade_total'])
df_modelo['gender'] = df_modelo['gender'].map({'M': 1, 'F': 0})

# Novas variáveis derivadas
df_modelo['razao_diagnosticos_admissoes'] = df_modelo['num_diagnosticos'] / (df_modelo['num_admissoes_hospital'] + 1)
df_modelo['razao_uti_hosp'] = df_modelo['total_icu_los'] / (df_modelo['duracao_internacao'] + 1)

for col in ['dif_exames']:
    df_modelo[f'log_{col}'] = np.log1p(df_modelo[col])


# Heart Rate e Respiratory Rate
clinical_vars = chartevents_df[chartevents_df['itemid'].isin([220045, 220210])]
clinical_vars = clinical_vars.pivot_table(index='subject_id', columns='itemid', values='valuenum', aggfunc='first').reset_index()
clinical_vars.columns = ['subject_id', 'heart_rate', 'respiratory_rate']

# Mesclar no dataset principal
df_modelo = df_modelo.merge(clinical_vars, on='subject_id', how='left')

# Lactato, Creatinina, Bilirrubina total
lab_vars = labevents_df[labevents_df['itemid'].isin([50813, 50912, 50885])]
lab_vars = lab_vars.pivot_table(index='subject_id', columns='itemid', values='valuenum', aggfunc='first').reset_index()
lab_vars.columns = ['subject_id', 'lactate', 'creatinine', 'bilirubin_total']

# Mesclar no dataset principal
df_modelo = df_modelo.merge(lab_vars, on='subject_id', how='left')

# Preencher valores faltantes com a mediana
cols_imputar = ['heart_rate', 'respiratory_rate', 'lactate', 'creatinine', 'bilirubin_total']
df_modelo[cols_imputar] = df_modelo[cols_imputar].fillna(df_modelo[cols_imputar].median())



# Remover duplicidades com base em 'subject_id'
df_modelo = df_modelo.drop_duplicates(subset=['subject_id'])

df_modelo.to_csv("dataset_UTI.csv", index=False)
print("✅ Dataset 'dataset_UTI.csv' salvo!")