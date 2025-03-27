# 📌 IMPORTAÇÃO E PREPARO DOS DADOS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
dataset_uti = patients_df[['subject_id', 'gender', 'anchor_age']]

# 📌 Variáveis clínicas
num_admissoes_uti = icustays_df.groupby('subject_id')['stay_id'].nunique().reset_index(name='num_admissoes_uti')
total_icu_los = icustays_df.groupby('subject_id')['icu_los'].sum().reset_index(name='total_icu_los')
num_diagnosticos = diagnoses_df.groupby('subject_id')['icd_code'].nunique().reset_index(name='num_diagnosticos')

# 📌 Tempo antes da UTI
admissions_uti = admissions_df.merge(icustays_df[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='left')
admissions_uti['tempo_antes_uti'] = (admissions_uti['intime'] - admissions_uti['admittime']).dt.total_seconds() / (24 * 3600)
tempo_antes_uti = admissions_uti[['subject_id', 'tempo_antes_uti']].drop_duplicates()

# 📌 Preencher valores ausentes com mediana
mediana_tempo = tempo_antes_uti['tempo_antes_uti'].median()
tempo_antes_uti['tempo_antes_uti'] = tempo_antes_uti['tempo_antes_uti'].fillna(mediana_tempo)

# 📌 Exames laboratoriais
exames_variancia = labevents_df.groupby('subject_id')['valuenum'].var().reset_index(name='variancia_exames_lab')
exames_first_last = labevents_df.groupby('subject_id')['valuenum'].agg(['first', 'last']).reset_index()
exames_first_last['dif_exames'] = exames_first_last['last'] - exames_first_last['first']
exames_first_last = exames_first_last[['subject_id', 'dif_exames']]

# 📌 Diagnósticos de alto risco
cid_alto_risco = ['A419', 'J960', 'R6521']
diagnosticos_alto_risco = diagnoses_df[diagnoses_df['icd_code'].isin(cid_alto_risco)]
diagnosticos_alto_risco = diagnosticos_alto_risco.groupby('subject_id')['icd_code'].nunique().reset_index(name='num_diagnosticos_alto_risco')

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
    on=['subject_id', 'hadm_id'], how='left'
)
mortalidade_intra_uti = mortalidade_intra_uti[
    (mortalidade_intra_uti['deathtime'].notna()) &
    (mortalidade_intra_uti['deathtime'] >= mortalidade_intra_uti['intime']) &
    (mortalidade_intra_uti['deathtime'] <= mortalidade_intra_uti['outtime'])
][['subject_id']].drop_duplicates()

mortalidade_hospitalar = admissions_df[
    (admissions_df['hospital_expire_flag'] == 1) &
    (~admissions_df['subject_id'].isin(mortalidade_intra_uti['subject_id']))
][['subject_id']].drop_duplicates()

mortalidade_1_ano = admissions_df.merge(patients_df[['subject_id', 'dod']], on='subject_id', how='left')
mortalidade_1_ano = mortalidade_1_ano[
    (mortalidade_1_ano['dod'].notna()) &
    (mortalidade_1_ano['dischtime'].notna()) &
    ((mortalidade_1_ano['dod'] - mortalidade_1_ano['dischtime']).dt.days <= 365)
][['subject_id']].drop_duplicates()

# 📌 Adicionar flags de mortalidade
dataset_uti['morte_na_uti'] = dataset_uti['subject_id'].isin(mortalidade_intra_uti['subject_id']).astype(int)
dataset_uti['morte_hospitalar'] = dataset_uti['subject_id'].isin(mortalidade_hospitalar['subject_id']).astype(int)
dataset_uti['morte_1ano'] = dataset_uti['subject_id'].isin(mortalidade_1_ano['subject_id']).astype(int)

# 📌 Criar status categórico
def definir_status(row):
    if row['morte_na_uti'] == 1:
        return 'morte_uti'
    elif row['morte_hospitalar'] == 1:
        return 'morte_hospitalar'
    elif row['morte_1ano'] == 1:
        return 'morte_1ano'
    else:
        return 'vivo'

dataset_uti['status_morte'] = dataset_uti.apply(definir_status, axis=1)

# 📌 Unir todas as variáveis
dataset_uti = dataset_uti.merge(num_admissoes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(total_icu_los, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_diagnosticos, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(tempo_antes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_variancia, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_first_last, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(diagnosticos_alto_risco, on='subject_id', how='left').fillna({'num_diagnosticos_alto_risco': 0})
dataset_uti = dataset_uti.merge(num_procedimentos, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(pacientes_ventilacao, on='subject_id', how='left').fillna({'ventilacao': 0})
dataset_uti = dataset_uti.merge(pacientes_vasopressores, on='subject_id', how='left').fillna({'vasopressor': 0})
dataset_uti = dataset_uti.merge(pacientes_tsr, on='subject_id', how='left').fillna({'TSR': 0})

# 📌 Remover duplicatas e limitar a 100 pacientes
dataset_uti = dataset_uti.drop_duplicates(subset=['subject_id']).sort_values(by='subject_id').head(100)

# 📌 Salvar
dataset_uti.to_csv("dataset_UTI_completo.csv", index=False)
print("✅ Dataset salvo como 'dataset_UTI_completo.csv'")