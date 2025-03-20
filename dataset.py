import pandas as pd
import os

# 📌 Definir a pasta onde os arquivos CSV estão armazenados
pasta_dados = "dados_UTI"

# 📌 Carregar os arquivos CSV da pasta
arquivos = {arquivo.split('.')[0]: pd.read_csv(os.path.join(pasta_dados, arquivo)) for arquivo in os.listdir(pasta_dados) if arquivo.endswith(".csv")}

# 📌 Atribuir os DataFrames às variáveis correspondentes
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
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'], errors='coerce')
icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'], errors='coerce')
patients_df['dod'] = pd.to_datetime(patients_df['dod'], errors='coerce')

# 📌 Calcular a duração da internação na UTI
icustays_df['icu_los'] = (icustays_df['outtime'] - icustays_df['intime']).dt.total_seconds() / (24 * 3600)

# 📌 Criar dataset base com informações demográficas
dataset_uti = patients_df[['subject_id', 'gender', 'anchor_age']]

# 📌 Número de admissões na UTI por paciente
num_admissoes_uti = icustays_df.groupby('subject_id')['stay_id'].nunique().reset_index()
num_admissoes_uti.columns = ['subject_id', 'num_admissoes_uti']

# 📌 Tempo total de internação na UTI por paciente
total_icu_los = icustays_df.groupby('subject_id')['icu_los'].sum().reset_index()
total_icu_los.columns = ['subject_id', 'total_icu_los']

# 📌 Número total de diagnósticos por paciente
num_diagnosticos = diagnoses_df.groupby('subject_id')['icd_code'].nunique().reset_index()
num_diagnosticos.columns = ['subject_id', 'num_diagnosticos']

# 📌 Tempo antes da UTI (quantos dias o paciente ficou no hospital antes da UTI)
admissions_uti = admissions_df.merge(icustays_df[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='left')
admissions_uti['tempo_antes_uti'] = (admissions_uti['intime'] - admissions_uti['admittime']).dt.total_seconds() / (24 * 3600)
tempo_antes_uti = admissions_uti[['subject_id', 'tempo_antes_uti']].drop_duplicates()

# 📌 Variabilidade dos exames laboratoriais
exames_variancia = labevents_df.groupby('subject_id')['valuenum'].var().reset_index()
exames_variancia.columns = ['subject_id', 'variancia_exames_lab']

# 📌 Diferença entre primeiro e último exame
exames_first_last = labevents_df.groupby('subject_id')['valuenum'].agg(['first', 'last']).reset_index()
exames_first_last['dif_exames'] = exames_first_last['last'] - exames_first_last['first']
exames_first_last = exames_first_last[['subject_id', 'dif_exames']]

# 📌 Contagem de diagnósticos de alto risco (Sepse, choque séptico, insuficiência respiratória)
cid_alto_risco = ['A419', 'J960', 'R6521']
diagnosticos_alto_risco = diagnoses_df[diagnoses_df['icd_code'].isin(cid_alto_risco)]
diagnosticos_alto_risco = diagnosticos_alto_risco.groupby('subject_id')['icd_code'].nunique().reset_index()
diagnosticos_alto_risco.columns = ['subject_id', 'num_diagnosticos_alto_risco']

# 📌 Número total de procedimentos recebidos
num_procedimentos = procedureevents_df.groupby('subject_id')['ordercategoryname'].nunique().reset_index()
num_procedimentos.columns = ['subject_id', 'num_procedimentos']

# 📌 Uso de ventilação mecânica
ventilacao_df = procedureevents_df[procedureevents_df['ordercategoryname'].str.contains("Ventilation", case=False, na=False)]
pacientes_ventilacao = ventilacao_df[['subject_id']].drop_duplicates()
pacientes_ventilacao['ventilacao'] = 1

# 📌 Uso de vasopressores
vasopressores = ["dopamine", "epinephrine", "norepinephrine", "phenylephrine", "vasopressin"]
pacientes_vasopressores = prescriptions_df[prescriptions_df['drug'].str.contains('|'.join(vasopressores), case=False, na=False)]
pacientes_vasopressores = pacientes_vasopressores[['subject_id']].drop_duplicates()
pacientes_vasopressores['vasopressor'] = 1

# 📌 Uso de Terapia de Substituição Renal (TSR)
tsr_procedures = ["Dialysis", "CRRT Filter Change"]
pacientes_tsr = procedureevents_df[procedureevents_df['ordercategoryname'].isin(tsr_procedures)][['subject_id']].drop_duplicates()
pacientes_tsr['TSR'] = 1

# 📌 Unir todas as variáveis ao dataset principal
dataset_uti = dataset_uti.merge(num_admissoes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(total_icu_los, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(num_diagnosticos, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(tempo_antes_uti, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_variancia, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(exames_first_last, on='subject_id', how='left')
dataset_uti = dataset_uti.merge(diagnosticos_alto_risco, on='subject_id', how='left').fillna({'num_diagnosticos_alto_risco': 0})
dataset_uti = dataset_uti.merge(num_procedimentos, on='subject_id', how='left')

# Adicionar colunas de tratamento
dataset_uti = dataset_uti.merge(pacientes_ventilacao, on='subject_id', how='left').fillna({'ventilacao': 0})
dataset_uti = dataset_uti.merge(pacientes_vasopressores, on='subject_id', how='left').fillna({'vasopressor': 0})
dataset_uti = dataset_uti.merge(pacientes_tsr, on='subject_id', how='left').fillna({'TSR': 0})

# 📌 Adicionar a variável-alvo: Mortalidade intra-UTI
pacientes_mortos_uti = admissions_df[admissions_df['discharge_location'] == 'DIED'][['subject_id', 'hadm_id']]
pacientes_mortos_uti = pacientes_mortos_uti.merge(icustays_df[['subject_id', 'hadm_id']], on=['subject_id', 'hadm_id'], how='inner')
dataset_uti['mortalidade_intra_uti'] = dataset_uti['subject_id'].isin(pacientes_mortos_uti['subject_id']).astype(int)

# 📌 Remover duplicatas e garantir 100 pacientes
dataset_uti = dataset_uti.drop_duplicates(subset=['subject_id']).sort_values(by='subject_id').head(100)

# 📌 Salvar o dataset final
dataset_uti.to_csv("dataset_UTI_completo.csv", index=False)

print("✅ Dataset salvo como 'dataset_UTI_completo.csv'")