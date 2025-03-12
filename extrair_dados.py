from database import conectar_banco
import pandas as pd

# Conectar ao banco
conn = conectar_banco()

# Query para extrair dados relevantes
query = """
SELECT icu.subject_id, icu.stay_id, icu.icu_intime, icu.icu_outtime,
       vs.heart_rate, vs.resp_rate, vs.spo2, vs.temperature,
       vaso.norepinephrine, vaso.epinephrine, vaso.dopamine, vaso.vasopressin,
       sofa.respiration, sofa.coagulation, sofa.liver, sofa.cardiovascular, sofa.cns, sofa.renal,
       adm.hospital_expire_flag
FROM mimiciv_derived.icustay_detail icu
LEFT JOIN mimiciv_derived.vitalsign vs ON icu.stay_id = vs.stay_id
LEFT JOIN mimiciv_derived.vasoactive_agent vaso ON icu.stay_id = vaso.stay_id
LEFT JOIN mimiciv_derived.sofa sofa ON icu.stay_id = sofa.stay_id
LEFT JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id;

"""

# Rodar a query e salvar os dados
df = pd.read_sql(query, conn)
conn.close()

# Salvar os dados para modelagem
df.to_csv("dados_pacientes.csv", index=False)

print("✅ Dados extraídos e salvos como 'dados_pacientes.csv'")
