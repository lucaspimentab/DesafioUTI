import os
import psycopg2
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

def conectar_banco():
    try:
        print("Tentando conectar ao banco...")
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        print("✅ Conexão bem-sucedida!")
        return conn
    except Exception as e:
        print("❌ Erro ao conectar ao banco:", e)
        return None

if __name__ == "__main__":
    conectar_banco()