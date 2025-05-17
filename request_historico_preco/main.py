import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Configuração da interface com Streamlit
st.title("TechChallenge - Fase 04")
st.write("Este aplicativo permite consultar o histórico de preços de ações da B3.")
st.write("**Membros do projeto:** Kleryton de Souza, Lucas Paim, Maiara Giavoni, Rafael Tafelli")

# Entrada do usuário
acoes = st.multiselect(
    "Selecione as ações:",
    options=[
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 
        'ABEV3.SA', 'MGLU3.SA', 'WEGE3.SA', 'RENT3.SA', 'B3SA3.SA'
    ],
    default=['PETR4.SA']
)

data_inicio = st.date_input("Data de início:", value=datetime(2025, 1, 6))
data_fim = st.date_input("Data de fim:", value=datetime(2025, 1, 10))

if st.button("Consultar"):
    # vEndPoint = 'http://localhost:8000/api/historico_preco'
    vEndPoint = 'http://api:8000/api/historico_preco'
    vBase = pd.DataFrame()

    for acao in acoes:
        vPayload = {
            'acao': acao,
            'data_inicio': data_inicio.strftime('%Y-%m-%d'),
            'data_fim': data_fim.strftime('%Y-%m-%d')
        }

        vResponse = requests.post(vEndPoint, json=vPayload)

        if vResponse.status_code == 200:
            vBase_Json = vResponse.json()
            vBaseTemp = pd.DataFrame(vBase_Json)
            vBase = pd.concat([vBase, vBaseTemp], ignore_index=True)
        else:
            st.error(f"Erro ao consultar a ação {acao}: {vResponse.status_code}")

    if not vBase.empty:
        # Tratando valores nulos
        vBase.fillna(0, inplace=True)

        # Convertendo a coluna 'Date' para o formato datetime
        vBase['Date'] = pd.to_datetime(vBase['Date'])

        # Exibindo os dados na interface
        st.dataframe(vBase)

        # Exporta para CSV
        csv_filename = f"saida_historico_csv/historico_precos_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        vBase.to_csv(csv_filename, index=False)
        st.success(f"Arquivo CSV exportado: {csv_filename}")
    else:
        st.warning("Nenhum dado retornado para os parâmetros informados.")