import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Configuração da interface com Streamlit
st.title("TechChallenge - Fase 04")
st.write("Este aplicativo permite consultar o histórico de preços de ações da B3.")
st.write("**Membros do projeto:** Kleryton de Souza, Lucas Paim, Maiara Giavoni, Rafael Tafelli")

# Entrada do usuário
acoes = st.multiselect(
    "Selecione as ações:",
    options=[
        'PETR4.SA', 'DIS',
        # 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 
        # 'ABEV3.SA', 'MGLU3.SA', 'WEGE3.SA', 'RENT3.SA', 'B3SA3.SA'
    ],
    default=['DIS']
    # default=['PETR4.SA']
)

hoje = datetime.today()
um_ano_atras = hoje - timedelta(days=365)

data_inicio = st.date_input("Data de início:", value=um_ano_atras)
data_fim = st.date_input("Data de fim:", value=hoje)

def consultar_historico(acoes, data_inicio, data_fim):
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
        vBase.fillna(0, inplace=True)
        vBase['Date'] = pd.to_datetime(vBase['Date'])
    return vBase, data_inicio, data_fim

if st.button("Consultar"):
    vBase, data_inicio, data_fim = consultar_historico(acoes, data_inicio, data_fim)

    if not vBase.empty:
        st.dataframe(vBase)
        csv_filename = f"saida_historico_csv/historico_precos_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        vBase.to_csv(csv_filename, index=False)
        st.success(f"Arquivo CSV exportado: {csv_filename}")
        # Exemplo de uso das datas:
        st.write(f"Data de início usada: {data_inicio}")
        st.write(f"Data de fim usada: {data_fim}")
    else:
        st.warning("Nenhum dado retornado para os parâmetros informados.")