import requests
import pandas as pd
import streamlit as st
from LSTMStockPredictor import LSTMStockPredictor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Configuração da interface com Streamlit
st.title("TechChallenge - Fase 04")
st.write("Este aplicativo permite consultar o histórico de preços de ações da B3.")
st.write("**Membros do projeto:** Kleryton de Souza, Lucas Paim, Maiara Giavoni, Rafael Tafelli")

# Entrada do usuário
acao = st.selectbox(
    "Selecione a ação:",
    options=[
        'PETR4.SA', 'DIS',
        'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 
        'ABEV3.SA', 'MGLU3.SA', 'WEGE3.SA', 'RENT3.SA', 'B3SA3.SA'
    ],
    index=0
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
        vBase['Date'] = pd.to_datetime(vBase['Date']).dt.strftime('%Y-%m-%d')
        vBase = vBase[['Date', 'Close']].copy()
        vBase['Close'] = vBase['Close'].astype('float64')
        vBase = vBase.reset_index(drop=True) 
    return acoes, data_inicio, data_fim, vBase

if st.button("Consultar"):
    with st.spinner("Consultando e processando os dados, por favor aguarde..."):
        symbol, start_date, end_date, df = consultar_historico([acao], data_inicio, data_fim)

        if not df.empty:
            predictor = LSTMStockPredictor(symbol, start_date, end_date, df)
        
            predictor.preprocess()
            predictor.build_model()
            predictor.train_model(epochs=20, batch_size=32, save=True)
            predictor.evaluate_and_forecast()
        
            # Exibir métricas de avaliação em formato expansível
            with st.expander("Métricas de Avaliação"):
                metrics_df = predictor.get_metrics_df()
                st.table(metrics_df)

            forecast_df = predictor.get_forecast_df()
            forecast_df['Data'] = pd.to_datetime(forecast_df['Data'])
            df['Date'] = pd.to_datetime(df['Date'])

            # Exibe a tabela de previsões antes do gráfico
            st.subheader("Tabela de Previsões dos Próximos 7 Dias")
            st.table(forecast_df)

            st.subheader("Histórico e Previsão dos Próximos 7 Dias")
            fig, ax = plt.subplots()

            # Plota o histórico normalmente
            ax.plot(df['Date'], df['Close'], label='Histórico', color='blue')

            # Plota a previsão a partir do último ponto do histórico, sem marcador
            previsao_datas = pd.concat([pd.Series([df['Date'].iloc[-1]]), forecast_df['Data']], ignore_index=True)
            previsao_precos = pd.concat([pd.Series([df['Close'].iloc[-1]]), forecast_df['Preço Previsto']], ignore_index=True)
            ax.plot(previsao_datas, previsao_precos, label='Previsão', color='orange', linestyle='-') 

            ax.set_xlabel('Data')
            ax.set_ylabel('Preço de Fechamento')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Nenhum dado retornado para os parâmetros informados.")

