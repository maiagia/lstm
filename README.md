# 📈 Previsão de Preços de Ações com LSTM

Este projeto é o desafio da Fase 4 do MBA em Machine Learning da FIAP. Ele tem como objetivo aplicar técnicas de Deep Learning — em especial, redes LSTM — para prever preços de ações da bolsa de valores com base em séries temporais. A aplicação final entrega uma API RESTful e uma interface interativa em Streamlit para interação com o modelo.

---
👨‍💻 Equipe

Kleryton de Souza, Lucas Paim, Maiara Giavoni, Rafael Tafelli

## 🚀 Visão Geral

- Coleta dados históricos da B3 via API customizada integrada com yFinance.
- Treina um modelo LSTM para prever o fechamento de ações.
- Disponibiliza endpoints RESTful para previsão e consulta de histórico.
- Cria uma interface visual com Streamlit.
- Utiliza Docker para facilitar o deploy e a escalabilidade da solução.

---

## ✅ Requisitos do Desafio e Como Foram Atendidos

### 1. Coleta e Pré-processamento dos Dados

- ✔️ Utilização da biblioteca `yfinance` para baixar dados históricos de ações diretamente na API, através do endpoint `/api/historico_preco`.
- ✔️ Interface Streamlit permite que o usuário defina a ação e o intervalo de datas.
- ✔️ Os dados são transformados e normalizados com `MinMaxScaler`.

### 2. Desenvolvimento do Modelo LSTM

- ✔️ Modelo LSTM construído com Keras e TensorFlow com duas camadas LSTM e camadas de Dropout.
- ✔️ O modelo é treinado com dados históricos passados pela API.
- ✔️ Métricas de avaliação implementadas: MAE, MAPE, RMSE e SMAPE.
- ✔️ Avaliação e predição para os próximos 7 dias são exibidas com gráfico e tabela.

### 3. Salvamento e Exportação do Modelo

- ✔️ Após o treinamento, o modelo é salvo no formato `.h5` e o scaler como `.pkl` usando `joblib`.
- ✔️ Isso permite reutilizar o modelo treinado para inferências futuras sem retraining.

### 4. Deploy do Modelo

- ✔️ API criada com FastAPI com endpoints:
  - `/api/historico_preco`: retorna dados históricos via yFinance.
  - `/api/predict`: recebe uma lista de preços e retorna a previsão.
- ✔️ Interface Streamlit comunica com a API para orquestrar toda a operação.
- ✔️ Geração de relatórios em PDF com tabela e gráfico de previsões.

### 5. Escalabilidade e Monitoramento

- ✔️ Deploy Dockerizado com dois containers: um para API (FastAPI) e outro para Frontend (Streamlit).
- ✔️ Comunicação entre serviços via `docker-compose` com rede interna isolada.

---

## 📁 Estrutura do Projeto

├── api/ # API FastAPI
│   ├── main.py # Inicialização e roteamento
│   ├── rotas/ # Endpoints da API
│   ├── modelos/ # Modelos Pydantic para validação
│   ├── servicos/ # Lógica de negócio e integração
│   └── Dockerfile.api # Dockerfile da API
│
├── app/ # Interface Streamlit
│   ├── app.py # Interface gráfica
│   ├── LSTMStockPredictor.py # Classe de treinamento/predição
│   └── Dockerfile.streamlit
│
├── docker-compose.yml # Orquestração dos containers
├── modelo_lstm.h5 # Modelo treinado
├── scaler.pkl # Scaler salvo
└── requirements-*.txt # Dependências
---

## 🧪 Como Executar o Projeto

### Pré-requisitos

- Docker e Docker Compose instalados.

### Executar com Docker

```bash
docker-compose up --build

Acessar:
📊 Streamlit: http://localhost:8501

🔗 API Swagger (FastAPI): http://localhost:8000/docs

🔌 Endpoints da API
POST /api/historico_preco
Solicita o histórico de preços de uma ação via yFinance.

Exemplo de requisição JSON:

json
Copiar
Editar
{
  "acao": "PETR4.SA",
  "data_inicio": "2023-01-01",
  "data_fim": "2024-01-01"
}
POST /api/predict
Recebe uma lista de preços anteriores e retorna a previsão para o próximo dia.

Exemplo de requisição JSON:

json
Copiar
Editar
{
  "precos_anteriores": [28.34, 28.55, 28.42, ..., 30.15]
}
🎨 Interface Streamlit
Permite ao usuário selecionar uma ação e o intervalo de tempo.

Exibe métricas do modelo.

Mostra gráfico comparando histórico real com a previsão.

Gera relatório em PDF com tabela + gráfico.

📦 Requisitos
Instalados automaticamente via requirements-api.txt e requirements-streamlit.txt durante o build do Docker.

Principais pacotes:

fastapi, uvicorn

tensorflow, scikit-learn, yfinance, pandas, numpy

streamlit, fpdf, matplotlib


🎥 Entregáveis
✅ Código-fonte completo com README.

✅ Containers Docker prontos para deploy.

✅ API RESTful funcional.

✅ Interface Streamlit interativa.

✅ Geração de PDF com previsões.
