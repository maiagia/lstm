

✅ Validação por Requisito do Desafio
1. Coleta e Pré-processamento dos Dados ✅
Uso de yfinance com símbolo configurável (symbol='DIS')

Extração da coluna 'Close'

Normalização com MinMaxScaler

Transformação em sequências temporais para LSTM (time_steps = 30)

Separação treino/teste sem embaralhar

2. Modelo LSTM e Treinamento ✅
Arquitetura LSTM com 2 camadas LSTM + Dropout

Camada densa de saída

Otimizador adam + mean_squared_error

Parâmetros epochs e batch_size configuráveis

Validação com conjunto de teste

3. Avaliação com Métricas ✅
MAE, MAPE, RMSE e SMAPE calculadas e retornadas

Resultados armazenados em self.df_metricas como DataFrame

4. Salvamento do Modelo ✅
Método save_all() salva:

Modelo (.h5)

Scaler (.pkl)

Chamado automaticamente se save=True em train_model()

5. Previsão Futura (7 dias) ✅
Forecast recursivo para 7 dias além da última data conhecida

Resultados formatados em self.df_previsoes_futuras

6. Visualização ✅
Gráfico com últimos 30 dias de dados reais vs. previstos + 7 dias futuros

Formatação de datas com matplotlib.dates
