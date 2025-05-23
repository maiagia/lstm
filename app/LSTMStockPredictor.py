import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import joblib

class LSTMStockPredictor:
    def __init__(self, symbol, start_date, end_date, df):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        self.df = df

        self.scaler = MinMaxScaler()
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_test_inv = self.y_pred_inv = None
        self.datas_futuros = None
        self.future_predictions_inv = None
        self.df_metricas = None
        self.df_previsoes_futuras = None
        self.time_steps = 30

    def load_data(self):
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.df = df[['Close']]
        return self.df

    def preprocess(self):
        close_values = self.df[['Close']].values  
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = self.scaler.fit_transform(close_values)

        X, y = [], []
        for i in range(self.time_steps, len(df_scaled)):
            X.append(df_scaled[i - self.time_steps:i, 0])
            y.append(df_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def build_model(self):
        time_steps = self.X_train.shape[1]
        features = self.X_train.shape[2]

        model = Sequential()
        model.add(Input(shape=(time_steps, features)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=20, batch_size=32, save=False):
        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_test, self.y_test)
        )
        if save:
            self.save_all()

    def save_all(self, model_path='modelo_lstm.h5', scaler_path='scaler.pkl'):
        """Salva o modelo treinado e o scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Modelo salvo em: {model_path}")
        print(f"Scaler salvo em: {scaler_path}")

    def evaluate_and_forecast(self):
        y_pred = self.model.predict(self.X_test)
        self.y_test_inv = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        self.y_pred_inv = self.scaler.inverse_transform(y_pred)

        # Previsão futura (7 dias)
        last_sequence = self.scaler.transform(self.df[['Close']])[-self.time_steps:].reshape(1, self.time_steps, 1)
        future_predictions = []

        for _ in range(7):
            next_pred = self.model.predict(last_sequence, verbose=0)[0][0]
            future_predictions.append(next_pred)
            last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

        self.future_predictions_inv = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        last_date = pd.to_datetime(self.df['Date'].iloc[-1])
        self.datas_futuros = [last_date + timedelta(days=i + 1) for i in range(7)]

        self.df_previsoes_futuras = pd.DataFrame({
            "Data": self.datas_futuros,
            "Preço Previsto": [f"{v:.2f}" for v in self.future_predictions_inv.flatten()]
        })

        mae = mean_absolute_error(self.y_test_inv, self.y_pred_inv)
        mape = np.mean(np.abs((self.y_test_inv - self.y_pred_inv) / self.y_test_inv)) * 100
        rmse = np.sqrt(mean_squared_error(self.y_test_inv, self.y_pred_inv))
        smape = 100 * np.mean(2 * np.abs(self.y_pred_inv - self.y_test_inv) /
                              (np.abs(self.y_test_inv) + np.abs(self.y_pred_inv)))

        self.df_metricas = pd.DataFrame([{
            "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "RMSE": round(rmse, 2),
            "SMAPE (%)": round(smape, 2)
        }])

    def get_forecast_df(self):
        return self.df_previsoes_futuras

    def get_metrics_df(self):
        return self.df_metricas

if __name__ == "__main__":
    
    symbol = "PETR4.SA"  
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    predictor = LSTMStockPredictor(symbol, start_date, end_date)

    df = predictor.load_data()
    predictor.preprocess()
    predictor.build_model()
    predictor.train_model(epochs=20, batch_size=32, save=True)
    predictor.evaluate_and_forecast()
    # predictor.plot_results()

    print("Métricas de avaliação:")
    print(predictor.get_metrics_df())
    print("\nPrevisão para os próximos 7 dias:")
    print(predictor.get_forecast_df())
    print("\nDataFrame:")
    print(predictor.load_data().to_string(max_rows=20))
