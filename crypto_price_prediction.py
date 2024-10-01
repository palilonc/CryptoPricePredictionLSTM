import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from datetime import datetime
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue

# Ustawienia logów
logging.basicConfig(level=logging.INFO)

# Cache danych
cache = {}

# Queue for thread-safe communication
message_queue = queue.Queue()

# Funkcja do pobierania danych z CoinGecko
def fetch_crypto_data_from_coingecko(cryptocurrency, vs_currency='usd', days=365):
    if cryptocurrency in cache and (datetime.now() - cache[cryptocurrency]['timestamp']).seconds < 3600:
        logging.info(f"Użycie cached danych dla {cryptocurrency}.")
        return cache[cryptocurrency]['data']

    logging.info(f"Pobieranie danych dla {cryptocurrency} z CoinGecko na ostatnie {days} dni...")
    url = f'https://api.coingecko.com/api/v3/coins/{cryptocurrency}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info("Dane pobrane pomyślnie.")

        cache[cryptocurrency] = {
            'data': data['prices'],
            'timestamp': datetime.now()
        }
        return data['prices']
    except requests.exceptions.RequestException as e:
        logging.error(f"Błąd podczas pobierania danych: {e}")
        return None

# Funkcja do przygotowania danych dla modelu LSTM
def prepare_data_for_lstm(prices):
    logging.info("Przygotowywanie danych dla modelu LSTM...")

    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['price']].values)

    logging.info("Dane zostały przygotowane.")
    return df, scaled_data, scaler

# Funkcja do trenowania modelu LSTM
def train_lstm_model(scaled_data, epochs=200, batch_size=32, sequence_length=60):
    logging.info("Trenowanie modelu LSTM...")

    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    logging.info("Model LSTM został wytrenowany.")
    plot_training_loss(history)
    model.save('lstm_model.keras')
    logging.info("Model LSTM zapisany jako 'lstm_model.keras'.")
    
    return model

# Funkcja do wizualizacji strat treningu
def plot_training_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.title('Wykres strat modelu LSTM')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()
    plt.grid()
    plt.show()

# Funkcja do przewidywania cen na x dni do przodu
def predict_future_prices(model, data, scaler, sequence_length=60, days=30):
    logging.info(f"Przewidywanie przyszłych cen na {days} dni...")

    predictions = []
    last_data = data[-sequence_length:]  
    last_data_scaled = scaler.transform(last_data)

    for _ in range(days):
        X_test = np.reshape(last_data_scaled, (1, last_data_scaled.shape[0], 1))
        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        predictions.append(predicted_price[0][0])
        last_data_scaled = np.append(last_data_scaled, predicted_price_scaled)
        last_data_scaled = last_data_scaled[1:] 

    logging.info(f"Przewidywane ceny na {days} dni: {predictions}")
    return predictions

# Funkcja do obliczania dokładności przewidywań
def calculate_accuracy(predicted_prices, actual_prices):
    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)

    differences = np.abs(predicted_prices - actual_prices)
    accurate_predictions = np.sum(differences < (actual_prices * 0.02))  # 2% tolerancji na błąd
    accuracy = (accurate_predictions / len(predicted_prices)) * 100 if len(predicted_prices) > 0 else 0

    logging.info(f"Dokładność przewidywań: {accuracy:.2f}%")
    return accuracy

# Funkcja do zapisywania przewidywanych cen do pliku CSV
def save_predictions_to_csv(predictions, cryptocurrency):
    df = pd.DataFrame(predictions, columns=['Predicted Price'])
    df.to_csv(f"{cryptocurrency}_predicted_prices.csv", index=False)
    logging.info(f"Przewidywane ceny zapisane do pliku {cryptocurrency}_predicted_prices.csv.")

# Funkcja do wyświetlania wykresu
def plot_predictions(df, predicted_prices, actual_prices):
    plt.figure(figsize=(12, 6))

    last_actual_prices = df['price'].tail(30).values  
    last_dates = df.index[-30:]  

    future_dates = pd.date_range(start=last_dates[-1] + pd.Timedelta(days=1), periods=len(predicted_prices))

    plt.plot(last_dates, last_actual_prices, color='#2ca02c', label="Rzeczywiste ceny (aktualne)", linestyle='-', marker='x')
    plt.plot(future_dates, predicted_prices, color='#ff7f0e', label="Przewidywane ceny", linestyle='-', marker='o')

    plt.title('Ceny kryptowalut - Rzeczywiste vs Przewidywane')
    plt.xlabel('Data')
    plt.ylabel('Cena (USD)')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Funkcja do odświeżania wykresu
def refresh_plot(selected_currency, days_to_predict):
    try:
        if not selected_currency:
            messagebox.showerror("Błąd", "Wybierz kryptowalutę.")
            return

        if days_to_predict <= 0:
            messagebox.showerror("Błąd", "Podaj poprawną ilość dni.")
            return
        
        # Pobranie danych historycznych
        prices = fetch_crypto_data_from_coingecko(selected_currency)

        if prices:
            df, scaled_data, scaler = prepare_data_for_lstm(prices)
            model = train_lstm_model(scaled_data)
            predicted_prices = predict_future_prices(model, df[['price']].values, scaler, days=days_to_predict)

            # Obliczanie dokładności
            actual_prices = df['price'].tail(days_to_predict).values
            accuracy = calculate_accuracy(predicted_prices, actual_prices)

            # Enqueue message for accuracy display
            message_queue.put(f"Dokładność przewidywań: {accuracy:.2f}%")

            # Update plot on the main thread
            window.after(0, plot_predictions, df, predicted_prices, actual_prices)

            # Zapisanie przewidywań do pliku
            save_predictions_to_csv(predicted_prices, selected_currency)

    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")

# Funkcja do obsługi wątków
def start_refresh_plot_thread():
    selected_currency = currency_combobox.get()
    try:
        days_to_predict = int(days_entry.get())
        threading.Thread(target=refresh_plot, args=(selected_currency, days_to_predict)).start()
    except ValueError:
        messagebox.showerror("Błąd", "Podaj poprawną liczbę dni.")

# Funkcja do aktualizacji GUI
def update_gui():
    while not message_queue.empty():
        message = message_queue.get()
        messagebox.showinfo("Informacja", message)
    window.after(100, update_gui)

# Tworzenie GUI
window = tk.Tk()
window.title("LSTM do Przewidywania Ceny Kryptowalut")

frame = ttk.Frame(window)
frame.pack(pady=10)

currency_label = ttk.Label(frame, text="Wybierz kryptowalutę:")
currency_label.grid(row=0, column=0, padx=5, pady=5)

currency_combobox = ttk.Combobox(frame, values=['bitcoin', 'ethereum', 'dogecoin'])
currency_combobox.grid(row=0, column=1, padx=5, pady=5)

days_label = ttk.Label(frame, text="Liczba dni do przewidzenia:")
days_label.grid(row=1, column=0, padx=5, pady=5)

days_entry = ttk.Entry(frame)
days_entry.grid(row=1, column=1, padx=5, pady=5)

refresh_button = ttk.Button(frame, text="Przewiduj", command=start_refresh_plot_thread)
refresh_button.grid(row=2, columnspan=2, pady=10)

window.after(100, update_gui)
window.mainloop()
