import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="BTC/USDT Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('btc_usdt_historical.csv')
        
        # Pastikan kolom timestamp dalam format datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Urutkan berdasarkan timestamp
        df = df.sort_values('timestamp')
        
        # Tambahkan fitur waktu
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Tambahkan pct_change kolom (persentase perubahan harga)
        df['pct_change'] = df['close'].pct_change()
        
        # Ekstrak tanggal saja untuk memudahkan pengelompokan
        df['date'] = df['timestamp'].dt.date
        
        # Tambahkan label untuk klasifikasi (positif jika close minggu depan > close saat ini)
        df['next_week_close'] = df['close'].shift(-7)
        df['label'] = (df['next_week_close'] > df['close']).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Fungsi untuk membuat fitur teknikal
def add_technical_features(df):
    # SMA (Simple Moving Average) - 7 dan 30 hari
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    
    # EMA (Exponential Moving Average) - 7 dan 30 hari
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # Bollinger Bands
    df['std_20'] = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['sma_30'] + (df['std_20'] * 2)
    df['bollinger_lower'] = df['sma_30'] - (df['std_20'] * 2)
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Jarak harga dari SMA
    df['price_sma7_ratio'] = df['close'] / df['sma_7']
    df['price_sma30_ratio'] = df['close'] / df['sma_30']
    
    # Volume features
    df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_7']
    
    return df

# Fungsi untuk mempersiapkan data pelatihan
def prepare_training_data(df, start_date, end_date):
    # Definisikan fitur yang akan digunakan (tetap)
    selected_features = [
        'open', 'high', 'low', 'close', 'volume', 'pct_change', 
        'day_of_week', 'month', 'week_of_year', 'sma_7', 'sma_30', 
        'ema_7', 'ema_30', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'price_sma7_ratio', 'price_sma30_ratio', 'volume_sma_7', 
        'volume_sma_30', 'volume_ratio'
    ]
    
    # Filter data berdasarkan rentang tanggal
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    train_df = df.loc[mask].copy()
    
    # Drop NaN yang mungkin muncul karena fitur teknikal
    train_df = train_df.dropna()
    
    # Memisahkan fitur dan label
    X = train_df[selected_features]
    y = train_df['label']
    
    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, selected_features

# Fungsi untuk melatih model
def train_model(X, y, model_type, param_grid=None):
    if model_type == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
    elif model_type == 'SVM':
        base_model = SVC(probability=True, random_state=42)
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            
    elif model_type == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 10]
            }
    
    # Grid search untuk tuning hyperparameter
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Fungsi untuk melakukan backtest
def perform_backtest(df, model, scaler, selected_features, start_date, end_date, entry_day):
    # Filter data berdasarkan rentang tanggal
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    test_df = df.loc[mask].copy()
    
    # Filter untuk hari masuk (entry day) yang dipilih
    day_map = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
    test_df = test_df[test_df['day_of_week'] == day_map[entry_day]].copy()
    
    if test_df.empty:
        return None, None, None, None, None, None
    
    # Persiapkan data untuk prediksi
    X_test = test_df[selected_features]
    X_test_scaled = scaler.transform(X_test)
    y_true = test_df['label']
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilitas kelas positif
    
    # Hitung metrik
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Win rate (% prediksi benar dari total prediksi)
    win_rate = sum(y_pred == y_true) / len(y_true) if len(y_true) > 0 else 0
    
    # Tambahkan hasil prediksi ke dataframe
    test_df['prediction'] = y_pred
    test_df['prediction_proba'] = y_pred_proba
    test_df['correct'] = y_pred == y_true
    
    # Hitung win rate per minggu
    test_df['week'] = test_df['timestamp'].dt.isocalendar().week
    test_df['year'] = test_df['timestamp'].dt.year
    weekly_results = test_df.groupby(['year', 'week']).agg(
        correct_count=('correct', 'sum'),
        total_count=('correct', 'count')
    )
    weekly_results['win_rate'] = weekly_results['correct_count'] / weekly_results['total_count']
    
    return accuracy, precision, recall, f1, win_rate, test_df

# Fungsi untuk membuat visualisasi interaktif
def create_visualization(df, backtest_results=None):
    # Buat subplot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('BTC/USDT Price', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    # Tambahkan candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Tambahkan garis SMA
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['sma_7'],
            name='SMA 7',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['sma_30'],
            name='SMA 30',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # Tambahkan volume
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # Jika ada hasil backtest, tambahkan penanda buy/sell
    if backtest_results is not None and not backtest_results.empty:
        # Tandai prediksi positif
        positive_predictions = backtest_results[backtest_results['prediction'] == 1]
        correct_positive = positive_predictions[positive_predictions['correct']]
        incorrect_positive = positive_predictions[~positive_predictions['correct']]
        
        # Tandai prediksi benar dengan warna hijau
        fig.add_trace(
            go.Scatter(
                x=correct_positive['timestamp'],
                y=correct_positive['close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Correct Positive Prediction'
            ),
            row=1, col=1
        )
        
        # Tandai prediksi salah dengan warna merah
        fig.add_trace(
            go.Scatter(
                x=incorrect_positive['timestamp'],
                y=incorrect_positive['close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-up'),
                name='Incorrect Positive Prediction'
            ),
            row=1, col=1
        )
        
        # Tandai prediksi negatif
        negative_predictions = backtest_results[backtest_results['prediction'] == 0]
        correct_negative = negative_predictions[negative_predictions['correct']]
        incorrect_negative = negative_predictions[~negative_predictions['correct']]
        
        # Tandai prediksi benar dengan warna hijau
        fig.add_trace(
            go.Scatter(
                x=correct_negative['timestamp'],
                y=correct_negative['close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-down'),
                name='Correct Negative Prediction'
            ),
            row=1, col=1
        )
        
        # Tandai prediksi salah dengan warna merah
        fig.add_trace(
            go.Scatter(
                x=incorrect_negative['timestamp'],
                y=incorrect_negative['close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Incorrect Negative Prediction'
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title='BTC/USDT Historical Price with Predictions',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Fungsi untuk mendapatkan tanggal Senin minggu depan
def get_next_monday(date):
    # Dapatkan hari dalam seminggu (0=Senin, 6=Minggu)
    day_of_week = date.weekday()
    
    # Hitung berapa hari ke Senin berikutnya
    days_until_monday = 7 - day_of_week if day_of_week > 0 else 7
    
    # Dapatkan tanggal Senin berikutnya
    next_monday = date + timedelta(days=days_until_monday)
    
    return next_monday

# Fungsi untuk plotting win rate per minggu
def plot_weekly_winrate(backtest_results):
    if backtest_results is None or backtest_results.empty:
        return None
    
    weekly_results = backtest_results.groupby(['year', 'week']).agg(
        correct_count=('correct', 'sum'),
        total_count=('correct', 'count')
    )
    weekly_results['win_rate'] = weekly_results['correct_count'] / weekly_results['total_count']
    
    # Buat indeks tanggal dari year dan week
    dates = []
    for idx, row in weekly_results.iterrows():
        year, week = idx
        # Konversi year-week ke tanggal (menggunakan hari pertama dari minggu tersebut)
        try:
            date = datetime.strptime(f'{year}-W{week}-1', '%Y-W%W-%w')
            dates.append(date)
        except ValueError:
            # Untuk menangani kasus khusus seperti week 53
            date = datetime.strptime(f'{year}-12-31', '%Y-%m-%d')
            dates.append(date)
    
    weekly_results['date'] = dates
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_results['date'],
        y=weekly_results['win_rate'],
        name='Weekly Win Rate',
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='Win Rate per Minggu',
        xaxis_title='Tanggal',
        yaxis_title='Win Rate',
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

# Fungsi untuk membuat prediksi untuk tanggal tertentu
def make_prediction(model, scaler, selected_features, df, selected_date):
    # Convert selected_date to datetime if it's not already
    if not isinstance(selected_date, datetime):
        selected_date = pd.to_datetime(selected_date)
    
    # Filter data untuk mendapatkan baris dengan tanggal yang dipilih
    date_mask = df['timestamp'].dt.date == selected_date.date()
    if not any(date_mask):
        return None, None, None
    
    selected_row = df[date_mask].iloc[0]
    
    # Persiapkan data untuk prediksi
    X_pred = selected_row[selected_features].values.reshape(1, -1)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Lakukan prediksi
    prediction = model.predict(X_pred_scaled)[0]
    prediction_proba = model.predict_proba(X_pred_scaled)[0]
    
    # Dapatkan tanggal Senin berikutnya
    next_monday = get_next_monday(selected_date)
    
    return prediction, prediction_proba, next_monday

# Fungsi untuk menyimpan model terlatih
def save_model(model, scaler, selected_features):
    # Buat direktori 'models' jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Simpan model
    joblib.dump(model, 'models/btc_prediction_model.joblib')
    
    # Simpan scaler
    joblib.dump(scaler, 'models/btc_prediction_scaler.joblib')
    
    # Simpan selected features
    with open('models/btc_prediction_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    return True

# Fungsi untuk memuat model terlatih
def load_model():
    try:
        model = joblib.load('models/btc_prediction_model.joblib')
        scaler = joblib.load('models/btc_prediction_scaler.joblib')
        
        with open('models/btc_prediction_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        
        return model, scaler, selected_features
    except:
        return None, None, None

# Main App
def main():
    # Sidebar untuk navigasi
    st.sidebar.title("BTC/USDT Forecast")
    
    # Tabs untuk navigasi
    tab1, tab2 = st.tabs(["User Interface", "Model Settings"])
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Gagal memuat data. Pastikan file CSV tersedia.")
        return
    
    # Tambahkan fitur teknikal
    df = add_technical_features(df)
    
    # Dapatkan nilai min dan max dari timestamp
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    with tab2:
        st.header("Model Settings")
        
        # Parameter model
        st.subheader("Model Parameters")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "SVM", "Gradient Boosting"],
            key="model_type"
        )
        
        # Periode pelatihan
        st.subheader("Training Period")
        
        col1, col2 = st.columns(2)
        with col1:
            train_start_date = st.date_input(
                "Start Date",
                min_value=min_date,
                max_value=max_date - timedelta(days=30),
                value=min_date,
                key="train_start_date"
            )
        
        with col2:
            train_end_date = st.date_input(
                "End Date",
                min_value=min_date + timedelta(days=30),
                max_value=max_date,
                value=max_date - timedelta(days=90),
                key="train_end_date"
            )
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        
        auto_tune = st.checkbox("Auto-tune hyperparameters", value=True, key="auto_tune")
        
        param_grid = None
        if not auto_tune:
            st.write("Custom Hyperparameters:")
            
            if model_type == "Random Forest":
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
                with col2:
                    max_depth = st.slider("Max depth", 1, 50, 10, 1)
                with col3:
                    min_samples_split = st.slider("Min samples split", 2, 20, 2, 1)
                
                param_grid = {
                    'n_estimators': [n_estimators],
                    'max_depth': [max_depth],
                    'min_samples_split': [min_samples_split]
                }
                
            elif model_type == "SVM":
                col1, col2, col3 = st.columns(3)
                with col1:
                    C = st.select_slider("C", options=[0.1, 1.0, 10.0, 100.0], value=1.0)
                with col2:
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)
                with col3:
                    gamma = st.select_slider("Gamma", options=["scale", "auto"], value="scale")
                
                param_grid = {
                    'C': [C],
                    'kernel': [kernel],
                    'gamma': [gamma]
                }
                
            elif model_type == "Gradient Boosting":
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
                with col2:
                    learning_rate = st.select_slider("Learning rate", options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.1)
                with col3:
                    max_depth = st.slider("Max depth", 1, 20, 3, 1)
                
                param_grid = {
                    'n_estimators': [n_estimators],
                    'learning_rate': [learning_rate],
                    'max_depth': [max_depth]
                }
        
        # Train button
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                # Prepare data
                X, y, scaler, selected_features = prepare_training_data(
                    df, 
                    pd.Timestamp(train_start_date), 
                    pd.Timestamp(train_end_date)
                )
                
                # Train model
                model, best_params = train_model(X, y, model_type, param_grid)
                
                # Save model
                save_model(model, scaler, selected_features)
                
                st.success(f"Model trained successfully with parameters: {best_params}")
                
                # Display feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.subheader("Feature Importance")
                    fig = go.Figure(go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Feature'],
                        orientation='h'
                    ))
                    
                    fig.update_layout(
                        title='Feature Importance',
                        xaxis_title='Importance',
                        yaxis_title='Feature',
                        height=500
                    )
                    
                    st.plotly_chart(fig)
    
    with tab1:
        st.header("BTC/USDT Prediction Interface")
        
        # Check if model exists
        model, scaler, model_features = load_model()
        
        if model is None:
            st.warning("No trained model found. Please go to Model Settings tab to train a model first.")
        else:
            st.success(f"Model loaded successfully.")
            
            # UI untuk tanggal prediksi
            st.subheader("Make Prediction")
            
            selected_date = st.date_input(
                "Select Date for Prediction",
                min_value=min_date,
                max_value=max_date - timedelta(days=7),
                value=max_date - timedelta(days=7),
                key="prediction_date"
            )
            
            # Lakukan prediksi
            prediction, prediction_proba, next_monday = make_prediction(
                model, scaler, model_features, df, pd.Timestamp(selected_date)
            )
            
            if prediction is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date Selected:** {selected_date}")
                    st.write(f"**Target Monday:** {next_monday.date()}")
                    
                    if prediction == 1:
                        st.markdown(f"<h3 style='color:green'>Prediction: POSITIF (↑)</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color:red'>Prediction: NEGATIF (↓)</h3>", unsafe_allow_html=True)
                    
                    st.write(f"**Confidence:** {prediction_proba[prediction]:.2%}")
                
                with col2:
                    # Visualisasi probabilitas
                    fig = go.Figure(go.Bar(
                        x=['NEGATIF', 'POSITIF'],
                        y=[prediction_proba[0], prediction_proba[1]],
                        marker_color=['red', 'green']
                    ))
                    
                    fig.update_layout(
                        title='Prediction Probability',
                        yaxis=dict(tickformat='.0%'),
                        height=300
                    )
                    
                    st.plotly_chart(fig)
            
            # Backtest
            st.subheader("Backtest")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                backtest_start_date = st.date_input(
                    "Backtest Start Date",
                    min_value=min_date,
                    max_value=max_date - timedelta(days=30),
                    value=max_date - timedelta(days=365),
                    key="backtest_start_date"
                )
            
            with col2:
                backtest_end_date = st.date_input(
                    "Backtest End Date",
                    min_value=min_date + timedelta(days=30),
                    max_value=max_date,
                    value=max_date,
                    key="backtest_end_date"
                )
            
            with col3:
                entry_day = st.selectbox(
                    "Entry Day",
                    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
                    index=0,
                    key="entry_day"
                )
            
            if st.button("Run Backtest", key="backtest_button"):
                with st.spinner("Running backtest..."):
                    # Lakukan backtest
                    accuracy, precision, recall, f1, win_rate, backtest_results = perform_backtest(
                        df, model, scaler, model_features, 
                        pd.Timestamp(backtest_start_date), 
                        pd.Timestamp(backtest_end_date),
                        entry_day
                    )
                    
                    if backtest_results is not None and not backtest_results.empty:
                        # Tampilkan metrik backtest
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Precision", f"{precision:.2%}")
                        with col3:
                            st.metric("Recall", f"{recall:.2%}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.2%}")
                        with col5:
                            st.metric("Win Rate", f"{win_rate:.2%}")
                        
                        # Visualisasi backtest
                        st.subheader("Backtest Visualization")
                        
                        # Tampilkan grafik harga dengan prediksi
                        backtest_chart = create_visualization(df[df['timestamp'].between(
                            pd.Timestamp(backtest_start_date), 
                            pd.Timestamp(backtest_end_date)
                        )], backtest_results)
                        
                        st.plotly_chart(backtest_chart, use_container_width=True)
                        
                        # Tampilkan win rate per minggu
                        st.subheader("Weekly Win Rate")
                        weekly_winrate_chart = plot_weekly_winrate(backtest_results)
                        st.plotly_chart(weekly_winrate_chart, use_container_width=True)
                        
                        # Tampilkan hasil prediksi dalam tabel
                        st.subheader("Prediction Results")
                        
                        # Format the results table
                        display_columns = ['timestamp', 'close', 'prediction', 'prediction_proba', 'correct', 'next_week_close']
                        display_df = backtest_results[display_columns].copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.date
                        display_df['prediction'] = display_df['prediction'].map({1: 'POSITIF', 0: 'NEGATIF'})
                        display_df['prediction_proba'] = display_df['prediction_proba'].map(lambda x: f"{x:.2%}")
                        display_df['correct'] = display_df['correct'].map({True: '✓', False: '✗'})
                        display_df.columns = ['Tanggal', 'Harga Penutupan', 'Prediksi', 'Probabilitas', 'Benar', 'Harga Minggu Depan']
                        
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.warning("No data available for the selected backtest period and entry day.")

# Run the app
if __name__ == "__main__":
    main()