import investpy
import pandas as pd
from datetime import datetime
import os

# Folder output CSV
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Daftar simbol crypto di Investing.com
CRYPTO_MAP = {
    "BTC": "Bitcoin",
}

# Rentang tanggal
START_DATE = "01/01/2019"
END_DATE = "01/01/2025"


def fetch_crypto_data(symbol_key):
    """
    Mengambil data historis dari Investing.com via investpy
    """
    crypto_name = CRYPTO_MAP[symbol_key]
    print(f"🔄 Mengambil data untuk {crypto_name}...")

    try:
        df = investpy.get_crypto_historical_data(
            crypto=crypto_name,
            from_date=START_DATE,
            to_date=END_DATE
        )

        # Rename kolom sesuai format yang diinginkan
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        df.index.name = 'timestamp'
        df.reset_index(inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"❌ Gagal mengambil data untuk {crypto_name}: {e}")
        return None


def save_to_csv(df, filename):
    """Menyimpan DataFrame ke file CSV."""
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    print(f"✅ Data disimpan ke {path}")


def main():
    print("📊 Mulai scraping data historis dari Investing.com...\n")
    for symbol in CRYPTO_MAP.keys():
        df = fetch_crypto_data(symbol)
        if df is not None:
            filename = f"{symbol.lower()}_usdt_historical.csv"
            save_to_csv(df, filename)
    print("\n🏁 Scraping selesai.")


if __name__ == "__main__":
    main()