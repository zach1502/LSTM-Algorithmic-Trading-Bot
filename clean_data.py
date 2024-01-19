
import os
import pandas as pd

chunk_size = 10 ** 6
output_file = './btc_usdt_data/full_btc_usdt_data_cleaned.csv'
headers = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

def clean_data(df):
    df = df.dropna()
    df = df.drop(columns=['Ignore', 'Close time'])
    return df

i=0
for chunk in pd.read_csv('./btc_usdt_data/full_btc_usdt_data.csv', chunksize=chunk_size):
    cleaned_chunk = clean_data(chunk)
    cleaned_chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    i+=len(cleaned_chunk)
    print(f'Processed {i} rows')
