import requests as r
import pandas as pd
try:
    r = r.get("https://api.open-meteo.com/v1/forecast?latitude=33.8938&longitude=35.5018&hourly=temperature_2m")
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    combined = df.groupby('hour')['temperature_2m'].agg(['sum', 'count']).rename(columns={
        'sum': 'total_temperature',
        'count': 'readings_count'
    })
    combined['average_temperature'] = (combined['total_temperature'] / combined['readings_count']).round(2)
    combined = combined.sort_values(by='average_temperature', ascending=False)
    print(combined['average_temperature'])
except r.exceptions.RequestException as e:
    print(f"Request failed: {e}")
