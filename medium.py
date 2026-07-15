import requests
import csv
import os 

try:
    key = os.environ.get('EXCHANGERATE_API_KEY')
    r = requests.get(f"https://v6.exchangerate-api.com/v6/{key}/latest/USD", timeout = 10)
    r.raise_for_status()
    data = r.json()  # Parse the JSON response
    with open('exchange_rates.csv', mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Currency', 'Rate'])
        writer.writeheader()
        for currency, rate in data['conversion_rates'].items():
            if rate > 10000:
                writer.writerow({'Currency': currency, 'Rate': rate})
    print("Exchange rates saved to exchange_rates.csv")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")