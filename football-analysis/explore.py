import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("results.csv")
df_clean = df.dropna( subset=['home_score', 'away_score'] )
nb_wc= len(df_clean[df_clean['tournament'] == 'FIFA World Cup'])
nb_friendly = len(df_clean[df_clean['tournament'] == 'Friendly'])
print(f"Number of matches in the FIFA World Cup: {nb_wc}")
print(f"Number of matches in Friendly: {nb_friendly}")

type = ['FIFA World Cup', 'Friendly']
counts = [nb_wc, nb_friendly]
plt.bar(type, counts, color=['blue', 'orange'])

plt.xlabel('Tournament Type')
plt.ylabel('Number of Matches')
plt.title('Number of Matches by Tournament Type')
plt.yticks(np.arange(0, 20000, 1000))
plt.show()