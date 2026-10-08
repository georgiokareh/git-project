import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")
df_clean = df.dropna( subset=['home_score', 'away_score'] )
goals = df_clean.groupby('home_team')['home_score'].sum() + df_clean.groupby('away_team')['away_score'].sum()
sorted_goals = goals.sort_values(ascending=False)
print("Top 10 teams with the most goals scored:")
print(sorted_goals.head(10))
teams = sorted_goals.head(10).index
goals_scored = sorted_goals.head(10).values
plt.bar(teams, goals_scored, color='green')
plt.xlabel('Team')
plt.ylabel('Number of goals')
plt.title('Number of Goals by Team')
plt.yticks(np.arange(1000 ,3000, 200))
plt.xticks(rotation=45, ha='right')
plt.show()