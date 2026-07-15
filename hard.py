import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_= pd.read_csv("results.csv")
df = df_.dropna( subset=['home_score', 'away_score'] )
df_home = df[['home_team', 'home_score']].rename(columns={
    'home_team': 'team',
    'home_score': 'goals'
})
df_away = df[['away_team', 'away_score']].rename(columns={
    'away_team': 'team',
    'away_score': 'goals'
})
combined = pd.concat([df_home, df_away], ignore_index=True)
group = combined.groupby('team')['goals'].agg(['sum', 'count']).rename(columns={
    'sum': 'total_goals',
    'count': 'matches_played'
})
groupClean = group[group['matches_played'] > 200]
df_filtered = df[
    df['home_team'].isin(groupClean.index) &
    df['away_team'].isin(groupClean.index)
]
df_home = df_filtered[['home_team', 'home_score', 'away_score']].copy()
df_home['is_win'] = (df_home['home_score'] > df_home['away_score']).astype(int)
df_home = df_home[['home_team', 'is_win']].rename(columns={'home_team': 'team'})

df_away = df_filtered[['away_team', 'home_score', 'away_score']].copy()
df_away['is_win'] = (df_away['away_score'] > df_away['home_score']).astype(int)
df_away = df_away[['away_team', 'is_win']].rename(columns={'away_team': 'team'})
combined = pd.concat([df_home, df_away], ignore_index=True)
group = combined.groupby('team')['is_win'].agg(['sum', 'count']).rename(columns={
    'sum': 'total_wins',
    'count': 'matches_played'
})
groupClean = group[group['matches_played'] > 50]
groupClean['win_percentage'] = (groupClean['total_wins'] / groupClean['matches_played']).round(2)
groupClean = groupClean.sort_values(by='win_percentage', ascending=False)
print(groupClean['win_percentage'].head(10).round(2))
top10 = groupClean['win_percentage'].head(10).sort_values()  # ascending for barh — best at top

plt.figure(figsize=(8, 5))
plt.barh(top10.index, top10.values, color='steelblue')
plt.xlabel('Win rate')
plt.title('Top 10 teams by win rate vs qualified opposition')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()