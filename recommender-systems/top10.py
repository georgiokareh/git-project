import pandas as pd

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# aggregate
stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])

# require a minimum number of ratings so single-rating flukes don't dominate
min_ratings = 20
popular = stats[stats['count'] >= min_ratings].sort_values('mean', ascending=False)

top10 = popular.head(10).join(movies.set_index('movieId'))
print(top10[['title', 'mean', 'count']])