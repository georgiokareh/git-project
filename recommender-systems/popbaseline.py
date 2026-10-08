import pandas as pd
import numpy as np

ratings = pd.read_csv("ratings.csv")

# sort by timestamp first


# pick a cutoff point — e.g. 80% of the way through, by row count
movies = pd.read_csv('movies.csv')
exploded = movies['genres'].str.split('|').explode()
unique_genres = exploded.unique()
for genre in unique_genres:
    movies[f'{genre}'] = (movies['genres'].str.contains(f'{genre}',regex = False)).astype(int)
categories = ['title','genres']
genres = movies.drop(categories, axis=1)
genres = genres.set_index('movieId')

ratings = ratings.sort_values(['timestamp','movieId'])
test = ratings.groupby('userId').tail(1) #leave one out of each group
train = ratings.drop(test.index)          # uses original row labels — correct, now
movie_stats = train.groupby('movieId')['rating'].agg(['mean', 'count'])
movie_stats.loc[movie_stats['count'] <= 20, 'mean'] = -np.inf

Y = train.pivot_table(index='movieId', columns='userId', values='rating')

pop_scores = movie_stats['mean'].reindex(genres.index).fillna(-np.inf)
pop_matrix = np.tile(pop_scores.values.reshape(-1, 1), (1, len(Y.columns)))

# mask movies each user already rated in train
movie_pos = genres.index.get_indexer(train['movieId'])
user_pos = Y.columns.get_indexer(train['userId'])
pop_matrix[movie_pos, user_pos] = -np.inf

# rank each user's column, best first
ranked = np.argsort(-pop_matrix, axis=0)
top_k = ranked[:50]

# positions for the held-out test ratings
test_movie_pos = genres.index.get_indexer(test['movieId'])
test_user_pos = Y.columns.get_indexer(test['userId'])

hits = []
for n in range(len(test)):
    hit = test_movie_pos[n] in top_k[:, test_user_pos[n]]
    hits.append(hit)

print("Hit rate@10:", np.mean(hits), "| users with a hit:", sum(hits), "of", len(hits))

#pop = train[train['count'] >= 50].sort_values('mean', ascending=False)
#movies = pd.read_csv("movies.csv")
#top10 = pop.head(10).join(movies.set_index('movieId'))
#print(top10[['title', 'mean', 'count']])
# how many users have fewer than 2 test rows
#print(test.groupby('userId').size().describe())
#train = train.reset_index()

#test = test.join(train.set_index('movieId'))
#test = test.reset_index()
#print(test.head(10)[['userId','movieId','rating','mean']])
#print(train[train['movieId']==5060])
#global_mean = ratings.drop(test.index)['rating'].mean()
#test['mean'] = test['mean'].fillna(global_mean) # fill NAN values
#rmse = np.zeros((len(test), 3))
#rmse[:,0] = test['rating']
#rmse[:,1] = test['mean']
#rmse[:,2] = (rmse[:,0]-rmse[:,1])**2
#rmse_nb = np.sqrt(np.sum(rmse[:,2])/len(test))
#print(f'RMSE: {rmse_nb}')

