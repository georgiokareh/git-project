import numpy as np
import pandas as pd
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask, hit_rate_at_k, predict_pairs
from contentbased import compute_cost, compute_gradients, grad_loop

def main():
    ratings, movie_ids, user_ids = load_data()
    movies = pd.read_csv('movies.csv')
    train, cv, test = make_split(ratings)
    exploded = movies['genres'].str.split('|').explode()
    unique_genres = exploded.unique()
    for genre in unique_genres:
        movies[f'{genre}'] = (movies['genres'].str.contains(f'{genre}',regex = False)).astype(int)
    categories = ['title','genres']
    genres = movies.drop(categories, axis=1)
    genres = genres.set_index('movieId')
    Y = train.pivot_table(index='movieId', columns='userId', values='rating').reindex(index=movie_ids, columns=user_ids)
    R = Y.notnull().astype(int)
    X = genres.reindex(movie_ids)
    n_features = X.shape[1]
    rng = np.random.default_rng(seed = 0)
    W = rng.normal(size=(len(user_ids), n_features)) * 0.01
    B = train.groupby('userId')['rating'].mean().reindex(user_ids).values.reshape(1, -1).copy()
    X_arr = X.values          # or X.to_numpy()
    Y_arr = Y.fillna(0).values
    R_arr = R.values
    assert X_arr.shape == (len(movie_ids), 20)
    assert Y_arr.shape == (len(movie_ids), len(user_ids))
    assert B.shape == (1, len(user_ids))
    w, b, J_hist =grad_loop(W, X_arr, B, Y_arr, 0.1, 2000, R_arr, 1)
    cand = candidate_mask(train, 1, movie_ids)
    pred_matrix = X_arr @ w.T + b
    preds = predict_pairs(pred_matrix, cv, movie_ids, user_ids)
    rmse_nb = rmse(preds, cv['rating'].values)
    hits = hit_rate_at_k(pred_matrix, cv, train, cand, movie_ids, user_ids, k = 20)
    print(f'RMSE :{rmse_nb}')
    print(f'Percentage : {hits.mean()}, total number : {hits.sum()}')

if __name__ == "__main__":
    main()


