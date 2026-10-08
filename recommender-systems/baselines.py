import numpy as np
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask, hit_rate_at_k, predict_pairs
def main():
    #-------------random generated baseline-------------

    ratings, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    cand = candidate_mask(train, min_count=1, movie_ids=movie_ids)
    seed = 0
    rng = np.random.default_rng(seed)
    n = len(movie_ids)
    scores = rng.random((len(movie_ids), len(user_ids)))
    hits = hit_rate_at_k(scores, cv, train, cand, movie_ids, user_ids, k = 20)
    low, high = bootstrap_ci(hits, n_boot=1000, seed = 0)
    print("Random generated baseline")
    print(hits.mean(), hits.sum())
    print(low, high)
    print("----------------------------")
    #-------------most rated baseline-------------
    train_counts = train.groupby('movieId').size().reindex(movie_ids).fillna(0)
    scores = np.tile(train_counts.values[:, None], (1, len(user_ids)))
    hits = hit_rate_at_k(scores, cv, train, cand, movie_ids, user_ids, k = 20)
    low, high = bootstrap_ci(hits, n_boot=1000, seed = 0)
    print("Most rated baseline")
    print(hits.mean(), hits.sum())
    print(low, high)
    print('----------------------------------------')
    #-------------top rated baseline-------------
    train_mean =  train.groupby('movieId')['rating'].agg(['mean']).reindex(movie_ids).fillna(0)
    cand20 = candidate_mask(train, min_count=1, movie_ids=movie_ids)
    scores = np.tile(train_mean['mean'].values[:, None], (1, len(user_ids)))
    hits = hit_rate_at_k(scores, cv, train, cand20, movie_ids, user_ids, k = 20)
    low, high = bootstrap_ci(hits, n_boot=1000, seed = 0)
    print("Top rated baseline")
    print(hits.mean(), hits.sum())
    print(low, high)
    print('----------------------------------------')

    #-------------Global mean rmse-------------
    pred_matrix = np.full((len(movie_ids), len(user_ids)), train['rating'].mean())
    preds = predict_pairs(pred_matrix, cv, movie_ids, user_ids)
    print('Global mean RMSE')
    print(rmse(preds, cv['rating'].values))
    print('----------------------------------------')

    #-------------User mean rmse-------------
    user_mean_train = train.groupby('userId')['rating'].mean().reindex(user_ids)
    user_mean_array = user_mean_train.values.reshape(1,-1)
    pred_matrix = np.tile(user_mean_array, (len(movie_ids), 1))
    preds = predict_pairs(pred_matrix, cv, movie_ids, user_ids)
    print('User mean RMSE')
    print(rmse(preds, cv['rating'].values))
    print('----------------------------------------')

    #-------------Movie mean rmse-------------
    movie_mean_train = train.groupby('movieId')['rating'].mean().reindex(movie_ids).fillna( train['rating'].mean())
    pred_matrix = np.tile(movie_mean_train.values.reshape(-1, 1), (1, len(user_ids)))
    preds = predict_pairs(pred_matrix, cv, movie_ids, user_ids)
    print('Movie mean RMSE')
    print(rmse(preds, cv['rating'].values))
    print('----------------------------------------')

if __name__ == "__main__":
    main()
