import pandas as pd
import numpy as np
def make_split(ratings: pd.DataFrame):
    assert ratings.index.is_unique
    ratings = ratings.sort_values(['timestamp','movieId'])
    test = ratings.groupby('userId').tail(1)
    train = ratings.drop(test.index)  
    cv = train.groupby('userId').tail(1)
    train = train.drop(cv.index)
    return train, cv, test

def candidate_mask(train: pd.DataFrame, min_count: int, movie_ids: list):
    counts = train.groupby('movieId').size()
    counts = counts.reindex(movie_ids).fillna(0)
    assert list(counts.index) == list(movie_ids)
    cand = (counts >= min_count).values
    return cand

def hit_rate_at_k(scores, held_out, seen, cand, movie_ids, user_ids, k):
    assert scores.shape == (len(movie_ids), len(user_ids))
    scores = scores.copy()
    seen_movie = movie_ids.get_indexer(seen['movieId'])
    seen_user = user_ids.get_indexer(seen['userId'])
    assert (seen_movie >= 0).all()
    assert (seen_user >= 0).all()
    scores[seen_movie, seen_user] = -np.inf
    scores[~cand, :] = -np.inf
    scores_ranked = np.argsort(-scores, axis = 0, kind='stable')
    test_movie_pos = movie_ids.get_indexer(held_out['movieId'])
    test_user_pos = user_ids.get_indexer(held_out['userId'])
    assert (test_movie_pos >= 0).all()
    assert (test_user_pos >= 0).all()
    top_k = scores_ranked[:k]
    assert (np.isfinite(scores).sum(axis=0) >= k).all()
    hits = (top_k[:, test_user_pos] == test_movie_pos).any(axis=0)
    return hits
def bootstrap_ci(values, n_boot=1000, seed = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, len(values)))   # shape (1000, 610): each entry is a position 0..609
    samples = values[idx]                         # shape (1000, 610): the hit/miss at each of those positions
    boot_means = samples.mean(axis=1)             # shape (1000,): one hit rate per pretend group
    boot_means_sorted = np.sort(boot_means)
    low  = boot_means_sorted[int(0.025 * n_boot)]     # position 25
    high = boot_means_sorted[int(0.975 * n_boot)]     # position 975
    return low, high
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    movie_ids = pd.Index(movies['movieId'])
    user_ids = pd.Index(np.sort(ratings['userId'].unique()))
    assert len(movie_ids) == 9742 and len(user_ids) == 610
    assert movie_ids.is_unique and user_ids.is_unique
    return ratings, movie_ids, user_ids
def rmse(preds, truth):
    preds = np.asarray(preds)
    truth = np.asarray(truth)
    assert not np.isnan(preds).any() and not np.isnan(truth).any()
    assert len(preds) == len(truth)
    err = (preds - truth) ** 2
    rmse = np.sqrt(np.sum(err)/len(err))
    return rmse
def predict_pairs(pred_matrix, held_out, movie_ids, user_ids):
    assert pred_matrix.shape == (len(movie_ids),len(user_ids))
    user_pos = user_ids.get_indexer(held_out['userId'])
    movie_pos = movie_ids.get_indexer(held_out['movieId'])
    assert (user_pos >= 0).all() and (movie_pos >= 0).all()
    preds = pred_matrix[movie_pos, user_pos]
    assert len(preds) == len(held_out)
    return preds
def main():
    ratings, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    cand = candidate_mask(train, min_count=1, movie_ids=movie_ids)

    # --- oracle test ---
    scores = np.zeros((len(movie_ids), len(user_ids)))        # 1. build the matrix
    mpos = movie_ids.get_indexer(cv['movieId'])               # 2. positions of the held-out movies
    upos = user_ids.get_indexer(cv['userId'])
    scores[mpos, upos] = 1e9                                  # 3. give them huge scores
    hits = hit_rate_at_k(scores, cv, train, cand, movie_ids, user_ids, 20)   # 4. run and store
    print("oracle:", hits.mean(), hits.sum())                 # 5. expect 1.0 and 610

    # --- random test ---
    scores = np.random.default_rng(0).random((len(movie_ids), len(user_ids)))
    hits = hit_rate_at_k(scores, cv, train, cand, movie_ids, user_ids, 20)
    print("random:", hits.mean(), hits.sum())                 # expect roughly 20 / 9700 = 0.2%
    print(cand[mpos].sum())
if __name__ == "__main__":
    main()