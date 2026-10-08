import numpy as np
import pandas as pd
import tensorflow as tf
from common import (rmse, bootstrap_ci, load_data, make_split, candidate_mask,
                    hit_rate_at_k, predict_pairs)
# replace YOUR_CF_FILE with the name of your Chapter 4 file (without .py)
from collaborative_filtering import get_variables, normalization, train_model, get_shrunk_mean

# ---- WRITE YOUR PREDICTIONS HERE BEFORE RUNNING, THEN DO NOT EDIT ----
# Test RMSE model: ...            baseline: ...
# hit@20 model: ...               most-rated: ...
# Falsifier: ...

# Locked from cv. Do not change after seeing test numbers.
K, LAMBDA_, LR, ITERS, SHRINK_LAM = 10, 1e-4, 0.01, 200, 10


def rmse_with_ci(name, preds, truth):
    sq = (preds - truth) ** 2
    value = rmse(preds, truth)
    assert abs(np.sqrt(sq.mean()) - value) < 1e-6
    low, high = bootstrap_ci(sq, n_boot=1000, seed=0)
    print(f'{name}: RMSE = {value:.4f}  95% CI [{np.sqrt(low):.4f}, {np.sqrt(high):.4f}]')
    return sq


def hits_with_ci(name, hits):
    low, high = bootstrap_ci(hits, n_boot=1000, seed=0)
    print(f'{name}: hit@20 = {hits.mean() * 100:.2f}% ({int(hits.sum())}/{len(hits)})  '
          f'95% CI [{low * 100:.2f}%, {high * 100:.2f}%]')


def paired_hits(name, hits_a, hits_b):
    diff = hits_a.astype(int) - hits_b.astype(int)
    low, high = bootstrap_ci(diff, n_boot=1000, seed=0)
    print(f'{name}: net = {diff.sum()}, only first hit = {(diff == 1).sum()}, '
          f'only second hit = {(diff == -1).sum()}, CI of mean diff [{low:.4f}, {high:.4f}]')


def main():
    ratings, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)

    # ---- data setup (train only) ----
    y_train = (train.pivot_table(index='movieId', columns='userId', values='rating')
               .reindex(index=movie_ids, columns=user_ids))
    r_train = y_train.notnull().astype(int)
    y_train = y_train.fillna(0)
    assert y_train.shape == r_train.shape == (len(movie_ids), len(user_ids))
    assert not train.duplicated(['movieId', 'userId']).any()
    y_arr = y_train.to_numpy(dtype=np.float32)
    r_arr = r_train.to_numpy(dtype=bool)
    assert r_arr.sum() == len(train)
    y_norm, movie_mean, global_mean = normalization(train, movie_ids, y_arr, r_arr)

    # ---- train the factor model (same settings and seed as on cv) ----
    x, w, b = get_variables(user_ids, movie_ids, k=K)
    x, w, b, j_hist, rmse_hist = train_model(x, w, b, y_norm, r_arr, LAMBDA_, lr=LR, iterations=ITERS)

    # ---- score matrices ----
    pred_model = (x @ tf.transpose(w) + b).numpy() + movie_mean          # for RMSE
    assert (r_arr.sum(axis=0) > 0).all()
    user_offset = y_norm.sum(axis=0) / r_arr.sum(axis=0).reshape(1, -1)
    pred_base = movie_mean + user_offset                                  # for RMSE
    for m in (pred_model, pred_base):
        assert m.shape == (len(movie_ids), len(user_ids)) and np.isfinite(m).all()

    # ---- reproduction check on cv (should be about 0.9139 and 0.9199) ----
    truth_cv = cv['rating'].values
    print('--- reproduction check on cv ---')
    rmse_with_ci('model (cv)', predict_pairs(pred_model, cv, movie_ids, user_ids), truth_cv)
    rmse_with_ci('baseline (cv)', predict_pairs(pred_base, cv, movie_ids, user_ids), truth_cv)

    # ---- TEST: RMSE ----
    truth_test = test['rating'].values
    print('--- TEST RMSE ---')
    sq_model = rmse_with_ci('model', predict_pairs(pred_model, test, movie_ids, user_ids), truth_test)
    sq_base = rmse_with_ci('baseline', predict_pairs(pred_base, test, movie_ids, user_ids), truth_test)
    diff = sq_base - sq_model                      # positive = model closer
    low, high = bootstrap_ci(diff, n_boot=1000, seed=0)
    print(f'paired (baseline - model), mean squared error diff = {diff.mean():.4f}, '
          f'CI [{low:.4f}, {high:.4f}]')

    # ---- TEST: hit@20 ----
    seen = pd.concat([train, cv], ignore_index=True)
    assert len(seen) == len(train) + len(cv)
    assert not seen.duplicated(['movieId', 'userId']).any()
    cand = candidate_mask(train, 1, movie_ids)

    shrunk_mean = get_shrunk_mean(global_mean, train, SHRINK_LAM, movie_ids)
    rank_model = (x @ tf.transpose(w)).numpy() + shrunk_mean
    rank_ctrl = np.repeat(shrunk_mean, len(user_ids), axis=1)
    counts = r_arr.sum(axis=1, keepdims=True).astype(np.float32)
    rank_pop = np.repeat(counts, len(user_ids), axis=1)
    for m in (rank_model, rank_ctrl, rank_pop):
        assert m.shape == (len(movie_ids), len(user_ids)) and np.isfinite(m).all()

    hits_model = hit_rate_at_k(rank_model, test, seen, cand, movie_ids, user_ids, k=20)
    hits_ctrl = hit_rate_at_k(rank_ctrl, test, seen, cand, movie_ids, user_ids, k=20)
    hits_pop = hit_rate_at_k(rank_pop, test, seen, cand, movie_ids, user_ids, k=20)

    ceiling = int(cand[movie_ids.get_indexer(test['movieId'])].sum())
    print(f'--- TEST hit@20 (ceiling: {ceiling} of {len(user_ids)} users reachable) ---')
    hits_with_ci('model (shrunk mean + factors)', hits_model)
    hits_with_ci('control (shrunk mean only)', hits_ctrl)
    hits_with_ci('most-rated', hits_pop)
    paired_hits('model vs control', hits_model, hits_ctrl)
    paired_hits('model vs most-rated', hits_model, hits_pop)


if __name__ == '__main__':
    main()