import two_tower as idm
import two_tower_part2 as genm
import two_tower_part3 as mlpm
from common import bootstrap_ci, load_data, make_split, candidate_mask
import pandas as pd
import numpy as np
import tensorflow as tf
def prepare():
    ratings, movies, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    train['user_row'] = user_ids.get_indexer(train['userId'])
    train['movie_row'] = movie_ids.get_indexer(train['movieId'])
    cv['user_row'] = user_ids.get_indexer(cv['userId'])
    cv['movie_row'] = movie_ids.get_indexer(cv['movieId'])
    test['user_row'] = user_ids.get_indexer(test['userId'])
    test['movie_row'] = movie_ids.get_indexer(test['movieId'])
    for name, df in [('train', train), ('cv', cv), ('test', test)]:
        assert not (df['user_row'].values == -1).any()
        assert not (df['movie_row'].values == -1).any()
    overlap_seen = train.merge(cv, on=['user_row', 'movie_row'])
    assert len(overlap_seen) == 0, f"{len(overlap_seen)} pairs appear in both train and cv"
    seen = pd.concat([train, cv], ignore_index = True)
    overlap = seen.merge(test, on=['user_row', 'movie_row'])
    assert len(overlap) == 0, f"{len(overlap)} pairs appear in both seen and test"
    seen_by_user = seen.groupby('userId')['timestamp'].max()
    assert test['userId'].is_unique
    assert len(test) == len(user_ids)
    test_indexed = test.set_index('userId')['timestamp']
    assert (test_indexed >= seen_by_user.reindex(test_indexed.index)).all()
    g = idm.get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    assert g.shape == (len(movie_ids), 20)
    cand = candidate_mask(train, 1, movie_ids)
    counts = train['movie_row'].value_counts().reindex(range(len(movie_ids)), fill_value=0).to_numpy()
    log_q = idm.get_log_q(counts, len(train))
    return train, cv, test, seen, g, cand, counts, log_q, movie_ids, user_ids
def run_model(model, seed, train, cv, test, seen, g, cand, counts, log_q, movie_ids, user_ids):
    pop_scores = np.tile(counts[:, None], (1, len(user_ids))).astype(float)
    pop_hits, pop_sum = idm.evaluate(pop_scores, train, cv, cand, movie_ids, user_ids)
    assert pop_sum == 35
    user_table, movie_table, genre_weights = idm.get_variables(user_ids, movie_ids, g.shape[1], length=8, seed=seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    if model == 'id':
        means, eval_hist, hits = idm.training_loop(train, user_table, movie_table, optimizer, 256, 15, seed * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        scores = idm.get_full_score(user_table, movie_table, user_ids, movie_ids)
    elif model == 'gen':
        means, eval_hist, hits = genm.training_loop(train, user_table, movie_table, genre_weights, g, optimizer, 256, 15, seed * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        scores = genm.get_full_score (user_table, movie_table, genre_weights, g, user_ids, movie_ids)
    elif model == 'mlp':
        w1, w2 = mlpm.get_mlp_variables(g.shape[1], 32, 8, seed + 1000)
        mlps = [w1,w2]
        means, eval_hist, hits = mlpm.training_loop(train, user_table, movie_table, g, mlps, optimizer, 256, 15, seed * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        scores = mlpm.get_full_score(user_table, movie_table, g, mlps, user_ids, movie_ids)
    else: raise ValueError(f"unknown model {model}")
    test_hits, test_sum = idm.evaluate(scores, seen, test, cand, movie_ids, user_ids)
    return eval_hist[-1], test_hits.astype(int)
if __name__ == "__main__":
    train, cv, test, seen, g, cand, counts, log_q, movie_ids, user_ids = prepare()
    pop_scores = np.tile(counts[:, None], (1, len(user_ids))).astype(float)
    pop_hits, pop_sum = idm.evaluate(pop_scores, seen, test, cand, movie_ids, user_ids)
    assert pop_sum == 43
    test_results = {model: {} for model in ['id', 'gen', 'mlp']}
    cv_results = {model: {} for model in ['id', 'gen', 'mlp']}
    expected_cv = {'id': [79, 68, 64], 'gen': [76, 66, 72], 'mlp': [76, 70, 69]}
    print(f'Popularity hits = {pop_sum}')
    for model in ['id', 'gen', 'mlp']:
        for seed in range(3):
            cv_total, test_hits = run_model(model, seed, train, cv, test, seen, g, cand, counts, log_q, movie_ids, user_ids)
            assert cv_total == expected_cv[model][seed], f"{model} seed {seed}: got {cv_total}"
            test_results[model][seed] = test_hits 
            cv_results[model][seed] = cv_total
    for model in ['id', 'gen', 'mlp']:
        print(f'Model = {model}')
        np.savez(f'test_hits_{model}.npz', **{f'seed{s}': test_results[model][s] for s in range(3)})
        for seed in range(3):
            print(f'Seed = {seed}')
            print(f'CV hits = {cv_results[model][seed]}')
            print(f'Test hits = {test_results[model][seed].sum()}')
            
    
