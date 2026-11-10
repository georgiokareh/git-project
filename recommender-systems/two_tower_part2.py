from two_tower import calculate_loss, evaluate, diagnostic, average_top20_count, get_log_q, get_genres_matrix, get_variables
from two_tower import get_score_matrix as old_get_score_matrix
from two_tower import get_full_score as old_get_full_score
import numpy as np
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask, hit_rate_at_k, predict_pairs
import tensorflow as tf
import pandas as pd
import time
def get_movie_vectors(movie_table, g, genre_weights, rows = None):
    if rows is None:
        genres_table = tf.matmul(g, genre_weights)
        movie_vectors = genres_table + movie_table
    else:
        movie_vectors = tf.gather(movie_table, rows) + tf.matmul(tf.gather(g, rows), genre_weights)
    return movie_vectors
def get_score_matrix(user_rows, movie_rows, user_table, movie_table, genre_weights, g):
    user_vectors = tf.gather(user_table,  user_rows) 
    movie_vectors = get_movie_vectors(movie_table, g, genre_weights, movie_rows)
    score_mat = tf.matmul(user_vectors, movie_vectors, transpose_b = True)
    return score_mat
def test_score_mat():
    ratings, movies, movie_ids, user_ids = load_data()
    ut, mt, gw = get_variables(user_ids, movie_ids, 20, 8, 0)
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    zero = tf.zeros_like(gw)
    user_rows = tf.constant([0, 5, 609])
    movie_rows = tf.constant([3, 9000, 9741, 12])
    old = np.array(old_get_score_matrix(user_rows, movie_rows, ut, mt))
    new_zero = np.array(get_score_matrix(user_rows, movie_rows, ut, mt, zero, g))
    new_real = np.array(get_score_matrix(user_rows, movie_rows, ut, mt, gw, g))
    assert new_zero.shape == (3, 4)
    assert np.allclose(new_zero, old)          # zero weights -> identical to ID-only
    assert not np.allclose(new_real, old)      # real weights -> genre part matters
    genre_part = tf.linalg.matvec(gw, g[9000], transpose_a=True)   # (8,)
    by_hand = float(tf.reduce_sum(ut[5] * (mt[9000] + genre_part)))
    assert np.isclose(by_hand, new_real[1, 1])
    print("score matrix tests passed")
def get_full_score (user_table, movie_table, genre_weights, g, user_ids, movie_ids):
    movie_vectors = get_movie_vectors(movie_table, g, genre_weights, rows = None)
    full_score = tf.matmul(movie_vectors, user_table, transpose_b = True)
    assert full_score.shape == (len(movie_ids), len(user_ids))
    return full_score
def test_full_score_mat():
    ratings, movies, movie_ids, user_ids = load_data()
    ut, mt, gw = get_variables(user_ids, movie_ids, 20, 8, 0)
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    zero = tf.zeros_like(gw)
    old = np.array(old_get_full_score(ut, mt, user_ids, movie_ids))
    new = np.array(get_full_score (ut, mt, zero, g, user_ids, movie_ids))
    new_model = np.array(get_full_score (ut, mt, gw, g, user_ids, movie_ids))
    assert np.allclose(new, old)                    # zero weights -> same as ID-only
    assert new_model.shape == (len(movie_ids), len(user_ids))
    assert not np.allclose(old, new_model)             # real weights -> genre term is live
    # cross-path: batch (users x movies) vs full (movies x users)
    batch = get_score_matrix(tf.constant([5]), tf.constant([9000]), ut, mt, gw, g)
    assert batch.shape == (1, 1)
    assert np.isclose(float(batch[0, 0]), new_model[9000, 5])
    print("full score tests passed")
    return
def train_step(user_rows, movie_rows, user_table, movie_table, genre_weights, g, optimizer, log_q):
    with tf.GradientTape() as tape:
        score_mat = get_score_matrix(user_rows, movie_rows, user_table, movie_table, genre_weights, g)
        log_q_batch = tf.gather(log_q, movie_rows)
        new_score = score_mat - log_q_batch
        loss = calculate_loss(new_score)
    gradients = tape.gradient(loss,[user_table, movie_table, genre_weights])
    for gr in gradients: 
        assert gr is not None
    optimizer.apply_gradients(zip(gradients, [user_table, movie_table, genre_weights]))
    return loss
def test_train_step():
    ratings, movies, movie_ids, user_ids = load_data()
    ut, mt, gw = get_variables(user_ids, movie_ids, 20, 8, 0)
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    train, cv, test = make_split(ratings)
    train['user_row'] = user_ids.get_indexer(train['userId'])
    train['movie_row'] = movie_ids.get_indexer(train['movieId'])
    cv['user_row'] = user_ids.get_indexer(cv['userId'])
    cv['movie_row'] = movie_ids.get_indexer(cv['movieId'])
    columns = ['user_row', 'movie_row']
    for column in columns:
        assert not (train[column] == -1).any() ,(f"train ['{column}'] has a missing ID")
        assert not (cv[column] == -1).any()  ,(f"cv ['{column}'] has a missing ID")
    counts = train['movie_row'].value_counts().reindex(range(len(movie_ids)), fill_value=0).to_numpy()
    log_q = get_log_q(counts, len(train))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    train_batch = train.iloc[0:256]
    user_rows = train_batch['user_row'].values
    movie_rows = train_batch['movie_row'].values
    print(f'Max user_row value = {np.max(user_rows)}')
    print(f'Max moive_row value = {np.max(movie_rows)}')
    ut_saved = np.array(ut)
    mt_saved = np.array(mt)
    gw_saved = np.array(gw)
    loss = train_step(user_rows, movie_rows, ut, mt, gw, g, optimizer, log_q)
    assert np.isfinite(float(loss))
    assert not np.allclose(np.array(gw), gw_saved)
    assert not np.allclose(np.array(mt), mt_saved)
    assert not np.allclose(np.array(ut), ut_saved)
    print(f'Max loss across 3 arrays:')
    print(np.max(np.abs(np.array(gw) - gw_saved)))
    print(np.max(np.abs(np.array(mt) - mt_saved)))
    print(np.max(np.abs(np.array(ut) - ut_saved)))
    tf.random.set_seed(0)
    flag = False
    try:
        dummy = tf.Variable(tf.random.normal((len(user_ids), 8), stddev=0.1))
        with tf.GradientTape() as tape:
            score_mat = get_score_matrix(user_rows, movie_rows, ut, mt, gw, g)
            log_q_batch = tf.gather(log_q, movie_rows)
            new_score = score_mat - log_q_batch
            loss = calculate_loss(new_score)
        gradients = tape.gradient(loss,[ut, mt, gw, dummy])
        for gr in gradients: 
            assert gr is not None
        optimizer.apply_gradients(zip(gradients, [ut, mt, gw]))
    except AssertionError:
        flag = True
    if flag :
        print("all train step tests have passed")
def training_loop(train, user_table, movie_table, genre_weights, g, optimizer, batch_size, epochs, seed, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits):
    eval_hist = []
    means = []
    for i in range(epochs):
        train_shuffled = train.sample(frac=1, random_state=seed + i).reset_index(drop=True)
        J_hist = []
        for start in range(0, len(train), batch_size):
            end = start + batch_size
            sliced_batch = train_shuffled.iloc[start:end]
            user_rows = sliced_batch['user_row'].values
            movie_rows = sliced_batch['movie_row'].values
            J_hist.append(train_step(user_rows, movie_rows, user_table, movie_table, genre_weights, g, optimizer, log_q))
        print(f'Epoch: {i + 1}')
        means.append(np.mean(J_hist))
        print(f"\tMean loss  = {means[-1]}")
        scores= get_full_score(user_table, movie_table, genre_weights, g, user_ids, movie_ids)
        hits, hits_sum = evaluate(scores, train, cv, cand, movie_ids, user_ids)
        diff = hits.astype(int) - pop_hits.astype(int)
        low, high = bootstrap_ci(diff, n_boot=1000, seed=0)
        print(f'\tExtra hits over popularity, 95% CI: {low * len(diff):.1f} to {high * len(diff):.1f}')
        eval_hist.append(hits_sum)
        mean_counts = average_top20_count(scores, train, cand, counts)
        movie_norm = np.mean(tf.norm(movie_table, axis = 1).numpy()[cand])
        user_norm = np.mean(tf.norm(user_table, axis = 1).numpy())
        genre_norm = np.mean(tf.norm(g @ genre_weights, axis = 1).numpy()[cand])
        vector_norm = np.mean(tf.norm(get_movie_vectors(movie_table, g, genre_weights, rows = None), axis = 1).numpy()[cand])
        genre_weights_norm = np.mean(tf.norm(genre_weights, axis = 1).numpy())
        print(f'\tMovie norm = {movie_norm}')
        print(f'\tUser norm = {user_norm}')
        print(f'\tGenre matrix norm = {genre_norm}')
        print(f'\tFull movie vector norm = {vector_norm}')
        print(f'\tGenre weights norm = {genre_weights_norm}')
        print(f"\tMean count of top 20 movies = {mean_counts}")
        print(f"\tTotal hits on cv : {eval_hist[-1]}")
        print(f'\tCorrelation value = {diagnostic(scores, counts, cand)}')
    return means, eval_hist, hits
def main():
    regression = False
    ratings, movies, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    train['user_row'] = user_ids.get_indexer(train['userId'])
    train['movie_row'] = movie_ids.get_indexer(train['movieId'])
    cv['user_row'] = user_ids.get_indexer(cv['userId'])
    cv['movie_row'] = movie_ids.get_indexer(cv['movieId'])
    for name, df in [('train', train), ('cv', cv)]:                    # do it for both dataframes
        assert np.array_equal(
            user_ids[df['user_row'].to_numpy()].to_numpy(),            # rows -> back to userIds via the anchor
            df['userId'].to_numpy()), f"{name} user round trip failed"                                                       
        assert np.array_equal(movie_ids[df['movie_row'].to_numpy()].to_numpy(),
                            df['movieId'].to_numpy()), f"{name} movie round trip failed"
        assert not (df['user_row'].values == -1).any()
        assert not (df['movie_row'].values == -1).any()
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    if regression:
        g = tf.zeros_like(g)
    assert g.shape == (9742,20)
    genre_hits = {}
    print(f'Regression = {regression}')
    for i in range(3):
        print(f'Seed = {i}')
        user_table, movie_table, genre_weights = get_variables(user_ids, movie_ids, g.shape[1], length = 8, seed = i)
        cand = candidate_mask(train, 1, movie_ids)
        counts = train['movie_row'].value_counts().reindex(range(len(movie_ids)), fill_value=0).to_numpy()
        assert len(counts) == len(movie_ids) and counts.sum() == len(train)
        pop_scores = np.tile(counts[:, None], (1, len(user_ids))).astype(float)
        pop_hits, pop_sum = evaluate(pop_scores, train, cv, cand, movie_ids, user_ids)
        assert pop_sum == 35
        overlap = train.merge(cv, on=['user_row', 'movie_row'])
        assert len(overlap) == 0, f"{len(overlap)} pairs appear in both train and cv"
        log_q = get_log_q(counts, len(train))
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        t0 = time.time()
        means, eval_hist, hits = training_loop(train, user_table, movie_table, genre_weights, g, optimizer, 256, 15, i * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        print(f'Training duration {time.time() - t0}')
        genre_hits[i] = hits.astype(int) 
    if not regression:
        np.savez('hits_genres.npz', **{f'seed{k}': v for k, v in genre_hits.items()})
    id_only = np.load('hits_id_only.npz')
    for i in range (3):
        id_only_hits = id_only[f'seed{i}']
        print(f'Comparison on seed {i}')
        print(f'Id only total = {np.sum(id_only_hits)}')
        print(f'Genre enhanced total = {np.sum(genre_hits [i])}')
        id_only_total= np.sum(id_only_hits)
        genre_total = np.sum(genre_hits [i])
        diff_array = genre_hits[i] - id_only_hits 
        gained_users = (diff_array == 1).sum() 
        lost_users = (diff_array == -1).sum() 
        diff_users = gained_users - lost_users
        if id_only_total == genre_total:
            if np.array_equal(id_only_hits, genre_hits[i]):
                print('The two arrays are identical')
            else:
                print(f'Gained users = {gained_users}')
                print(f'Lost users = {lost_users}')
                print(f'Net = {diff_users}') 
        else:
            print(f'Difference between 2 models (genre - baseline) = {genre_total - id_only_total}')
            print(f'Gained users = {gained_users}')
            print(f'Lost users = {lost_users}') 
            print(f'Net = {diff_users}')


        
    
            
        
if __name__ == "__main__":
    main()