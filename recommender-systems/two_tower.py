import numpy as np
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask, hit_rate_at_k, predict_pairs
import tensorflow as tf
import pandas as pd
import time
# PREDICTION (ID-only two-tower, hit@20 on cv, cand min_count=1)
# Range: 40 to 50 hits (of 610)
# Reason: intuition, no mechanism
# Falsifier: if hits < 30 I suspect a bug; if hits > 65 I suspect leakage 
# The model would have outperformed my prediction if it hits between 50-65
def get_variables(user_ids, movie_ids, n_genres, length, seed):
    tf.random.set_seed(seed)
    movie_table = tf.Variable(tf.random.normal((len(movie_ids), length), stddev=0.1))
    user_table = tf.Variable(tf.random.normal((len(user_ids), length), stddev=0.1))
    genre_weights = tf.Variable(tf.random.normal((n_genres, length), stddev=0.1))
    return user_table, movie_table, genre_weights
def get_score_matrix(user_rows, movie_rows, user_table, movie_table):
    user_vectors  = tf.gather(user_table,  user_rows)    # shape (B, 8)
    movie_vectors = tf.gather(movie_table, movie_rows)   # shape (B, 8)
    score_matrix = tf.matmul(user_vectors, movie_vectors, transpose_b = True)
    return score_matrix
def calculate_loss(score_matrix):
    mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= tf.range(len(score_matrix)), logits= score_matrix))
    return mean_loss
def train_step(user_rows, movie_rows, user_table, movie_table, optimizer, log_q):
    with tf.GradientTape() as tape:
        score_matrix = get_score_matrix(user_rows, movie_rows, user_table, movie_table)
        log_q_batch = tf.gather(log_q, movie_rows)       # shape (B,)
        corrected = score_matrix - log_q_batch           # broadcasts along columns
        loss = calculate_loss(corrected)                  
    grads = tape.gradient(loss, [user_table, movie_table])
    optimizer.apply_gradients(zip(grads, [user_table, movie_table]))
    return loss
def training_loop(train, user_table, movie_table, optimizer, batch_size, epochs, seed, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits):
    means = []
    eval_hist = []
    for epoch in range(epochs):
        train_shuffled = train.sample(frac=1, random_state=seed + epoch).reset_index(drop=True)
        start = 0
        J_hist = []
        for start in range(0, len(train_shuffled), batch_size):
            sliced_batch = train_shuffled.iloc[start:start + batch_size]
            user_rows = sliced_batch['user_row'].values
            movie_rows = sliced_batch['movie_row'].values
            J_hist.append(train_step(user_rows, movie_rows, user_table, movie_table, optimizer, log_q).numpy())
        print(f"Epoch : {epoch + 1}")
        means.append(np.mean(J_hist))
        print(f"\tMean loss  = {means[-1]}")
        scores= get_full_score(user_table, movie_table, user_ids, movie_ids)
        hits, hits_sum = evaluate(scores, train, cv, cand, movie_ids, user_ids)
        diff = hits.astype(int) - pop_hits.astype(int)
        low, high = bootstrap_ci(diff, n_boot=1000, seed=0)
        print(f'\tExtra hits over popularity, 95% CI: {low * len(diff):.1f} to {high * len(diff):.1f}')
        eval_hist.append(hits_sum)
        mean_counts = average_top20_count(scores, train, cand, counts)
        movie_norm = np.mean(tf.norm(movie_table, axis = 1).numpy()[cand])
        user_norm = np.mean(tf.norm(user_table, axis = 1).numpy())
        print(f'\tMovie norm = {movie_norm}')
        print(f'\tUser norm = {user_norm}')
        print(f"\tMean count of top 20 movies = {mean_counts}")
        print(f"\tTotal hits on cv : {eval_hist[-1]}")
        print(f'\tCorrelation value = {diagnostic(scores, counts, cand)}')
    return means, eval_hist, hits
def get_full_score(user_table, movie_table, user_ids, movie_ids):
    full_score_matrix = tf.matmul(movie_table, user_table, transpose_b = True)
    assert full_score_matrix.shape == (len(movie_ids), len(user_ids))
    return full_score_matrix
def evaluate(scores, train, cv, cand, movie_ids, user_ids):
    scores = np.array(scores, dtype=float)
    hits = hit_rate_at_k(scores, cv, train, cand, movie_ids, user_ids, k = 20)
    return hits, hits.sum()
def diagnostic(scores, counts, cand):
    scores = np.asarray(scores, dtype=float)
    mean_scores = scores.mean(axis = 1)
    cand_mean_scores = mean_scores[cand]
    cand_counts = counts[cand]
    corr_matrix = np.corrcoef(cand_mean_scores, cand_counts)
    corr_value = corr_matrix[0, corr_matrix.shape[1] - 1]
    return corr_value
def average_top20_count(scores, seen, cand, counts):
    scores = np.array(scores, dtype=float)
    scores[seen['movie_row'], seen['user_row']] = -np.inf
    scores[~cand, :] = -np.inf
    idx = np.argpartition(scores, -20, axis = 0,)[-20:]
    return np.mean(counts[idx])
def get_log_q(counts, n_train):
    log_q = np.log(np.maximum(counts, 1) / n_train)
    return log_q.astype(np.float32)
def get_genres_matrix(movies, movie_ids):
    g = movies.set_index('movieId').loc[movie_ids, 'genres'].str.get_dummies(sep='|')
    assert g.index.equals(movie_ids)
    assert g.shape[0] == len(movie_ids)
    assert (g.sum(axis=1) >= 1).all(), "a movie has no genre column set"
    return g

def main():
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
    recovered_users = user_ids[train['user_row'].to_numpy()]
    assert np.array_equal(recovered_users.to_numpy(), train['userId'].to_numpy()), "train user round trip failed"
    g = get_genres_matrix(movies, movie_ids)
    print(g.shape, g.sum(axis=1).min())
    print(g.loc[1][g.loc[1] == 1].index.tolist()) 
    final_hits = {} 
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
        print(f'Mean of candidate counts = {counts[cand].mean()}')
        log_q = get_log_q(counts, len(train))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        t0 = time.time()
        means, eval_hist, hits = training_loop(train, user_table, movie_table, optimizer, 256, 15, i * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        print(f'Training duration {time.time() - t0}')
        final_hits[i] = hits.astype(int) 
    np.savez('hits_id_only.npz', **{f'seed{k}': v for k, v in final_hits.items()})



if __name__ == "__main__":
    main()