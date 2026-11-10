from two_tower import calculate_loss, evaluate, diagnostic, average_top20_count, get_log_q, get_genres_matrix, get_variables
from two_tower import get_score_matrix as old_get_score_matrix
from two_tower import get_full_score as old_get_full_score
import numpy as np
from common import bootstrap_ci, load_data, make_split, candidate_mask
import tensorflow as tf
import time
def get_mlp_variables(n_genres, width, length, seed):
    tf.random.set_seed(seed)
    w1 = tf.Variable(tf.random.normal((n_genres, width), stddev=0.1))
    w2 = tf.Variable(tf.random.normal((width, length), stddev=0.1))
    return w1, w2
def test_mlp_variables():
    ratings, movies, movie_ids, user_ids = load_data()
    seed = 0
    ut, mt, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    w1, w2 = get_mlp_variables(20, 32, 8, seed + 1000)

    # 1. shapes
    assert w1.shape == (20, 32)
    assert w2.shape == (32, 8)

    # 2. user and movie tables are exactly what get_variables gives for this seed
    ut2, mt2, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    assert np.array_equal(ut.numpy(), ut2.numpy())
    assert np.array_equal(mt.numpy(), mt2.numpy())

    # 3. w1 is not a copy of the first 640 numbers of movie_table (20 * 32 = 640)
    movie_prefix = mt.numpy().ravel()[:640]
    assert not np.allclose(w1.numpy().ravel(), movie_prefix)

    # 4. same seed twice gives identical MLP weights (reproducibility)
    w1_again, w2_again = get_mlp_variables(20, 32, 8, seed + 1000)
    assert np.array_equal(w1.numpy(), w1_again.numpy())
    assert np.array_equal(w2.numpy(), w2_again.numpy())

    # 5. the test can fail: with the SAME seed, check 3 should be caught
    get_variables(user_ids, movie_ids, 20, 8, seed)           # re-seeds with 0
    w1_bad, _ = get_mlp_variables(20, 32, 8, seed)            # same seed on purpose
    assert np.allclose(w1_bad.numpy().ravel(), movie_prefix), "check 3 could not detect the shared-seed bug"

    print(f"shapes: w1 {w1.shape}, w2 {w2.shape}")
    print("mlp variable tests passed")
def get_movie_vectors(movie_table, g, mlp_vars: list, rows=None):
    if rows is None:
        movie_vectors = movie_table + tf.nn.relu(g @ mlp_vars[0]) @ mlp_vars[1]
    else:
        movie_vectors = tf.gather(movie_table, rows) + tf.nn.relu(tf.gather(g, rows) @ mlp_vars[0]) @ mlp_vars[1]
    return movie_vectors
def test_movie_vectors():
    ratings, movies, movie_ids, user_ids = load_data()
    seed = 0
    ut, mt, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    w1, w2 = get_mlp_variables(20, 32, 8, seed + 1000)
    mlp_vars = [w1, w2]
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))

    row_list = [3, 9000, 9741, 12]
    rows = tf.constant(row_list)
    mt_np, w1_np, w2_np, g_np = mt.numpy(), w1.numpy(), w2.numpy(), g.numpy()

    # 1. zero g reduces to ID-only, exactly
    g_zero = tf.zeros_like(g)
    out_zero_all = get_movie_vectors(mt, g_zero, mlp_vars).numpy()
    assert out_zero_all.shape == (9742, 8)
    assert np.array_equal(out_zero_all, mt_np)
    out_zero_rows = get_movie_vectors(mt, g_zero, mlp_vars, rows).numpy()
    assert np.array_equal(out_zero_rows, mt_np[row_list])

    # 2. real g changes the output
    out_all = get_movie_vectors(mt, g, mlp_vars).numpy()
    assert not np.allclose(out_all, mt_np)

    # 3. by hand for movie row 9000, from both paths
    hidden_by_hand = np.maximum(g_np[9000] @ w1_np, 0)               # (32,)
    expected = mt_np[9000] + hidden_by_hand @ w2_np                  # (8,)
    out_rows = get_movie_vectors(mt, g, mlp_vars, rows).numpy()
    assert np.allclose(out_rows[1], expected, atol=1e-6)             # rows path
    assert np.allclose(out_all[9000], expected, atol=1e-6)           # rows=None path

    # 4. the relu is live: some hidden values are clipped, some pass through
    hidden_pre = g_np @ w1_np                                        # (9742, 32), before relu
    hidden_post = np.maximum(hidden_pre, 0)
    assert (hidden_pre < 0).any() and (hidden_post > 0).any()
    print(f"share of hidden values clipped by relu: {(hidden_pre < 0).mean():.2f}")

    # 5. the test can fail: the linear version (no relu) must differ
    linear = mt_np + (g_np @ w1_np) @ w2_np
    assert not np.allclose(out_all, linear)

    print("movie vector tests passed")
def get_score_matrix(user_rows, movie_rows, user_table, movie_table, g, mlp_vars):
    #rows are users, columns are movies, shape (B, B)
    movie_vectors = get_movie_vectors(movie_table, g, mlp_vars, movie_rows)
    score_mat = tf.matmul(tf.gather(user_table, user_rows), movie_vectors, transpose_b = True)
    return score_mat
def get_full_score(user_table, movie_table, g, mlp_vars, user_ids, movie_ids):
    #rows are movies, columns are users, shape (9742, 610)
    movie_vectors = get_movie_vectors(movie_table, g, mlp_vars, rows=None)
    full_score = tf.matmul(movie_vectors, user_table, transpose_b = True)
    assert full_score.shape == (len(movie_ids), len(user_ids))
    return full_score
def test_score_functions():
    ratings, movies, movie_ids, user_ids = load_data()
    seed = 0
    ut, mt, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    w1, w2 = get_mlp_variables(20, 32, 8, seed + 1000)
    mlp_vars = [w1, w2]
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    g_zero = tf.zeros_like(g)

    user_rows = tf.constant([0, 5, 609])
    movie_rows = tf.constant([3, 9000, 9741, 12])

    old_batch = np.array(old_get_score_matrix(user_rows, movie_rows, ut, mt))
    old_full = np.array(old_get_full_score(ut, mt, user_ids, movie_ids))

    new_batch_zero = np.array(get_score_matrix(user_rows, movie_rows, ut, mt, g_zero, mlp_vars))
    new_full_zero = np.array(get_full_score(ut, mt, g_zero, mlp_vars, user_ids, movie_ids))
    new_batch = np.array(get_score_matrix(user_rows, movie_rows, ut, mt, g, mlp_vars))
    new_full = np.array(get_full_score(ut, mt, g, mlp_vars, user_ids, movie_ids))

    # shapes (layouts)
    assert new_batch.shape == (3, 4)                      # users x movies
    assert new_full.shape == (len(movie_ids), len(user_ids))   # movies x users

    # 1. zero g -> identical to ID-only
    assert np.allclose(new_batch_zero, old_batch)
    assert np.allclose(new_full_zero, old_full)

    # 2. real g -> the MLP part is live
    assert not np.allclose(new_batch, old_batch)
    assert not np.allclose(new_full, old_full)

    # 3. one cell by hand: user 5, movie 9000 is cell [1, 1] of the batch
    ut_np, mt_np, g_np = ut.numpy(), mt.numpy(), g.numpy()
    w1_np, w2_np = w1.numpy(), w2.numpy()
    hidden = np.maximum(g_np[9000] @ w1_np, 0)
    movie_vec = mt_np[9000] + hidden @ w2_np
    by_hand = float(ut_np[5] @ movie_vec)
    assert np.isclose(by_hand, new_batch[1, 1], atol=1e-6)

    # 4. both layouts give the same score for the same (user, movie)
    assert np.isclose(new_batch[1, 1], new_full[9000, 5], atol=1e-6)

    print(f"batch shape {new_batch.shape}, full shape {new_full.shape}")
    print("score function tests passed")
def train_step(user_rows, movie_rows, user_table, movie_table, g, mlp_vars, optimizer, log_q):
    with tf.GradientTape() as tape:
        score_mat = get_score_matrix(user_rows, movie_rows, user_table, movie_table, g, mlp_vars)
        log_q_batch = tf.gather(log_q, movie_rows)
        new_score = score_mat - log_q_batch
        loss = calculate_loss(new_score)
    variables = [user_table, movie_table] + mlp_vars
    gradients = tape.gradient(loss, variables)
    for gr in gradients: 
        assert gr is not None
    optimizer.apply_gradients(zip(gradients, variables))
    return loss
def test_train_step():
    ratings, movies, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    train['user_row'] = user_ids.get_indexer(train['userId'])
    train['movie_row'] = movie_ids.get_indexer(train['movieId'])
    assert not (train['user_row'] == -1).any()
    assert not (train['movie_row'] == -1).any()

    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    counts = train['movie_row'].value_counts().reindex(range(len(movie_ids)), fill_value=0).to_numpy()
    log_q = get_log_q(counts, len(train))

    batch = train.iloc[0:256]
    user_rows = batch['user_row'].values
    movie_rows = batch['movie_row'].values

    # ---- real g: one step, all four variables must move ----
    seed = 0
    ut, mt, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    w1, w2 = get_mlp_variables(20, 32, 8, seed + 1000)
    mlp_vars = [w1, w2]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    saved = {name: np.array(v) for name, v in
             [('user_table', ut), ('movie_table', mt), ('w1', w1), ('w2', w2)]}
    loss = train_step(user_rows, movie_rows, ut, mt, g, mlp_vars, optimizer, log_q)
    assert np.isfinite(float(loss))
    print(f"loss after one step: {float(loss):.4f}")
    for name, v in [('user_table', ut), ('movie_table', mt), ('w1', w1), ('w2', w2)]:
        change = np.max(np.abs(np.array(v) - saved[name]))
        print(f"max change {name}: {change:.8f}")
        assert change > 0, f"{name} did not change"

    # ---- the guard can fail: a variable that is not in the loss gets a None gradient ----
    dummy = tf.Variable(tf.random.normal((len(user_ids), 8), stddev=0.1))
    with tf.GradientTape() as tape:
        score_mat = get_score_matrix(user_rows, movie_rows, ut, mt, g, mlp_vars)
        loss = calculate_loss(score_mat - tf.gather(log_q, movie_rows))
    grads = tape.gradient(loss, [ut, mt, w1, w2, dummy])
    assert all(gr is not None for gr in grads[:4])
    assert grads[4] is None
    print("guard check: dummy variable gets a None gradient, as it should")

    # ---- zero g: w1 and w2 must not move at all, the ID tables must ----
    ut0, mt0, _ = get_variables(user_ids, movie_ids, 20, 8, seed)
    w1_0, w2_0 = get_mlp_variables(20, 32, 8, seed + 1000)
    optimizer0 = tf.keras.optimizers.Adam(learning_rate=0.01)     # new optimizer for new variables
    saved0 = {name: np.array(v) for name, v in
              [('user_table', ut0), ('movie_table', mt0), ('w1', w1_0), ('w2', w2_0)]}
    train_step(user_rows, movie_rows, ut0, mt0, tf.zeros_like(g), [w1_0, w2_0], optimizer0, log_q)
    assert np.array_equal(np.array(w1_0), saved0['w1'])
    assert np.array_equal(np.array(w2_0), saved0['w2'])
    assert not np.array_equal(np.array(ut0), saved0['user_table'])
    assert not np.array_equal(np.array(mt0), saved0['movie_table'])
    print("zero g: w1 and w2 unchanged, user_table and movie_table changed")

    print("train step tests passed")
def dead_unit_share(g, mlp_vars, cand):
    hidden_layer = tf.nn.relu(g @ mlp_vars[0]).numpy()
    cand_layers = hidden_layer[cand]
    max_value = cand_layers.max(axis=0)
    count = np.sum(max_value == 0)
    count /= len(max_value)
    return count
def test_dead_unit_share():
    ratings, movies, movie_ids, user_ids = load_data()
    g = get_genres_matrix(movies, movie_ids)
    g = tf.constant(g.to_numpy(dtype=np.float32))
    cand = np.zeros(len(movie_ids), dtype=bool)
    cand[::2] = True                                   # every second movie, just for the test
    w2 = tf.constant(np.zeros((32, 8), dtype=np.float32))   # unused by the helper

    # all weights positive -> every unit alive for every movie
    w1_np = np.full((20, 32), 0.1, dtype=np.float32)
    assert dead_unit_share(g, [tf.constant(w1_np), w2], cand) == 0.0

    # one column all -1 -> that unit is 0 for every movie
    w1_np[:, 5] = -1.0
    share_one = dead_unit_share(g, [tf.constant(w1_np), w2], cand)
    assert np.isclose(share_one, 1 / 32)

    # two columns -> 2/32
    w1_np[:, 17] = -1.0
    share_two = dead_unit_share(g, [tf.constant(w1_np), w2], cand)
    assert np.isclose(share_two, 2 / 32)

    # the real random init, for information
    w1, w2_real = get_mlp_variables(20, 32, 8, 1000)
    print(f"dead share at random init: {dead_unit_share(g, [w1, w2_real], cand):.4f}")
    print(f"one dead column: {share_one:.5f}, two: {share_two:.5f}")
    print("dead unit tests passed")
def training_loop(train, user_table, movie_table, g, mlp_vars, optimizer, batch_size, epochs, seed, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits):
    means = []
    eval_hist = []
    for epoch in range(epochs):
        train_shuffled = train.sample(frac=1, random_state=seed + epoch).reset_index(drop=True)
        j_hist = []
        for start in range(0, len(train), batch_size):
            sliced_batch = train_shuffled.iloc[start:start + batch_size]
            user_rows = sliced_batch['user_row'].values
            movie_rows = sliced_batch['movie_row'].values
            j_hist.append(train_step(user_rows, movie_rows, user_table, movie_table, g, mlp_vars, optimizer, log_q))
        print(f'Epoch: {epoch + 1}')
        means.append(np.mean(j_hist))
        print(f'\tMean loss = {means[-1]}')
        full_scores = get_full_score(user_table, movie_table, g, mlp_vars, user_ids, movie_ids)
        hits, hits_sum = evaluate(full_scores, train, cv, cand, movie_ids, user_ids)
        eval_hist.append(hits_sum)
        diff = hits.astype(int) - pop_hits.astype(int)
        low, high = bootstrap_ci(diff, n_boot=1000, seed=0)
        print(f'\tExtra hits over popularity, 95% CI: {low * len(diff):.1f} to {high * len(diff):.1f}')
        print(f'\tAverage top 20 count = {average_top20_count(full_scores, train, cand, counts)}')
        movie_norm = np.mean(tf.norm(movie_table, axis = 1).numpy()[cand])
        user_norm = np.mean(tf.norm(user_table, axis = 1).numpy())
        genre_norm = np.mean(tf.norm(tf.nn.relu(g @ mlp_vars[0]) @ mlp_vars[1], axis = 1).numpy()[cand])
        vector_norm = np.mean(tf.norm(get_movie_vectors(movie_table, g, mlp_vars, rows = None), axis = 1).numpy()[cand])
        print(f'\tMovie norm = {movie_norm}')
        print(f'\tUser norm = {user_norm}')
        print(f'\tGenre norm = {genre_norm}')
        print(f'\tFull movie vector norm = {vector_norm}')
        print(f'\tDead unit share = {dead_unit_share(g, mlp_vars, cand)}')
        print(f'\tHits on cv = {hits_sum}')
        print(f'\tCorrelation value = {diagnostic(full_scores, counts, cand)}')
    return means, eval_hist, hits
def compare_hits(label_a, a, label_b, b):
    total_a = np.sum(a)
    total_b = np.sum(b)
    diff = a - b
    gained = np.sum(diff == 1)
    lost = np.sum(diff == -1)
    net = gained - lost
    print(f'{label_a} total = {total_a}')
    print(f'{label_b} total = {total_b}')
    if total_a == total_b:
        if np.array_equal(a, b):
            print('The two arrays are identical')
        else:
            print(f'Gained users = {gained}')
            print(f'Lost users = {lost}')
            print(f'Net = {net}') 
    else:
        print(f'Difference between 2 models ({label_a} - {label_b}) = {total_a - total_b}')
        print(f'Gained users = {gained}')
        print(f'Lost users = {lost}') 
        print(f'Net = {net}')
    return 
def test_compare_hits():
    import io
    from contextlib import redirect_stdout

    def run(a, b):
        buf = io.StringIO()
        with redirect_stdout(buf):
            compare_hits('A', np.array(a), 'B', np.array(b))
        return buf.getvalue()

    # asymmetric: gained 2, lost 0 (catches the old lost == diff 1 bug)
    out = run([1, 1, 1, 0], [1, 0, 0, 0])
    assert 'Gained users = 2' in out and 'Lost users = 0' in out and 'Net = 2' in out
    assert 'Difference between 2 models (A - B) = 2' in out

    # same total, different users: gained 1, lost 1, no difference line
    out = run([1, 1, 0, 0], [1, 0, 1, 0])
    assert 'Gained users = 1' in out and 'Lost users = 1' in out and 'Net = 0' in out
    assert 'Difference' not in out

    # opposite direction: gained 0, lost 2
    out = run([1, 0, 0], [1, 1, 1])
    assert 'Gained users = 0' in out and 'Lost users = 2' in out and 'Net = -2' in out

    # identical arrays
    out = run([1, 0, 1], [1, 0, 1])
    assert 'identical' in out and 'Gained' not in out

    print("compare_hits tests passed")
def main():
    test_mlp_variables()
    test_movie_vectors()
    test_score_functions()
    test_train_step()
    test_dead_unit_share()
    test_compare_hits()
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
    mlp_hits = {}
    print(f'Regression = {regression}')
    for i in range(3):
        print(f'Seed = {i}')
        user_table, movie_table, _ = get_variables(user_ids, movie_ids, g.shape[1], length=8, seed=i)
        w1, w2 = get_mlp_variables(g.shape[1], 32, 8, i + 1000)
        mlps = [w1,w2]
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
        means, eval_hist, hits = training_loop(train, user_table, movie_table, g, mlps, optimizer, 256, 15, i * 100, user_ids, movie_ids, cv, cand, counts, log_q, pop_hits)
        print(f'Training duration {time.time() - t0}')
        mlp_hits[i] = hits.astype(int) 
    if not regression:
        np.savez('hits_genres_mlp.npz', **{f'seed{k}': v for k, v in mlp_hits.items()})
    if regression:
        with np.load('hits_id_only.npz') as f:
            for i in range(3):
                print(f'Comparison on seed {i}')
                compare_hits('MLP (g=0)', mlp_hits[i], 'ID-only', f[f'seed{i}'])
    else:
        with np.load('hits_genres.npz') as f:
            for i in range(3):
                print(f'Comparison on seed {i}')
                compare_hits('MLP', mlp_hits[i], 'Genre-linear', f[f'seed{i}'])
        with np.load('hits_id_only.npz') as f:
            for i in range(3):
                print(f'Comparison on seed {i}')
                compare_hits('MLP', mlp_hits[i], 'ID-only', f[f'seed{i}'])
if __name__ == "__main__":
    main()

