import numpy as np
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask, hit_rate_at_k, predict_pairs
import tensorflow as tf
import pandas as pd
# Rmse about 0.90-0.95
# hit@20 about 0.04
# Falsifier: hit@20 < 4%, Rmse > 0.98-0.99
def get_variables(user_ids, movie_ids, k):
    tf.random.set_seed(42)
    w = tf.Variable(tf.random.normal((len(user_ids), k), stddev=0.01))
    x = tf.Variable(tf.random.normal((len(movie_ids), k), stddev=0.01))
    b = tf.Variable(tf.zeros((1, len(user_ids))))
    return x, w, b
def normalization(train, movie_ids, y, r):
    global_mean = train['rating'].mean()
    movie_mean = train.groupby('movieId')['rating'].mean().reindex(movie_ids).fillna(global_mean)
    assert movie_mean.shape == (len(movie_ids),)
    movie_mean = movie_mean.to_numpy(dtype = np.float32).reshape(-1, 1)
    y_norm = (y - movie_mean) * r
    assert y_norm.dtype == np.float32
    return y_norm , movie_mean, global_mean
def cost(x, w, b, y_norm, r, lambda_):
    pred = x @ tf.transpose(w) + b
    err = tf.where(r, (pred - y_norm)**2, 0.0)
    j = tf.reduce_sum(err)/(2*tf.reduce_sum(tf.cast(r, tf.float32)))
    train_rmse = tf.sqrt(j*2)
    j += lambda_ * tf.reduce_sum(w**2)
    j += lambda_ *tf.reduce_sum(x**2)
    return j, train_rmse
def train_model(x, w, b, y_norm, r, lambda_, lr, iterations):
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    rmse_hist = []
    j_hist = []
    for i in range(iterations):
        with tf.GradientTape() as tape:
            j, train_rmse = cost(x, w, b, y_norm, r, lambda_)
        grads = tape.gradient(j, [x, w, b])
        optimizer.apply_gradients(zip(grads, [x, w, b]))
        if (i+1) % 50 == 0 or i == 0:
            j_hist.append(float(j))
            rmse_hist.append(float(train_rmse))
            print(f'Iteration {i+1}:')
            print(f'\tCost = {j_hist[-1]}')
            print(f'\tRMSE = {rmse_hist[-1]}')
    return x, w, b, j_hist, rmse_hist
def diganostic_a(pred_matrix, r_arr, cand, movie_mean):
    count = r_arr.sum(axis=1)
    for i in range(3):
        scores = np.where(cand & ~r_arr[:, i], pred_matrix[:, i], -np.inf)
        idx = np.argsort(-scores)[:20]
        print(count[idx])
        print(movie_mean[idx].ravel())
    return
def get_shrunk_mean(global_mean, train, lambda_, movie_ids):
    stats= train.groupby('movieId')['rating'].agg(['sum','count']).reindex(movie_ids).fillna(0)
    shrunk_mean = ((stats['sum'].values + lambda_ * global_mean) / (stats['count'].values + lambda_)).reshape(-1,1).astype(np.float32)
    assert shrunk_mean.shape == (len(movie_ids), 1)
    assert not np.isnan(shrunk_mean).any()
    return shrunk_mean
def main():
    ratings, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    y_train = train.pivot_table(index ='movieId', columns = 'userId', values = 'rating').reindex(index = movie_ids, columns = user_ids)
    r_train = y_train.notnull().astype(int)
    y_train = y_train.fillna(0)
    assert y_train.shape == r_train.shape == (len(movie_ids), len(user_ids))
    assert not train.duplicated(['movieId','userId']).any()
    y_arr = y_train.to_numpy(dtype = np.float32)
    r_arr = r_train.to_numpy(dtype = bool)
    assert r_arr.sum() == len(train)
    y_norm, movie_mean, global_mean = normalization(train, movie_ids, y_arr, r_arr)
    # for lam in lambdas:
    #     print(f'Lambda = {lam}')
    
    x, w, b = get_variables(user_ids, movie_ids, k = 10)
    x, w, b, j_hist, rmse_hist = train_model(x, w, b, y_norm, r_arr, 1e-4, lr = 0.01, iterations = 200)
    pred_matrix_model = (x @ tf.transpose(w) + b).numpy() + movie_mean
    truth = cv['rating'].values
    preds_model = predict_pairs(pred_matrix_model, cv, movie_ids, user_ids)
    sq_err_model = (preds_model - truth) ** 2

    #        rmse_nb = rmse(preds, truth)
    #     print(f'x = {tf.reduce_max(tf.abs(x))}')
    #     print(f'w = {tf.reduce_max(tf.abs(w))}')
    #     print(f'Final train rmse :{rmse_hist[-1]}')
    #     print(f'cv RMSE :{rmse_nb}')
    #     print('--------------------------------------------------------------------')
    #sq_err = (preds - truth) ** 2
    #low, high = bootstrap_ci(sq_err, n_boot=1000, seed = 0)        
    #print(f'Low: {np.sqrt(low)}')
    #print(f'High: {np.sqrt(high)}')
    #lambdas = [5, 10, 20, 50]
    #for count in counts:
    #    print(f' Count = {count}')
    #    cand = candidate_mask(train, count, movie_ids)
    #    hits = hit_rate_at_k(pred_matrix, cv, train, cand, movie_ids, user_ids, k = 20)
    #   print(f'Percentage : {hits.mean()}, total number : {hits.sum()}')
    #    print('--------------------------------------------------------------------')
    #control_scores = np.repeat(shrunk_mean, len(user_ids), axis=1)
    #assert control_scores.shape == pred_matrix.shape
    #counts_ = [20, 50]
    #print('Control test')
    #for count in counts_:
    #    print(f' Count = {count}')
    #    cand = candidate_mask(train, count, movie_ids)
    #    hits_ctrl = hit_rate_at_k(control_scores, cv, train, cand, movie_ids, user_ids, k = 20)
    #    print(f'Percentage : {hits_ctrl.mean()}, total number : {hits_ctrl.sum()}')
    assert (r_arr.sum(axis=0) > 0).all()
    user_offset = y_norm.sum(axis=0) / r_arr.sum(axis=0).reshape(1,-1)
    pred_matrix = movie_mean + user_offset
    preds = predict_pairs(pred_matrix, cv, movie_ids, user_ids)
    sq_err_base = (preds - truth) ** 2
    sq_err_diff = sq_err_base - sq_err_model
    low, high = bootstrap_ci(sq_err_diff, n_boot=1000, seed = 0)      
    print(f'Low: {np.sqrt(low)}')
    print(f'High: {np.sqrt(high)}')
    





if __name__ == '__main__':
    main()
