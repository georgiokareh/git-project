import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def compute_cost(w, x, b, y, lmb, R):
    m = np.sum(R)
    cost = np.dot(x, w.T) + b
    error = (cost - y)**2
    w_sum = np.sum(w**2)
    filtered_mat = R * error
    sum = np.sum(filtered_mat)/(2*m)
    reg = (lmb * w_sum)/(2*m)
    final_cost = sum + reg
    return final_cost
def compute_gradients(w, x, b, y, lmb, R):
    cost = np.dot(x, w.T) + b
    err = (cost - y) * R
    dW = np.dot(err.T, x) + lmb * w
    db = np.sum(err, axis = 0)
    return dW , db
def grad_loop(w, x, b, y, alpha, nb_iter, R, lmb):
    m = np.sum(R)
    J_hist = []
    
    for i in range(nb_iter):
        if i == 0:
            
            J_hist.append(compute_cost(w, x, b, y, lmb, R))
        
        dw, db = compute_gradients(w, x, b, y, lmb, R)
        dw /= m
        db /= m
        w -= alpha * dw
        b -= alpha * db.reshape(1,-1)
        if(i+1) % 250 == 0:
            J_hist.append(compute_cost(w, x, b, y, lmb, R))
            
        if (i+1)%1000 == 0:
            print(f'Iteration {i+1}, Cost : {J_hist[-1]}')
    return w, b , J_hist      
def compute_rmse(w,b,test,Y_aligned,genres):
    err = np.zeros (len(test))
    for i, row in enumerate(test.itertuples()):
        user_id = row.userId
        movie_id = row.movieId
        true_rating = row.rating
        user_pos = list(Y_aligned.columns).index(user_id)
        pred = genres.loc[movie_id].values @ w[user_pos] + b[0, user_pos]
        err[i] = (pred - true_rating) ** 2
    rmse = np.sqrt(np.sum(err)/len(err))
    return rmse
def main():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    exploded = movies['genres'].str.split('|').explode()
    unique_genres = exploded.unique()
    for genre in unique_genres:
        movies[f'{genre}'] = (movies['genres'].str.contains(f'{genre}',regex = False)).astype(int)
    categories = ['title','genres']
    genres = movies.drop(categories, axis=1)
    genres = genres.set_index('movieId')
    ratings = ratings.sort_values('timestamp')
    test = ratings.groupby('userId').tail(1) #leave one out of each group
    train = ratings.drop(test.index)  
    user_means = train.groupby('userId')['rating'].agg(['mean'])

    Y = train.pivot_table(index='movieId', columns='userId', values='rating') #create a table that will consist of rows and columns 

    R = Y.notnull().astype(int)
    shared_ids = genres.index.intersection(Y.index) # find the index values that are present in both dfs
    X = genres.loc[shared_ids] # keep the values with indexes present in the intersection index values
    Y_aligned = Y.loc[shared_ids]
    #print(user_means.index == Y_aligned.columns)
    #print(X.shape[0] == Y_aligned.shape[0]) #to check if they have the same row dimension
    #print((X.index == Y_aligned.index).all())  #to check if the indexes are aligned
    n_users = Y_aligned.shape[1]
    n_features = X.shape[1]
    rng = np.random.default_rng(42)
    W = rng.normal(size=(n_users, n_features)) * 0.01
    b = user_means['mean'].values.reshape(1,-1).copy()
    X_arr = X.values          # or X.to_numpy()
    Y_arr = Y_aligned.fillna(0).values
    R_arr = R.values
    w, b, J_hist, rmse_hist =grad_loop(W, X_arr, b, Y_arr, 0.1, 2000, R_arr, 1, genres, Y_aligned, test)
    print("Final test RMSE:", rmse_hist[-1])
    genres_arr = genres.values
    scores = genres_arr @ w.T + b
    movie_pos = genres.index.get_indexer(train['movieId'])
    user_pos = Y_aligned.columns.get_indexer(train['userId'])
    scores[movie_pos, user_pos] = -np.inf
    scores_ranked = np.argsort(-scores,axis = 0)
    top_k = scores_ranked[:50]
    test_movie_pos = genres.index.get_indexer(test['movieId'])
    test_user_pos = Y_aligned.columns.get_indexer(test['userId'])
    hits = []
    for n in range(len(test)):
        hit = test_movie_pos[n] in top_k[:, test_user_pos[n]]
        hits.append(hit)
    print("Hit rate@10:", np.mean(hits), "| users with a hit:", sum(hits), "of", len(hits))
    #scores_df = pd.DataFrame(scores, index=genres.index, columns=Y_aligned.columns)
    #top10 = scores_df[5].sort_values(ascending=False).head(10)
    #top10 = top10.to_frame(name='predicted')
    #result = top10.join(movies.set_index('movieId')[['title', 'genres']])
    #top20 = Y_aligned[5].sort_values(ascending=False).head(20)
    #top20 = top20.to_frame(name="rating")
    #real_top20= top20.join(movies.set_index('movieId')[['title', 'genres']])
    #print(result)
    #print(real_top20)
    #print(np.sum(scores == -np.inf) == len(train))
    print((movie_pos == -1).sum() == 0)
    print((user_pos == -1).sum() == 0)
    x = np.linspace(0,2000,len(J_hist))
    x_rmse = np.linspace(0,2000,len(rmse_hist))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, J_hist)
    ax1.set_title("Cost function variation")
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylabel("Cost function")

    ax2.plot(x_rmse, rmse_hist)
    ax2.set_title("RMSE variation")
    ax2.set_xlabel("Number of iterations")
    ax2.set_ylabel("RMSE")

    plt.show()
    

    
    
if __name__ == "__main__":
    main()
    