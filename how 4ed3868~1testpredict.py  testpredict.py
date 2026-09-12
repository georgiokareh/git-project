[33mcommit 4ed38681761c858fa71b3dfd1c87645cacc053d6[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmaster[m[33m)[m
Author: Georgio Kareh <jorgekh38@gmail.com>
Date:   Tue Jul 28 00:49:42 2026 +0300

    First time with poly features, got a grasp of how to pick the right features

[1mdiff --git a/testpredict.py b/testpredict.py[m
[1mindex a38b9f0..50d221c 100644[m
[1m--- a/testpredict.py[m
[1m+++ b/testpredict.py[m
[36m@@ -1,28 +1,41 @@[m
 import pandas as pd[m
 import numpy as np[m
[32m+[m[32mimport matplotlib.pyplot as plt[m
 [m
 def sigmoid(z):[m
     sig = 1/(1 + np.exp(-z))[m
     return sig[m
[31m-def cost(x, y, w, b):[m
[32m+[m[32mdef poly(x):[m
[32m+[m[32m    n = x.shape[0][m
[32m+[m[32m    p = np.zeros([n,5])[m
[32m+[m[32m    p[:,0]= x[:,0][m
[32m+[m[32m    p[:,1]= x[:,1][m
[32m+[m[32m    p[:,2]= x[:,2][m
[32m+[m[32m    p[:,3]= x[:,0] ** 2[m
[32m+[m[32m    p[:,4]= x[:,1] ** 2[m
[32m+[m[32m    return p[m[41m [m
[32m+[m
[32m+[m[41m    [m
[32m+[m[32mdef cost(x, y, w, b,lamb):[m
     cost = 0[m
     m = x.shape[0][m
     err = sigmoid(np.dot(x,w)+b)[m
[31m-    cost = -np.sum(y * np.log(err) + (1 - y) * np.log(1 - err)) / m[m
[31m-    return cost[m
[31m-def grad (x, y, w, b):[m
[32m+[m[32m    cost = -np.sum(y * np.log(err) + (1 - y) * np.log(1 - err)) / m[m[41m [m
[32m+[m[32m    reg_cost = (lamb / (2 * m)) * np.sum(w ** 2)[m
[32m+[m[32m    return cost + reg_cost[m
[32m+[m[32mdef grad (x, y, w, b, lamb):[m
     m = x.shape[0][m
     err = sigmoid(np.dot(x,w)+b) - y[m
[31m-    dj_dw = (np.dot(x.T,err)) / m[m
[32m+[m[32m    dj_dw = (np.dot(x.T,err)) / m + (lamb/m) * w[m
     dj_db = np.sum(err)/m[m
     return dj_dw, dj_db[m
[31m-def grad_descent(x, y, w, b, alpha, nb_iter):[m
[32m+[m[32mdef grad_descent(x, y, w, b, alpha, nb_iter, lamb):[m
     J_history = [][m
     for i in range (nb_iter):[m
[31m-        dj_dw, dj_db = grad(x, y, w, b)[m
[32m+[m[32m        dj_dw, dj_db = grad(x, y, w, b, lamb)[m
         w -= alpha * dj_dw[m
         b -= alpha * dj_db[m
[31m-        J_history.append(cost(x, y, w, b))[m
[32m+[m[32m        J_history.append(cost(x, y, w, b, lamb))[m
         if (i+1) % 1000 == 0:[m
             print(f"Iteration {i+1}, cost {J_history[-1]:2f}")[m
     return w,b,J_history[m
[36m@@ -32,23 +45,51 @@[m [mdef main():[m
     mu = x_raw.mean(axis=0)[m
     sigma = x_raw.std(axis=0)[m
     x = (x_raw - mu) / sigma[m
[32m+[m[32m    p = poly(x)[m
     y = df['passed'].values[m
     alpha = 1e-2[m
     nb = 10000[m
     m = x.shape[1][m
[31m-    w_in = np.zeros(m)[m
[32m+[m[32m    w_in = np.zeros(5)[m
     b_in = 0[m
[31m-    w, b, J = grad_descent(x, y, w_in, b_in, alpha, nb)[m
[32m+[m[32m    lamb = 1[m
[32m+[m
[32m+[m[32m    w, b, J = grad_descent(p, y, w_in, b_in, alpha, nb, lamb)[m
     print(f"w = {w}, b = {b}")[m
[31m-    x_pred = np.zeros(m)[m
[31m-    x_pred[0] = float(input('How many hours studied? '))[m
[31m-    x_pred[1] = float(input('How many hours slept? '))[m
[31m-    x_pred[2] = float(input('How many practice tests? '))[m
[32m+[m[32m    x_pred = np.zeros((1,m))[m
[32m+[m[32m    x_pred[0,0] = float(input('How many hours studied? '))[m
[32m+[m[32m    x_pred[0,1] = float(input('How many hours slept? '))[m
[32m+[m[32m    x_pred[0,2] = float(input('How many practice tests? '))[m
     x_pred = (x_pred - mu) / sigma[m
[31m-    y_pred = sigmoid(np.dot(x_pred, w) + b)[m
[32m+[m[32m    print(x_pred)[m
[32m+[m[32m    i = input("hi")[m
[32m+[m[32m    p_pred = poly(x_pred)[m
[32m+[m[32m    y_pred = sigmoid(np.dot(p_pred, w) + b)[m
[32m+[m[32m    y_pred = y_pred[0][m
     pass_percent = y_pred * 100[m
     fail_percent = (1 - y_pred) * 100[m
     print(f"Chance to pass: {pass_percent:.2f}%, chance to fail: {fail_percent:.2f}%")[m
[32m+[m[32m    h_vals = np.linspace(x[:,0].min() - 0.5, x[:,0].max() + 0.5, 100)[m
[32m+[m[32m    s_vals = np.linspace(x[:,1].min() - 0.5, x[:,1].max() + 0.5, 100)[m
[32m+[m[32m    H, S = np.meshgrid(h_vals, s_vals)[m
[32m+[m[32m    grid_shape = H.shape[m
[32m+[m[32m    H_flat= H.flatten()[m
[32m+[m[32m    S_flat = S.flatten()[m
[32m+[m[32m    n = H_flat.shape[0][m
[32m+[m[32m    T = np.zeros((n))[m
[32m+[m[32m    a = np.column_stack([H_flat, S_flat, T])[m
[32m+[m[32m    p_grid = poly(a)[m
[32m+[m[32m    Z = sigmoid(np.dot(p_grid, w) + b)[m
[32m+[m[32m    Z = Z.reshape(grid_shape)[m
[32m+[m[32m    plt.contour(H, S, Z, levels=[0.5], colors='red')[m
[32m+[m[32m    plt.scatter(x[y==1, 0], x[y==1, 1], color='blue', label='Pass')[m
[32m+[m[32m    plt.scatter(x[y==0, 0], x[y==0, 1], color='orange', label='Fail')[m
[32m+[m[32m    plt.ylim(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5)[m
[32m+[m[32m    plt.xlabel('Hours studied (scaled)')[m
[32m+[m[32m    plt.ylabel('Sleep hours (scaled)')[m
[32m+[m[32m    plt.legend()[m
[32m+[m[32m    plt.show()[m
[32m+[m
 [m
 [m
 if __name__ == "__main__":[m
