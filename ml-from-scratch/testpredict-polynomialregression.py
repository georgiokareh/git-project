import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig
def poly(x):
    n = x.shape[0]
    p = np.zeros([n,5])
    p[:,0]= x[:,0]
    p[:,1]= x[:,1]
    p[:,2]= x[:,2]
    p[:,3]= x[:,0] ** 2
    p[:,4]= x[:,1] ** 2
    return p 

    
def cost(x, y, w, b,lamb):
    cost = 0
    m = x.shape[0]
    err = sigmoid(np.dot(x,w)+b)
    cost = -np.sum(y * np.log(err) + (1 - y) * np.log(1 - err)) / m 
    reg_cost = (lamb / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost
def grad (x, y, w, b, lamb):
    m = x.shape[0]
    err = sigmoid(np.dot(x,w)+b) - y
    dj_dw = (np.dot(x.T,err)) / m + (lamb/m) * w
    dj_db = np.sum(err)/m
    return dj_dw, dj_db
def grad_descent(x, y, w, b, alpha, nb_iter, lamb):
    J_history = []
    for i in range (nb_iter):
        dj_dw, dj_db = grad(x, y, w, b, lamb)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost(x, y, w, b, lamb))
        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}, cost {J_history[-1]:2f}")
    return w,b,J_history
def main():
    df = pd.read_csv("students.csv")
    x_raw = df[['hours_studied', 'sleep_hours', 'practice_tests']].values
    mu = x_raw.mean(axis=0)
    sigma = x_raw.std(axis=0)
    x = (x_raw - mu) / sigma
    p = poly(x)
    y = df['passed'].values
    alpha = 1e-2
    nb = 10000
    m = x.shape[1]
    w_in = np.zeros(5)
    b_in = 0
    lamb = 1

    w, b, J = grad_descent(p, y, w_in, b_in, alpha, nb, lamb)
    print(f"w = {w}, b = {b}")
    x_pred = np.zeros((1,m))
    x_pred[0,0] = float(input('How many hours studied? '))
    x_pred[0,1] = float(input('How many hours slept? '))
    x_pred[0,2] = float(input('How many practice tests? '))
    x_pred = (x_pred - mu) / sigma
    print(x_pred)
    i = input("hi")
    p_pred = poly(x_pred)
    y_pred = sigmoid(np.dot(p_pred, w) + b)
    y_pred = y_pred[0]
    pass_percent = y_pred * 100
    fail_percent = (1 - y_pred) * 100
    print(f"Chance to pass: {pass_percent:.2f}%, chance to fail: {fail_percent:.2f}%")
    h_vals = np.linspace(x[:,0].min() - 0.5, x[:,0].max() + 0.5, 100)
    s_vals = np.linspace(x[:,1].min() - 0.5, x[:,1].max() + 0.5, 100)
    H, S = np.meshgrid(h_vals, s_vals)
    grid_shape = H.shape
    H_flat= H.flatten()
    S_flat = S.flatten()
    n = H_flat.shape[0]
    T = np.zeros((n))
    a = np.column_stack([H_flat, S_flat, T])
    p_grid = poly(a)
    Z = sigmoid(np.dot(p_grid, w) + b)
    Z = Z.reshape(grid_shape)
    plt.contour(H, S, Z, levels=[0.5], colors='red')
    plt.scatter(x[y==1, 0], x[y==1, 1], color='blue', label='Pass')
    plt.scatter(x[y==0, 0], x[y==0, 1], color='orange', label='Fail')
    plt.ylim(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5)
    plt.xlabel('Hours studied (scaled)')
    plt.ylabel('Sleep hours (scaled)')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
