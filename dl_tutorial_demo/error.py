import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == '__main__':
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1, 0.05, 0.6,0.0, 0.05,0.1, 0.0,0.1, 0.0, 0.0]
    ans = mean_squared_error(np.array(y), np.array(t))
    ans = cross_entropy_error(np.array(y), np.array(t))
    print(ans)
