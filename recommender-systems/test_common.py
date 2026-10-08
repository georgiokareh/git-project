import numpy as np
from common import rmse, bootstrap_ci, load_data, make_split, candidate_mask

def test_rmse():
    truth = np.array([5.0, 4.0])
    assert rmse(truth, truth) == 0.0
    assert abs(rmse(np.array([3.0, 4.0]), truth) - np.sqrt(2)) < 1e-9

def test_bootstrap():
    assert bootstrap_ci(np.ones(100)) == (1.0, 1.0)
    assert bootstrap_ci(np.zeros(100)) == (0.0, 0.0)

def test_split():
    ratings, movie_ids, user_ids = load_data()
    train, cv, test = make_split(ratings)
    assert len(cv) == 610 and len(test) == 610
    assert len(train) + len(cv) + len(test) == len(ratings)

if __name__ == "__main__":
    test_rmse()
    test_bootstrap()
    test_split()
    print("all tests passed")