import two_tower as idm
import two_tower_part2 as genm
import two_tower_part3 as mlpm
from common import bootstrap_ci, load_data, make_split, candidate_mask
import pandas as pd
import numpy as np
import tensorflow as tf
from ch5_test_eval import prepare
#Pred : Average loss across all runs and models = 5-6
#Falsifier : Loss >= 8
#Reason : The popular movies have a high probability to appear in the test set, but leaving some room for error for people who rated niche movies
values = {}
expected_test = {'id': [61, 73, 60], 'gen': [69, 66, 63], 'mlp': [69, 65, 59]}
for name in ['id', 'gen', 'mlp']:
    with np.load(f'test_hits_{name}.npz') as f:
        values[name] = {}
        for i in range(3):
            values[name][i] = np.array(f[f'seed{i}'])
            assert len(values[name][i]) == 610
            assert values[name][i].sum() == expected_test[name][i]
print(values['id'][0].dtype)
train, cv, test, seen, g, cand, counts, log_q, movie_ids, user_ids = prepare()
pop_scores = np.tile(counts[:, None], (1, len(user_ids))).astype(float)
pop_hits, pop_sum = idm.evaluate(pop_scores, seen, test, cand, movie_ids, user_ids)
assert pop_sum == 43
assert len(pop_hits) == 610
pop_hits = pop_hits.astype(int)
lost_list = []
diff_avgs = {}
names = ['id', 'gen', 'mlp']
for name in names:
    diff = np.zeros((len(pop_hits), 3))
    for i in range(3):
        diff [:,i]= values[name][i] - pop_hits
    diff_average = np.mean(diff, axis = 1)
    diff_avgs[name] = diff_average
    assert diff_average.shape == (610,)
    low, high = bootstrap_ci(diff_average, n_boot=1000, seed=0)
    low *= len(diff_average)
    high *= len(diff_average)
    gained = np.sum(diff_average > 0)
    lost = np.sum(diff_average < 0)
    always_lost = np.sum(diff_average == -1)
    print(f'Model : {name}')
    print(f'\tMean net = {np.sum(diff_average):.1f}')
    print(f'\tInterval = [{low:.1f},{high:.1f}]')
    print(f'\tGained users = {gained}')
    print(f'\tLost users = {lost}')
    print(f'\tAlways lost users = {always_lost}')
pairs = {'id': 'gen', 'gen': 'mlp', 'mlp': 'id'}
for name in names:
    d = diff_avgs[name] - diff_avgs[pairs[name]]
    assert d.shape == (610,)
    low, high = bootstrap_ci(d, n_boot=1000, seed=0)
    low *= len(d)
    high *= len(d)
    print(f'Comparison between {name} & {pairs[name]}:')
    print(f'\tMean difference in hits = {d.sum():.1f}')
    print(f'\tInterval = [{low:.1f},{high:.1f}]')
    print(f'\tGained users by {name} model = {np.sum(d>0)}')
    print(f'\tGained users by {pairs[name]} model = {np.sum(d<0)}')

