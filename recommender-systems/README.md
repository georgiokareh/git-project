# Recommender systems on MovieLens: from popularity to two-tower models

## Project summary

This project was aimed to test different recommender systems algorithms on a movie dataset (MovieLens Latest Small) which is composed of 610 users and a catalogue of 9742 movies. What these algorithms did is train the model to be familiar with the movies each user has rated (the two-tower models treat every rated movie as a positive, whatever its rating) and at the end check whether the user's last movie was in the top 20 movies that user had not yet rated, based on the algorithm's prediction. The best algorithm reached about 65 hits (about 10.6%, between 59 and 73 across runs) compared to the worst one with 35, which were both compared to the baseline which was of 43 hits. The three two-tower variants could not be told apart.

## Project core split and logic

In this project, the data was split into three different sets: train, cv, test. The split was based on sorting each users rating per timestamp, from the oldest movie rated to the most recent one. The test set split consisted of cutting the most recent rating of each user, the cv set consisted of the second most recent movie rated while the train set consisted of the rest. The algorithm was trained based on the training set and tuned on the cv set, and at the end of each algorithm it was tested once on the test set using the hit@20 metric.

## Models

All models are scored on the same split and the same candidate movies. The first baseline, most-rated, ranks movies by how many train ratings they have and is the same for every user. The second, rating-based matrix factorization, learns a vector for each user and each movie from the rating values (with a movie average shrunk toward the global mean) and ranks by predicted rating. The main models are two-tower models trained on implicit feedback: every (user, movie) pair in the training data counts as a positive, whatever its rating. Each user and each movie gets an 8-dimensional vector, and the score is their dot product. Training uses a softmax over the other movies in the same batch of 256 pairs, with a correction for how often each movie appears (logQ), so the model isn't pushed to avoid popular movies. I compared three versions of the movie tower: ID-only (one learned vector per movie); genre-linear (the ID vector plus a learned linear map of the movie's 20 genre flags); and MLP (the ID vector plus a small neural network with one hidden layer of 32 ReLU units, applied to the same genre flags). All three train for 15 epochs with Adam and are run with three random seeds.

## Results (test set, hit@20, 610 users)

| Model | Seed 0 | Seed 1 | Seed 2 | Mean hits | Hit rate |
|---|---|---|---|---|---|
| Most-rated (popularity) | 43 | | | 43 | 7.05% |
| Rating-based matrix factorization | 35 | | | 35 | 5.74% |
| Two-tower, ID-only | 61 | 73 | 60 | 64.7 | 10.6% |
| Two-tower, ID + genres (linear) | 69 | 66 | 63 | 66.0 | 10.8% |
| Two-tower, ID + genres (MLP) | 69 | 65 | 59 | 64.3 | 10.5% |

### Gain over most-rated (average of 3 seeds, 95% bootstrap interval over users)

| Model | Mean gain (hits) | Interval |
|---|---|---|
| Two-tower, ID-only | +21.7 | 7.3 to 34.3 |
| Two-tower, ID + genres (linear) | +23.0 | 9.3 to 36.0 |
| Two-tower, ID + genres (MLP) | +21.3 | 7.3 to 34.7 |

The three two-tower variants differ by at most 1.7 hits, which is well inside the noise (differences below about 8 hits cannot be detected with 610 users), so they cannot be told apart.

## Personal statement

This project was aimed to understand the application of different recommender systems algorithms and seeing how each one works and understanding practically the way each one works, also this project taught me how to evaluate different models based on the results and a lot of different metrics.

## Limits

The evaluation has one held-out movie per user, so there are only 610 test observations, and the 95% interval on a hit count is about ±14 hits. The three two-tower variants differ by at most 1.7 hits, so differences below roughly 8 hits cannot be detected here and the models cannot be ranked. The three random seeds are trained on the same users, so the intervals show uncertainty from which users were sampled, not from seed-to-seed variation. Hits dropped by 5 to 7 between the validation split and the test split while the popularity baseline rose from 35 to 43, so part of the validation score was optimistic, because design choices such as the popularity correction and the number of epochs were made on validation. The candidate set excludes movies with no training ratings (the best possible score is 586 of 610 users), so this project cannot show whether genre features help on new movies. Only warm movies are tested, which is exactly where the ID embeddings already carry the information. Finally, "next movie" is a task that rewards popularity, so most-rated is a strong baseline here, and a different task (for example, recommending from a catalogue the user has not browsed) could change the picture.

## How to run

This repository does not include the MovieLens data. Download MovieLens Latest Small from GroupLens and put ratings.csv and movies.csv in this folder. Install numpy, pandas and tensorflow. The two-tower models are in two_tower.py (ID-only), two_tower_part2.py (genres) and two_tower_part3.py (MLP). ch5_test_eval.py trains all nine runs (3 models × 3 seeds) and evaluates them once on the test split; it takes about 10 minutes and writes the test_hits_*.npz files. ch5_test_analysis.py loads those files and compares each model with the most-rated baseline, and the models with each other. Results may differ slightly across machines and TensorFlow versions.

The scripts contentbased.py, collaborative_filtering.py, final_collabtest.py and test_common.py were written for an earlier version of load_data() that returned three values; they need a small update to run with the current common.py.

The hits_*.npz and test_hits_*.npz files hold the per-user hit/miss results (one 0/1 value per user for each seed), included so the analysis can run without retraining.
