# Notes: Ch. 6 (retrieval and ranking) and Ch. 8 (cold start and hybrids)

## Ch. 6: Large-scale systems: retrieval and ranking

**1. What does each stage optimize, and why do the goals differ?**
Retrieval is the stage that transforms a large movie catalogue into a few hundred candidates for each user, which lowers the cost of computing every user with every movie, like what we did with our algorithms. It is cheap because the item vectors are precomputed once in advance: at request time (when a user logs in) only that user's vector is loaded and the closest items are selected. Retrieval focuses on recall rather than precision. A movie that retrieval leaves out is lost for good, while a mediocre movie that gets in will be sorted out later by ranking, so a high percentage of the movies the user would have liked should be selected, and bad movies in the candidate list are tolerated. Example: the user would like 10 movies; retrieval returns 300 candidates containing 9 of them; recall is 90%, precision is 3%, and it is still a good retrieval stage. Ranking works only on those candidates and orders them from best to worst using precision, so now it cares about which movie is better than the other. It can recommend the top N for a user, but only from the ones retrieval selected: if retrieval skips a movie the user would have liked, ranking can never bring it back.

**2. Why can't a user-item cross-feature be used at retrieval?**
A user-item cross-feature depends on both the user and the specific item, so it is computed for every user with every item. That is the opposite of what retrieval does: using it at retrieval would mean going over all the movies for each request, which is exactly the cost retrieval avoids. Item vectors do not depend on the user, so they can be computed once and indexed; a cross-feature depends on the pair, so it cannot be precomputed. At the ranking stage it is fine, because only a few hundred candidates are left.

**3. A founder with 500 items wants a full two-stage pipeline before launch (Section 6.3).**
It is a bad decision. For a few hundred movies it is cheap to score the whole catalogue directly (500 vectors can be computed in milliseconds); a retrieval stage is only worth it for a huge catalogue, so at this size it would be a waste of work. The alternative is to skip the two-stage pipeline and score all items with one simple model, and switch when the catalogue (hundreds of thousands to millions of items) or the latency budget makes direct scoring too slow. At launch there is also very little interaction data, so even a neural two-tower model would be premature; a simple popularity or content-based ranking is the better start (see Ch. 8).

**4. Where my project sits.**
For my recommender project (about 10k movies per user), exhaustive scoring is still cheap: my evaluation scored all 9742 movies for all 610 users in one matrix. My two-tower model is a retrieval-style model (a dot product of user and movie vectors that could be indexed), used here to score everything directly: no approximate search and no ranking stage were needed. Retrieval and ranking are two stages of one pipeline, not two options to choose between.

**Link to Phase 4:** RAG has the same shape: documents are embedded once (like item vectors), the question is embedded at request time, the nearest chunks are fetched (retrieval; vector databases are approximate nearest-neighbour indexes), and the LLM then reads those few chunks and writes the answer (the expensive, careful stage).

## Ch. 8: Cold start and hybrids

**1. The three cold-start scenarios and one mitigation for each.**
The new user scenario is mitigated by asking the user at signup for extra preferences (explicit onboarding), which helps recommend the right items to him. The new item scenario is mitigated by giving the new item a little more exposure (deliberate exploration) so we can see how users react to it and collect data on it; content-based scoring is the other option. The new platform scenario is mitigated with non-personalized popularity or editorial curation first, and content-based filtering as the main engine, since it works on the non-interaction data of the movies (one-hot genre vectors, etc.), until enough data builds up.

**2. Weighted vs switching hybrid.**
A weighted hybrid combines the scores of both models (content-based and collaborative) with fixed or learned weights, while a switching hybrid uses only one of them depending on how much data a user or item has: content-based when the history is sparse, collaborative once a threshold is reached.

**3. Cascade hybrid vs retrieval-then-ranking.**
They are structurally similar because both start with a high-recall stage that produces a smaller but effective candidate set, and the second stage ranks that candidate set, which emphasizes high precision. The difference is how the stages are chosen: in a cascade the first model is usually content-based or popularity and the second is a different algorithm family, so it combines two families to handle sparse data; in retrieval-then-ranking the two stages are split by cost and scale, a cheap broad stage over the whole catalogue and then an expensive precise stage over a few hundred candidates.

**4. The founder embarrassed by "just popularity ranking" (Section 8.4).**
It is a bad idea because it takes the founder extra time to have the product ready, for no important improvement; he could have started with basic popularity ranking and moved to a more sophisticated model when the data increases. Building a two-tower model before launch is not beneficial for the work put in, and sometimes it is even worse, since there is no data on the users and their preferences. Popularity at launch is not a failure to personalize: no algorithm can personalize from zero data, and the simple stage is the correct choice, not a compromise. My own project shows how strong popularity is: most-rated got 43 hits on test, against 35 for my rating-based matrix factorization.

**5. A staged cold-start plan for the marketplace.**
- **Day one (pre-launch to the first few hundred users):** non-personalized popularity (or editorial picks), plus explicit onboarding preferences at signup, because it is the most efficient choice for a platform with no data.
- **Early growth (some usage, noisy data):** a switching hybrid: content-based by default, using item features and the preferences from onboarding, and collaborative filtering for a user or an item once it passes a minimum number of interactions. Item threshold (starting guess, to tune on real data): about 20 to 50 interactions per item, because in my Ch. 4 sweep the hits of an average-based ranking rose from 0 to 31 as the minimum rating count went from 1 to 50 (1, 5, 10, 20, 50 gave 0, 2, 6, 16, 31 hits). User threshold: tune on real data.
- **Mature (millions of items, a lot of users and interactions):** a two-tower neural model served through retrieval and ranking.
- **Two different triggers:** the move to collaborative filtering is triggered by data volume per user or item; the move to retrieval and ranking is triggered by catalogue size and latency (direct scoring becomes too slow, hundreds of thousands to millions of items), and the volume of data also justifies the neural two-tower model.
