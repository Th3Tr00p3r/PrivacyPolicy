# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## TODO: see into turning this into a regression problem, i.e. giving PPs a score 0-1?
# * Idea - decide on scores for the ToS;DR labels, i.e. A=1.0, B=0.75, C=0.50, D=0.25, E=0.0 and when calculating the score for an inferred policy do it in the following way:
#     1) Average all labeled training policies of each label.
#     2) Calculate its similarity score to each of the mean label vectors, rescaled to \[0, 1\].
#     3) Normalize the similarities to one, and use a threshold to ignore (rescaled, normalized) similarities below, e.g. 0.5
#     4) For the surviving similarity scores, multiply each by the label score (A/B/C/D/E, as above) and sum the products - that is the policy score!
# ## TODO: Try [UMAP](https://github.com/lmcinnes/umap) visualization, for speed if anything else
# ## TODO: Topic Modeling - Consider using topic modeling techniques (e.g., Latent Dirichlet Allocation - LDA) to identify underlying topics within the documents. Visualize the topics and their prevalence in the dataset.

# %%
# import project
import sys

sys.path.append("D:/MEGA/Programming/ML/PPA/")

from IPython.display import display  # type: ignore
from winsound import Beep
from ppa.utils import config_logging
import logging
from ppa.display import Plotter
import numpy as np
from collections import Counter
import psutil
import pandas as pd

# Configure logging
config_logging()

# %% [markdown]
# # The General Idea
#
# Being, let's say, a privacy-oriented person, I have always been quite frustrated with the following:
# You want to start using a product/service. You want to know if you should be worried about consenting to the privacy policy (PP), but that means you'll have to read the whole thing! Is there any way to easily gauge the privacy policy laid before you?
#
# In order to understand if that should be possible (supposing it is, of course) and possibly come to a simple solution, I will attempt to build a (hoprfully) small project, which will inevitably involve NLP as privacy policies are textual.

# %% [markdown]
# # 1 The Data
# Obviously, I would need as many legitimate PPs as I can get my hands on. From these I would need to extract the text, clean it, tokenize it, and create vector embeddings. Currently I will not attempt to identify values in the text itself (e.g personal data retention period) as I'm not an expert on PPs (or any legal matter) and I'm unsure if PPs are standardized in any way.
#
# ## 1.1 ETL
# We begin with a simple ETL process, appropriate for most downstream NLP tasks.
#
# ### 1.1.1 Getting Raw Data (Extraction)
# Initially I decided to try and gather some privacy policies myself. I designed a spider using Scrapy to use several search-engines and look for "privacy policy". The spider scours each search page for links matching some simple pattern for what an actual privacy policy URL should look like, gets the text parts from the relevant URLs using BeautifulSoup, and keeps the URL-text pairs in a dictionary, which is saved and automatically updated with new links for subsequent scrapes.
#
# However, I only managed to get about 200 policies after a while.
#
# I then tried to look for publicly available databases of privacy policies, and surprisingly found a great one immediately, collected by the authors of this [paper](https://arxiv.org/pdf/2008.09159.pdf). Not only does it contain about a hundred-thousand policies, it also containg archived older versions of each! (which I may not need, but are nice to have). The data is available from [GitHub](https://github.com/citp/privacy-policy-historical/tree/master):

# %%
RAW_DATA_REPO_URL = "https://github.com/citp/privacy-policy-historical.git"

# %% [markdown]
# where all policies are available either in Markdown or HTML, and are alphabtically ordered in folders.
# The database is also said to be available by SQLite access, which must be requested by [email](privacy-policy-data@lists.cs.princeton.edu).
#
# First, let's clone the repo so we can permissionlessly access everything offline with ease:

# %%
from pathlib import Path

REPO_PATH = Path("D:/MEGA/Programming/ML/Data/") / "privacy-policy-historical"

# !git clone $RAW_DATA_REPO_URL $REPO_PATH

# %% [markdown]
# This may take a pretty long while, since the dataset is quite large! ($\approx5.5~GB$)

# %% [markdown]
# ### 1.1.2 Data Processing (Transforming) and Saving (Loading)

# %% [markdown]
#
# The cloned repository contains the texts in Markdown format, each in a seperate file in some alphabetically ordered subfolder. The company name/URL is contained in the filename. The following steps are taken:
#
# 1) All paths to PPs are kept in a list (shuffled)
# 2) a dictionary is created which serves two purposes: indexing the tokens (saves memory) and being used later on in the Doc2Vec model
# 2) The training data is serialized and compressed on disk, to be easily loaded during training
#
# The documents are never loaded together into memory (one-by-one at all stages)

# %%
import random

# limit number of policies in corpus
N_PATHS = 100_000_000
# N_PATHS = 10_000

# get all privacy policy markdown file paths in a (random) list
print("Loading all privacy policy paths to memory... ", end="")
all_policy_paths = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name != "README.md"]
SEED = 42 + 1
rng = np.random.default_rng(seed=SEED)
print("Shuffling... ", end="")
rng.shuffle(all_policy_paths)

# get `N_PATHS` random policy paths
print(f"Keeping first {N_PATHS:,}... ", end="")
policy_paths = all_policy_paths[:N_PATHS]
print("Re-shuffling...")
rng.shuffle(all_policy_paths)
print(f"Loaded {len(policy_paths):,}/{len(all_policy_paths):,} privacy policy files.")

# %% [markdown]
# Create a fresh CorpusProcessor instance, build a `gensim.corpora import Dictionary` and process the entire corpus, all while streaming to/from disk.

# %%
from ppa.processing import CorpusProcessor

# SHOULD_REPROCESS = True
SHOULD_REPROCESS = False

CORPUS_DIR_PATH = Path.cwd().parent / "corpus"

# build and save dictionary from all documents, process all documents and serialize (compressed) the TaggedDocument objects to disk
if SHOULD_REPROCESS:
    # create a document processor with the paths
    cp = CorpusProcessor()
    cp.process_corpus(
        policy_paths,
        lemmatize=False,
        should_filter_stopwords=False,
        bigrams=True,
        bigram_threshold=0.75,
        n_below=None,
        no_above=1.0,
        min_percentile=0.5,
        max_percentile=99.5,
    )
else:
    cp = CorpusProcessor()
    print("Loaded existing CorpusProcessor")

Beep(1000, 500)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens). It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset:

# %%
pp_lengths = np.array(
    [len(tagged_doc.words) for tagged_doc in cp.generate_samples(n_samples=5_000)]
)

print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

with Plotter(
    suptitle="Policy Length Histogram", xlabel="Length (tokens)", ylabel="Bin Count"
) as ax:
    ax.hist(pp_lengths, bins=int(np.sqrt(pp_lengths.size)))

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

# %% [markdown]
# Now, let's take a look at the `gensim.corpora.Dictionary` we created from the entire corpus:

# %%
cp.dict_info()

# %% [markdown]
# First, let's plot a histogram of the most frequent tokens in each document. For a robustly varied corpus, I would expect a long-tailed distribution of per-doc appearances for the most common tokens. If the distribution decays too quickly, most tokens would effectively be just noise, and the corpus could be too simple.

# %%
token_ids = np.array(list(cp.dct.dfs.keys()))
doc_freqs = np.array(list(cp.dct.dfs.values()))

# sort according to frequencies in descending order
sorted_idxs = np.argsort(doc_freqs)[::-1]
token_ids = token_ids[sorted_idxs]
doc_freqs = doc_freqs[sorted_idxs]

# limit to `LIMIT_ID` most common
LIMIT_ID = len(cp.dct) // 5
token_ids = token_ids[:LIMIT_ID]
doc_freqs = doc_freqs[:LIMIT_ID]

with Plotter(
    suptitle=f"Document Counts\nof {LIMIT_ID:,} Most Common Tokens",
    xlabel="Token Rank (most common is 0)",
    ylabel="Token Per-Doc. Counts",
) as ax:
    ax.bar(token_ids, doc_freqs)

# %% [markdown]
# Now, let's visualize the most frequent words in the entire corpus as a word-cloud, as well as those in more frequent per document:

# %%
from ppa.display import display_wordcloud

print("Total Frequency Word-Cloud:")
_ = display_wordcloud(cp.dct)

print("Document Frequency Word-Cloud:")
_ = display_wordcloud(cp.dct, per_doc=True)

# %% [markdown]
# Notice how the images are quite similar - this is expected since the whole corpus is comprised of the same type of documents (privacy policies).
#
# We can try filtering out the least common and most common words from the Dictionary - words like "privacy", "information" or "data" are present in almost all PPs, and are probably not good differentiators between them.

# %%
from copy import deepcopy

filtered_dct = deepcopy(cp.dct)
filtered_dct.filter_extremes()

print(f"After filtering extremens, Dictionary contains {len(filtered_dct):,} unique tokens")

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
_ = display_wordcloud(filtered_dct)
print("Document Frequency Word-Cloud:")
_ = display_wordcloud(filtered_dct, per_doc=True)

# %% [markdown]
# What immediately stands out is the difference between the "total frequency" and "per-document frequency" before and after filtering the most common words. In the "total frequency" picture, we are just seeing less common words (the most common ones being ignored). In the "per-document frequency" picture, this enables us to see past the noise.

# %% [markdown]
# # 3. Aqcuiring and incorporating labels
# ## 3.1 ETL for Policy Ratings
# In order to acquire labels for the policies, turned to the only source I could found which had bulk amounts of ratings for privacy policies was [ToS;DR](https://tosdr.org/) - actually, ToS;DR provides a community-effort-driven database of combined rating for privacy policy AND terms of service, which I assumed would be similar to just the privacy policy rating. The labeling process was as follows:
# 1) Gather domain names from all available policy file-paths
# 2) Use the [ToS;DR search API](https://api.tosdr.org/search/v4/?query=google.com) to get a letter score - 'A' (best) to 'E' (worst) for all rated policies (regretably, only around 1% had ratings)
# 3) To simplify, ratings 'A' or 'B' were labeled 'good', the rest ('C', 'D' and 'E') were labeled 'bad'
# 4) The labels are kept in an index file, where each row contains a file start position (index), a domain name and a label
# 5) The index file is used to add secondary tags to the `TaggedDocument` objects provided to Doc2Vec during training, and for synchronizing between policies and their labels during scoring of the model.

# %%
from itertools import chain

# N = 10
N = np.inf

print(f"Getting URLs... ", end="")
# tags = [tagged_doc.tags[0] for idx, tagged_doc in enumerate(chain(train_data, test_data)) if idx < N]
tags = [fpath.stem for idx, fpath in enumerate(policy_paths) if idx < N]
print(f"{len(tags):,} URLs obtained.")

# %% [markdown]
# ETL

# %%
import asyncio
from ppa.processing import ToSDRDataLoader

# get all URLs for which I have PPs


# set flags
# FORCE_EXT = True
FORCE_EXT = False

# FORCE_TRANS = True
FORCE_TRANS = False

# Instantiate data-loading object
data_loader = ToSDRDataLoader()

# ratings_df = await data_loader.load_data(  # type: ignore
#     tags,
#     timeout_s=15,
#     force_extract=FORCE_EXT,
#     force_transform=FORCE_TRANS,
# )
ratings_df = pd.DataFrame({})
raise ValueError("Uncomment the 'await'!!!")

Beep(1000, 500)
print("Done.")

# how many datapoints are there
print("Number of records: ", len(ratings_df))

# how many labels do I have
print("Number of labels: ", ratings_df["rating"].notna().sum())

# how many possible duplicates do I have
print("Possible duplicates: ", len(ratings_df) - ratings_df["id"].nunique())

ratings_df.sample(10)

# %% [markdown]
# ## 3.2 Checking for Bias in Labeled Data

# %%
letter_count_dict = dict(sorted(Counter(ratings_df["rating"]).items()))

with Plotter(suptitle="Letter Rating Counts", xlabel="Letter Rating", ylabel="Counts") as ax:
    ax.bar(letter_count_dict.keys(), letter_count_dict.values())


# %% [markdown]
# As one might expect, good privacy policies are hard to come by. As such, I will, for now, label privacy policies as either 'good' ('A' or 'B' rating) vs. 'bad' ('C', 'D', or 'E' rating):

# %%
def relabel_rating(rating: str):
    """Doc."""

    if rating in "AB":
        return "good"
    elif rating in "CDE":
        return "bad"
    else:
        return rating


ratings_df["label"] = ratings_df["rating"].apply(relabel_rating)

label_count_dict = dict(sorted(Counter(ratings_df["label"]).items()))

with Plotter(suptitle="Label Counts", xlabel="Label", ylabel="Counts") as ax:
    ax.bar(label_count_dict.keys(), label_count_dict.values())

# %% [markdown]
# Perhaps this classification could work as anomaly detection ('good' policies being the anomaly)?

# %% [markdown]
# Finally, let's update the data index file with labels for URLs which have them:

# %%
if SHOULD_REPROCESS:
    url2label = ratings_df.set_index("tag")["label"].to_dict()
    cp.add_label_tags(url2label)
else:
    print("Using existing labels from index file.")

# %% [markdown]
# # 4. Modeling
#
# Next, we want to transform our documents into some vector space. There are many techniques which could be used, but a well established one which captures in-document token relationships (important for semantic context) is Doc2Vec. Training a Doc2Vec model over our data will provide a vector embedding of each document in our corpus. This would facillitate several pathways:
# 1) Enabling cluster analysis and visualization
# 2) Similarity comparison between PPs
# 3) Inferring vector embedding for non-corpus policies
# 4) Ultimately, this will enable using standard classification methods (tabular data)

# %% [markdown]
# ## 4.1 Model Training and Optimization

# %% [markdown]
# Split to train/test sets in a stratified fashion, i.e. keep the same label ratio (in this case the percentages of "good" and "bad" policies) in the data, train a specific Doc2Vec model (unsupervised), and save:

# %%
from ppa.estimators import D2VClassifier
from ppa.utils import timer
from datetime import datetime
import time

# SHOULD_FIT_MODEL = True
SHOULD_FIT_MODEL = False

if SHOULD_FIT_MODEL:

    # train/test splitting
    train_set, test_set = cp.generate_train_test_sets(
        seed=SEED,
    )
    print("train_set: ", train_set)
    print("test_set: ", test_set)

    # initialize classifier
    classifier = D2VClassifier(
        epochs=5,
        vector_size=84,  # 84?
        window=5,
        negative=20,
        train_score=True,
        seed=SEED,
        #     iterative_training=True,
        workers=psutil.cpu_count(logical=False) - 1,
    )

    # fit the model
    classifier.fit(train_set, train_set.labels)

    # score the model
    print("Balanced ACC: ", classifier.score(test_set, test_set.labels))

    # save
    dt_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    classifier.save_model(Path(f"D:/MEGA/Programming/ML/PPA/ppa/models/pp_d2v_{dt_str}.model"))

else:
    print("Not fitting.")

# %% [markdown]
# Cross validate

# %%
# from sklearn.model_selection import cross_validate

# toy_train_set = cp.generate_samples(
#     seed=SEED,
# )
# print(toy_train_set)

# CV = 4

# tic = time.perf_counter()
# logging.info("Starting CV...")
# scores = cross_validate(
#     D2VClassifier(
#         epochs=10,
#         vector_size=84,
#         window=5,
#         negative=20,
#         seed=SEED,
#         #         train_score=True,
#         #     workers=psutil.cpu_count(logical=False) - 1
#     ),
#     toy_train_set,
#     toy_train_set.labels,
#     cv=CV,
#     return_train_score=True,
#     verbose=10,
#     n_jobs=min(CV, psutil.cpu_count(logical=False) - 1),
# )
# logging.info(f"CV timing: {(time.perf_counter() - tic)/60:.1f} mins")
# print("Mean test score: ", np.nanmean(scores["test_score"]))
# scores_df = pd.DataFrame(scores)
# display(scores_df)

# %% [markdown]
# Train on the entire corpus

# %%
# from ppa.estimators import D2VClassifier
# from ppa.utils import timer
# from datetime import datetime
# import time

# # SHOULD_FIT_MODEL = True
# SHOULD_FIT_MODEL = False

# if SHOULD_FIT_MODEL:
#     # initialize classifier
#     classifier = D2VClassifier(
#         epochs=5,
#         vector_size=84,  # 84?
#         window=5,
#         negative=20,
#         train_score=True,
#         seed=SEED,
#         #     iterative_training=True,
#         workers=psutil.cpu_count(logical=False) - 1,
#     )

#     # fit the model
#     classifier.fit(toy_train_set, toy_train_set.labels)

#     # save
#     dt_str = datetime.now().strftime("%d%m%Y_%H%M%S")
#     classifier.save_model(Path(f"D:/MEGA/Programming/ML/PPA/ppa/models/pp_d2v_{dt_str}.model"))

# else:
#     print("Not fitting.")

# %% [markdown]
# Perform search to find best set of hyperparameters

# %%
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, GridSearchCV
# from scipy.stats import randint, uniform

# # Create a larger parameter grid with more combinations
# param_dist = {
#     "epochs": [1],
#     #     "vector_size": [74, 100, 154],
#     #     "window": [5, 7, 9],
#     #     "negative": [5, 10, 20],
# }

# N_SPLITS = 2
# HRS_SEED = randint(0, 2**32).rvs()

# # Update the hyperparameter search to use the pipeline
# search = GridSearchCV(
#     estimator=D2VClassifier(
#         epochs=1,
#         vector_size=84,
#         window=5,
#         negative=20,
#         random_state=cp.seed,
#         #         workers=psutil.cpu_count(logical=False) - 1,
#     ),
#     #     param_distributions=param_dist,
#     param_grid=param_dist,
#     #     n_candidates="exhaust",
#     verbose=4,
#     #     random_state=HRS_SEED,
#     cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cp.seed),
#     #     min_resources=15000,
#     #     n_jobs=min(N_SPLITS, psutil.cpu_count(logical=False) - 1),
#     refit=False,
# )

# # Fit the hyperparameter search on your training data
# tic = time.perf_counter()
# logging.info("Starting search...")
# search.fit(toy_train_set, toy_train_set.labels)
# logging.info(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# # Print the best hyperparameters
# logging.info(f"Best Hyperparameters: {search.best_params_}")

# # display the entire CV results, too
# display(pd.DataFrame(search.cv_results_))

# # Beep when search is done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# # Refit the model using the best found hyperparameters, with multiprocessing
# best_model = D2VClassifier(
#     **search.best_params_,
#     random_state=cp.seed,
#     workers=psutil.cpu_count(logical=False) - 1,
# )
# best_model.fit(train_set, train_set.labels)

# # Get the score of the best model on the test set
# best_model_score = best_model.score(test_set, test_set.labels)
# logging.info("Best Model Score:", best_model_score)

# # Beep again when best model is ready
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# Load existing model to estimator

# %%
from gensim.models.doc2vec import Doc2Vec

MODEL_DIR_PATH = Path.cwd().parent / "models"

THRESH = 0.5075

if not SHOULD_FIT_MODEL:
    classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")

    # load originally used train/test sets as SampleGenerator objects
    train_set, test_set = classifier.generate_train_test_sets()

    # score the model
    print("Balanced ACC: ", classifier.score(test_set, test_set.labels, threshold=THRESH))

    # Beep when done
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
else:
    print("Using recently fitted model.")

# %% [markdown]
# # 5 Doc2Vec Model Evaluation
#
# Doc2Vec is an unsupervised model, so finding a metric to evaluate it is not a straightforward task.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %% [markdown]
# ## 5.1 Sanity check
# As a first test of the model, a reasonable sanity check (adapted from that suggested by [Radim Řehůřek](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py) himself) would be to see if most vectors inferred for the policies the model was trained upon are most similar to the corresponding document vectors of the model itself.

# %%
# classifier.sanity_check(train_set, max_rank=10, plot=True)

# %% [markdown]
# ## 5.2 Visualizing the results using dimensionallity reduction to 2D
# We begin by getting vectors for all labeled samples in the train set

# %%
train_urls, labeled_train_vecs_tuple, train_labels = zip(
    *[
        (k, classifier.model.dv[k], train_set.indexed_file.key2poslabel[k][1])
        for k in classifier.label2keys["good"] + classifier.label2keys["bad"]
    ]
)
labeled_train_vecs = np.array(
    labeled_train_vecs_tuple
)  # unzipping returns a tuple, need to convert to 2D array

# %% [markdown]
# ### 5.2.1 PCA

# %%
from sklearn.decomposition import PCA
from ppa.display import display_dim_reduction

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(labeled_train_vecs)

display_dim_reduction(
    pca_result, "PCA", labels=train_labels, annots=train_urls, annots_sample=0.01, figsize=(10, 8)
)

# %% [markdown]
# ### 5.2.2 t-SNE

# %%
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=1,
    learning_rate=200,
    n_iter=1000,
    n_iter_without_progress=200,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(labeled_train_vecs)

display_dim_reduction(
    tsne_result,
    "t-SNE",
    labels=train_labels,
    annots=train_urls,
    annots_sample=0.01,
    figsize=(10, 8),
)

# %% [markdown]
# I cannot see any clear pattern separating "good" policies from "bad" ones. This doesn't mean the model isn't sensitive to the labels, only that the 2D visualizations don't seem to capture it.

# %% [markdown]
# ## 5.3 Devising a metric <a id='sec-5-3'></a>
# Perhaps the similarity between like-labled policies is lost in the dimensionality reduction. Let's try measuring the cosine similarity between the vectors directly. We can check out the distribution of my custom similarity scores separately for "good" and "bad" policies, as well as for the rest of the (unlabeled) policies:

# %%
# y_test_scores_labeled = classifier.decision_function(test_set_labeled)
# y_test_scores_unlabeled = classifier.decision_function(
#     test_set.sample(1_000, idxs=np.nonzero(~labeled_test_idxs)[0])
# )

# # Beep when done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
# with Plotter(
#     figsize=(10, 6),
#     xlabel="Similarity Scores",
#     ylabel="Bin Counts (Density)",
#     suptitle="Distribution of Similarity Scores",
# ) as ax:

#     # Plot the distribution of similarity scores
#     ax.hist(
#         good_scores := y_test_scores_labeled[labeled_test_tags == -1],
#         bins=int(np.sqrt(good_scores.size)),
#         density=True,
#         alpha=0.6,
#         label=f"Good ({good_scores.size})",
#     )
#     ax.hist(
#         bad_scores := y_test_scores_labeled[labeled_test_tags == 1],
#         bins=int(np.sqrt(bad_scores.size)),
#         density=True,
#         alpha=0.6,
#         label=f"Bad ({bad_scores.size})",
#     )
#     ax.hist(
#         y_test_scores_unlabeled,
#         bins=int(np.sqrt(len(y_test_scores_unlabeled))),
#         density=True,
#         alpha=0.3,
#         label=f"Unlabeled (sample of {len(y_test_scores_unlabeled):,})",
#         zorder=1,
#     )

#     ax.legend(title="Label (Num.)")
#     ax.grid(True)

# %% [markdown]
# This shows how good the model is at separating "good" policies from "bad" ones - the less overlap between the two histograms, the less errors it will make when classifying. We will, for now, score the model using balanced accuracy, using the above method to predict "good" policies for those scoring above a threshold (default 0.5). It is worth mentioning that this overlap unavoidable if there are incorrect labels or otherwise noisy data (bad preprocessing etc.)

# %% [markdown]
# ### 5.3.1 Investigating Specific Policies
# In the distributions above we can see significant overlap between good/bad policies, including some outliers which score far beyond their group mean. Lets try to understand why - this could be the key to devise a better model.

# %% [markdown]
# Let's get the URLs, labels and scores in a `DataFrame`:

# %%
# df = pd.DataFrame(
#     {
#         "url": [td.tags[0] for td in test_set_labeled],
#         "label": test_set_labeled.labels,
#         "score": y_test_scores_labeled,
#     }
# )

# %% [markdown]
# What seems interesting to test now is what the actual 'letter score' (from ToS;DR) is for each of the so-called outliers among the top scores - if the outliers are all 'C's, this could mean that them being labeled "bad" is a consequence of the binary labeling forced unto the 5 letter ratings:

# %%
# merged_df = pd.merge(df, ratings_df[["tag", "rating"]], left_on="url", right_on="tag", how="left")
# df["letter"] = merged_df["rating"]
# df.sort_values(by="score", ascending=False).iloc[:20]

# %% [markdown]
# So, while "bad"-labled policies with high scores are indeed 'C's, there are some 'D' there too. Let's check out the distributions as in 5.3, this time separated and colored by letter rating:

# %%
# with Plotter(
#     suptitle="Histograms of Scores by Letter Rating",
#     xlabel="Score",
#     ylabel="Bin Counts (Density)",
# ) as ax:

#     # Iterate through unique letters and plot histograms for each
#     for letter in sorted(df["letter"].unique(), reverse=True):
#         scores_for_letter = df[df["letter"] == letter]["score"]
#         ax.hist(
#             scores_for_letter,
#             bins=int(np.sqrt(len(scores_for_letter))),
#             density=True,
#             alpha=0.25,
#             label=f"{letter} ({len(scores_for_letter):,})",
#         )

#     ax.hist(
#         y_test_scores_unlabeled,
#         bins=int(np.sqrt(len(y_test_scores_unlabeled))),
#         density=True,
#         alpha=0.5,
#         label=f"Unlabeled (sample of {len(y_test_scores_unlabeled):,})",
#         zorder=-1,
#     )

#     ax.legend(title="Letter (Num.)")

# %% [markdown]
# We can see that all but the 'E'-rated policies have some tail in the high-scores! Furthermore, many 'A' and 'B'-rated policies have fairly low scores!

# %% [markdown]
# One solution for this could be simply wrong labels. Let's check out the worst-scored "good" policies. Since there are so little labeled policies in thee test set, we can check them all out together in one table:

# %%
# df[df["label"] == "good"].sort_values(by="score")

# %% [markdown]
# Since I recognize some of the URLs as having legitimately "good" privacy policies, I have no reason to suspect the labels, and I therefore blame my model. It could be that there's just not enough good privacy models in my data for Doc2Vec to pick up on the qualities of good privacy policies.

# %% [markdown]
# # Investigating Dictionary Word Vectors
# ## TODO: See suggestions (max norms, divide by label, similarity to docvec of each token...)

# %%
from sklearn.manifold import TSNE
from ppa.display import display_dim_reduction

word_vecs = np.asarray(classifier.model.wv.vectors)
words = np.asarray(classifier.model.wv.index_to_key)  # fixed-width numpy strings

tsne = TSNE(
    n_components=2,
    perplexity=0.95,
    learning_rate=300,
    n_iter=500,
    n_iter_without_progress=250,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(word_vecs)

# Beep when done
Beep(1000, 1000)

# %%
display_dim_reduction(tsne_result, "t-SNE", annots=words, annots_sample=0.01, figsize=(10, 8))

# %% [markdown]
# ## Testing stuff with .wv and the Dictionary

# %% [markdown]
# getting the all vocabulary words and their vectors sorted by norm (using .wv)

# %%
norm_sorted_idxs = np.argsort(np.linalg.norm(word_vecs, axis=1))
norm_sorted_word_vecs = word_vecs[norm_sorted_idxs]
norm_sorted_words = words[norm_sorted_idxs]

# %% [markdown]
# getting the all vocabulary words sorted by prevalence (using the Dictionary)

# %%
word_count_dict = {cp.dct[id]: count for id, count in getattr(cp.dct, "cfs").items()}
prevalence_sorted_dict_words, _ = zip(*sorted(word_count_dict.items(), key=lambda item: item[1]))
prevalence_sorted_dict_words = np.array(prevalence_sorted_dict_words)

# %% [markdown]
# To find correlation between the two, lets turn the words to their IDs (so we can work with numbers), and correlate:

# %%
# Convert to IDs using the Dictionary
norm_sorted_ids = np.array([cp.dct.token2id[word] for word in norm_sorted_words], dtype=float)
prevalence_sorted_dict_ids = np.array(
    [cp.dct.token2id[word] for word in prevalence_sorted_dict_words], dtype=float
)

# Calculate the correlation coefficient between the values in the lists
correlation = np.corrcoef(norm_sorted_ids, prevalence_sorted_dict_ids)[0, 1]

print(f"The correlation between the values in the lists is: {correlation:.2f}")

# %% [markdown]
# Some correlation exists, but it's definitely no a direct connection.

# %% [markdown]
# Let's play around with WMD (word-mover's distant)

# %%
good_docs, _ = train_set.get_labeled("good", tokens_only=True)
bad_docs, _ = train_set.get_labeled("bad", tokens_only=True)

# %%
classifier.model.wv.wmdistance(good_docs[0].words, good_docs[5].words)

# %% [markdown]
# ## Word-vec / Doc-vec interaction

# %% [markdown]
# let's experiment on some labeled policy

# %%
labeled_train_set = train_set.get_labeled()

# %%
doc = labeled_train_set[1]
words = np.array(doc.words)

# %% [markdown]
# taking a look:

# %%
print("Domain name: ", doc.tags[0])
" ".join(words)

# %% [markdown]
# let's see if I can perform augmentation using similar words

# %%
# aug_protonmail_pp = []
# for word in words:
#     print(word)
#     aug_protonmail_pp
#     print(classifier.model.wv.most_similar(word, topn=3))

# %% [markdown]
# get the document vector

# %%
doc_dv = classifier.model.dv[doc.tags[0]]

# %% [markdown]
# get the document tokens as word-vectors

# %%
doc_wv = np.array([classifier.model.wv[word] for word in words])
doc_wv.shape

# %% [markdown]
# checking the similarity between the document vector and the sum/mean of all its word-vectors

# %%
classifier.model.dv.cosine_similarities(doc_dv, [doc_wv.mean(axis=0)])[0]

# %% [markdown]
# checking the similarity between each word vector and the doc vector

# %%
doc_word_sims = classifier.model.dv.cosine_similarities(doc_dv, doc_wv)

# %% [markdown]
# let's try sorting the words according to the similarity

# %%
doc_word_sims_sorted_idxs = np.argsort(doc_word_sims)[::-1]
sim_sorted_words_duplicates = words[doc_word_sims_sorted_idxs]

# TEST
print(sim_sorted_words_duplicates[:10])

# %% [markdown]
# remove the duplicates while keeping the order

# %%
sim_sorted_words, indices = np.unique(sim_sorted_words_duplicates, return_index=True)
sim_sorted_words[np.argsort(indices)]

# %% [markdown]
# Taking a broader look, we see that there is no distinct similarity between the mean word vector and the document vector

# %%
# TESTESTEST - SAME FOR ENTIRE LABELED CORPUS - SEE IF ALL MEAN WV AVERAGE OUT TO 0?
key2label = {k: l for k, l in zip(labeled_train_set.keys, labeled_train_set.labels)}

for td in labeled_train_set.sample(10):
    doc_dv = classifier.model.dv[td.tags[0]]
    doc_wv = np.array([classifier.model.wv[word] for word in td.words])
    dv2mean_wv_sim = classifier.model.dv.cosine_similarities(doc_dv, [doc_wv.mean(axis=0)])[0]
    print(key2label[td.tags[0]], dv2mean_wv_sim)

# %% [markdown]
# ## A different angle - let's try to find and compare the words most similar to the mean "good"/"bad" vectors, vs. those most common in the "good"/"bad" documents
#
# ### 1) Most common per-doc tokens for good/bad policies

# %%
# getting all "good" samples, lazily
good_train_set = labeled_train_set.get_labeled("good", tokens_only=True)
bad_train_set = labeled_train_set.get_labeled("bad", tokens_only=True)

# creating a Dictionary from all of them
from gensim.corpora.dictionary import Dictionary

good_dct = Dictionary(good_train_set)
bad_dct = Dictionary(bad_train_set)

# TEST - displaying word clouds for all good/bad policies
# _ = display_wordcloud(good_dct, per_doc=False)
# _ = display_wordcloud(bad_dct, per_doc=False)

# %% [markdown]
# Since they dictionaries will share many common words, let's use a filtered version of the total corpus dictionary to filter both

# %%
# no_above = 0.5

# filtered_corpus_dct = deepcopy(cp.dct)
# all_dct_tokens = set([filtered_corpus_dct[id] for id in filtered_corpus_dct.keys()])
# filtered_corpus_dct.filter_extremes(no_above=no_above)
# filtered_dct_tokens = set([filtered_corpus_dct[id] for id in filtered_corpus_dct.keys()])
# print(f"will filter {len(all_dct_tokens - filtered_dct_tokens)} tokens")
# banal_tokens = [token_ for token_ in all_dct_tokens - filtered_dct_tokens]
# print(banal_tokens)

# %% [markdown]
# Now, let's remove those banal tokens from both the good and bad dicts

# %%
# # get the id of each banal token for each Dictionary (good/bad)
# good_banal_ids = [good_dct.token2id[token_] for token_ in banal_tokens]
# bad_banal_ids = [bad_dct.token2id[token_] for token_ in banal_tokens]

# # remove the banal ids from each Dictionary
# good_dct.filter_tokens(good_banal_ids)
# bad_dct.filter_tokens(bad_banal_ids)

# %% [markdown]
# a different method would be to remove the common tokens

# %%
# get the id of each common token for each Dictionary (good/bad)
common_tokens = set(good_dct.values()) & set(bad_dct.values())
good_common_ids = [good_dct.token2id[token_] for token_ in common_tokens]
bad_common_ids = [bad_dct.token2id[token_] for token_ in common_tokens]

# remove the banal ids from each Dictionary
good_dct.filter_tokens(good_common_ids)
bad_dct.filter_tokens(bad_common_ids)

# %% [markdown]
# Let's checkout wordclouds to see a difference

# %%
print("'Good'")
_ = display_wordcloud(good_dct, per_doc=True)
print("'Bad'")
_ = display_wordcloud(bad_dct, per_doc=True)

# %% [markdown]
# Now lets keep the sorted tokens for the good/bad:

# %%
# good
good_tokens = np.array([good_dct[id] for id in good_dct.dfs.keys()])
good_dfs = np.array(list(good_dct.dfs.values()))
good_dfs_sorted_idxs = np.argsort(good_dfs)[::-1]
good_sorted_tokens = good_tokens[good_dfs_sorted_idxs]

# bad
bad_tokens = np.array([bad_dct[id] for id in bad_dct.dfs.keys()])
bad_dfs = np.array(list(bad_dct.dfs.values()))
bad_dfs_sorted_idxs = np.argsort(bad_dfs)[::-1]
bad_sorted_tokens = bad_tokens[bad_dfs_sorted_idxs]

# TEST
print("most common good tokens: ", good_sorted_tokens[:20])
print()
print("most common bad tokens: ", bad_sorted_tokens[:20])

# %% [markdown]
# This is the most promising one, I guess - but there are not enough good documents to get a good statistic for the most common good-only tokens...

# %% [markdown]
# ### 2) Seeing which words are most similar to the mean good/bad document vectors

# %%
mean_good_vec = classifier.mean_labeled_model_vector("good")
mean_bad_vec = classifier.mean_labeled_model_vector("bad")

mean_good_vs_wv_sims_sorted_idxs = np.argsort(
    classifier.model.dv.cosine_similarities(mean_good_vec, classifier.model.wv.vectors)
)[::-1]
sorted_good_sim_words = [
    classifier.model.wv.index_to_key[idx] for idx in mean_good_vs_wv_sims_sorted_idxs
]

mean_bad_vs_wv_sims_sorted_idxs = np.argsort(
    classifier.model.dv.cosine_similarities(mean_bad_vec, classifier.model.wv.vectors)
)[::-1]
sorted_bad_sim_words = [
    classifier.model.wv.index_to_key[idx] for idx in mean_bad_vs_wv_sims_sorted_idxs
]

# %% [markdown]
# Check out the most similar words to the mean good and mean bad document vectors, respectively

# %%
print(sorted_good_sim_words[:20])
print()
print(sorted_bad_sim_words[:20])

# %% [markdown]
# Compare to the prevalence-based collection:

# %%
print(good_sorted_tokens[:20])
print()
print(bad_sorted_tokens[:20])

# %%
raise RuntimeError("STOP HERE!")

# %% [markdown]
# # 6. Attaching a Classifier
# While it seems that the Doc2Vec model by itself is doing a fair job separating good privacy policies from bad ones, I am almost certain that more complex classification methods could make better use of the generated the vector embeddings as features for a binary classifier. Since I already have an transformer class for Doc2Vec, it should have been relatively easy to create a pipeline and attach an existing classifier.

# %% [markdown]
# # TODO: after extensive hyperparameter tuning attempts with XGB, I cannot seem to be able to reach a balanced accuracy score of over 0.7, which is significantly less then what I reach with my simple unsupervised good/bad thresholding method (Doc2Vec) - I guess there's just not enough examples! I could try plotting the learnig curve for my best XGB model to put some weight behind that claim (see if I'm converging anywhere with more samples).

# %% [markdown]
# Let's start by creating a simple pipeline, see if t works at all and go from there. we can start by using the already trained mode (should be refitted when searching for optimal hyperparameters)

# %%
# from sklearn.pipeline import Pipeline
# from ppa.estimators import D2VTransformer, ScoredIsolationForest

# # load pretrained Doc2Vec model
# d2vtrans = D2VTransformer.load_model(Path("D:/MEGA/Programming/ML/PPA/ppa/models/pp_d2v.model"))

# # get original train set
# original_train_set, original_test_set = d2vtrans.generate_train_test_sets()

# # transform the train_set using d2vtrabs
# original_train_set_vec = np.array(
#     [d2vtrans.model.dv.get_vector(key) for key in original_train_set.keys]
# )

# # create a pipeline
# pipe = Pipeline(
#     [
#         (
#             "clf",
#             ScoredIsolationForest(
#                 n_estimators=50,
#                 n_jobs=psutil.cpu_count(logical=False) - 1,
#             ),
#         )
#     ],
# )

# # fit it to the pre-transformed train_set_vec
# pipe.fit(original_train_set_vec)

# %% [markdown]
# or using the supervised state-of-the-art:

# %%
from ppa.estimators import D2VTransformer

# load pretrained Doc2Vec model
d2vtrans = D2VTransformer.load_model(Path("D:/MEGA/Programming/ML/PPA/ppa/models/pp_d2v.model"))

# get original train set
original_train_set, original_test_set = d2vtrans.generate_train_test_sets()

# IsolationForest - transform the train_set using d2vtrabs
original_train_set_vec = np.array(
    [d2vtrans.model.dv.get_vector(key) for key in original_train_set.keys]
)

# # XGBClassifier - get just the labeled parts of it (since this is supervised classification, which must have labeles)
# original_train_set_labeled = original_train_set.get_labeled()
# original_test_set_labeled = original_test_set.get_labeled()

# # transform the labeled train_set using d2vtrans
# original_train_set_labeled_vec = np.array(
#     [d2vtrans.model.dv.get_vector(key) for key in original_train_set_labeled.keys]
# )

# # transform the labeled test set using d2vtrans
# original_test_set_labeled_vec = d2vtrans.transform(original_test_set_labeled)

# %%
from sklearn.pipeline import Pipeline

# from ppa.estimators import ScoredXGBClassifier
from ppa.estimators import ScoredIsolationForest

# from sklearn.feature_selection import RFECV

# create a pipeline
pipe = Pipeline(
    [
        (
            "clf",
            ScoredIsolationForest(
                n_estimators=100,
                max_samples=1.0,
                max_features=1.0,
                contamination=0.05,
                bootstrap=False,
                random_state=SEED,
                n_jobs=psutil.cpu_count(logical=False) - 1,
            ),
        )
        #         (
        #             'sel',
        #             RFECV(
        #                 ScoredXGBClassifier(random_state=SEED),
        #                 step=1,
        #                 cv=4,
        #                 min_features_to_select=1,
        #             ),
        #         ),
        #         (
        #             'clf',
        #             ScoredXGBClassifier(
        #                 booster="gbtree",
        #                 max_depth=2,
        #                 learning_rate=0.12,
        #                 n_estimators=570,
        #                 subsample=0.1,
        #                 colsample_bytree=0.3,
        #                 gamma=1.2,
        #                 min_child_weight=1.48,
        #                 random_state=SEED,
        #             ),
        #         ),
    ]
)

# IsolationForest - fit it to the pre-transformed train_set_vec
pipe.fit(original_train_set_vec, original_train_set.labels)
# score on the test set
print(pipe.score(original_train_set_vec, original_train_set.labels))

# # XGBClassifier - fit it to the pre-transformed train_set_vec
# pipe.fit(original_train_set_labeled_vec, original_train_set_labeled.labels)
# # score on the test set
# print(pipe.score(original_test_set_labeled_vec, original_test_set_labeled.labels))

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# plot learning curve for "best" model

# %%
# from sklearn.model_selection import LearningCurveDisplay

# N_SPLITS = 4
# MAX_CV_SAMPS = int(len(original_train_set_labeled_vec) - len(original_train_set_labeled_vec) / N_SPLITS)
# train_sizes = np.logspace(2.0, np.log(MAX_CV_SAMPS) / np.log(10), num=50, dtype=int) # 100 to MAX_CV_SAMPS (log scale)
# with Plotter() as ax:
#     LearningCurveDisplay.from_estimator(
#         pipe,
#         original_train_set_labeled_vec,
#         original_train_set_labeled.labels,
#         train_sizes=train_sizes,
#         cv=N_SPLITS,
#         n_jobs=min(N_SPLITS, psutil.cpu_count(logical=False) - 1),
# #         scoring=SCORING,
#         ax=ax
#     )

# %% [markdown]
# Hyperparamter search for XGBoost

# %%
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
# from scipy.stats import randint, uniform, loguniform, norm

# # Create a larger parameter grid with more combinations
# param_dist = {
#     'clf__max_depth': randint(2, 8),
#     'clf__learning_rate': [0.01, 0.05, 0.1],
#     'clf__n_estimators': randint(300, 601),
#     'clf__subsample': norm(loc=0.1, scale=0.02),
#     'clf__colsample_bytree': norm(loc=0.313, scale=0.02),
#     'clf__gamma': norm(loc=1.0, scale=0.01),
#     "clf__min_child_weight": norm(loc=1.5, scale=0.02),
# }
# # param_grid = {
# #     'clf__max_depth': [2, 3],
# #     'clf__learning_rate': [0.125, 0.15],
# #     'clf__n_estimators': [450, 500, 550],
# #     'clf__subsample': [0.1],
# #     'clf__colsample_bytree': [0.3],
# #     'clf__gamma': [1.2],
# # #     "clf__booster": ["gbtree"],
# #     "clf__min_child_weight": [1.3, 1.5, 1.7],
# # }

# N_SPLITS = 4

# # Update the hyperparameter search to use the pipeline
# search = RandomizedSearchCV(
#     estimator=pipe,
#     param_distributions=param_dist,
# #     param_grid=param_grid,
#     return_train_score=True,
#     verbose=4,
#     cv=N_SPLITS,
#     n_jobs=min(N_SPLITS, psutil.cpu_count(logical=False) - 1),
#     refit=True,
#     random_state=SEED,
#     n_iter=10,
# )

# # Fit the hyperparameter search on your training data
# tic = time.perf_counter()
# logging.info("Starting search...")
# search.fit(original_train_set_labeled_vec, original_train_set_labeled.labels)
# logging.info(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# # Print the best hyperparameters
# logging.info(f"Best Hyperparameters: {search.best_params_}")

# # display selected CV results, too
# df = pd.DataFrame(search.cv_results_)
# columns = [
#     'rank_test_score',
#     'mean_test_score',
#     'mean_train_score',
#     'param_clf__colsample_bytree',
#     'param_clf__gamma',
#     'param_clf__learning_rate',
#     'param_clf__max_depth',
#     'param_clf__min_child_weight',
#     'param_clf__n_estimators',
#     'param_clf__subsample',
# ]
# display(df[columns])

# # Beep when search is done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# # Get the score of the best model on the test set
# best_model_score = search.best_estimator_.score(
#     original_test_set_labeled_vec, original_test_set_labeled.labels
# )
# logging.info(f"Best Model Score: {best_model_score}")

# # Beep again when best model is ready
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
# # TESTESTEST

# from scipy.stats import norm, gamma, poisson, expon, beta, chi2, lognorm, triang, uniform, nbinom


# data_dists = {
#     "normal": norm.rvs(loc=0, scale=1, size=10_000),
# #     "gamma": gamma.rvs(loc=50, a=1.5, scale=100, size=100_000),
# #     "nbinom": nbinom.rvs(1.5, 0.01, loc=50, size=10_000),
# #     "exponential": expon.rvs(scale=0.1, size=10_000),
# #     "uniform": uniform.rvs(loc=0.5, scale=0.5, size=10_000),
# }
# with Plotter() as ax:

#     for label, data_dist in data_dists.items():
#         ax.hist(data_dist, bins=round(np.sqrt(data_dist.size)), alpha=1/len(data_dists), label=label)
#         print(f"dist effective range: {np.median(data_dist) - np.std(data_dist):.2f}-{np.median(data_dist) + np.std(data_dist):.2f}")
#         print(f"dist total range: {min(data_dist):.2f}/{max(data_dist):.2f}")
#         print(f"dist median: {np.median(data_dist):.2f}")
#     ax.legend()

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, GridSearchCV
from scipy.stats import randint, uniform, nbinom

# Create a larger parameter grid with more combinations
param_dist = {
    "clf__n_estimators": nbinom(1.5, 0.01, loc=50),  # effective range: 46.48-289.52
    "clf__max_samples": uniform(loc=0.001, scale=(0.01 - 0.001)),  # 0.001 - 0.01
    "clf__contamination": uniform(loc=0.001, scale=(0.1 - 0.001)),  # 0.001 - 0.1
    "clf__max_features": uniform(loc=0.5, scale=(1 - 0.5)),  # 0.5 - 1
    "clf__bootstrap": [True, False],
}

N_SPLITS = 4
SEED = 42  # Assuming SEED is defined somewhere

# Update the hyperparameter search to use the pipeline
search = HalvingRandomSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    factor=3,
    min_resources=15000,
    #     aggressive_elimination=True,
    verbose=4,
    cv=N_SPLITS,
    #     cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    n_jobs=min(N_SPLITS, psutil.cpu_count(logical=False) - 1),
    refit=True,
    random_state=SEED,
)

# Fit the hyperparameter search on your training data
tic = time.perf_counter()
logging.info("Starting search...")
search.fit(original_train_set_vec, original_train_set.labels)
logging.info(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# Print the best hyperparameters
logging.info(f"Best Hyperparameters: {search.best_params_}")

# display the entire CV results, too
display(pd.DataFrame(search.cv_results_))

# Beep when search is done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# transform test_set using pretrained Doc2Vec
original_test_set_labeled = original_test_set.get_labeled()
original_test_set_labeled_vec = d2vtrans.transform(original_test_set_labeled)

# Get the score of the best model on the test set
best_model_score = search.best_estimator_.score(
    original_test_set_labeled_vec, original_test_set_labeled.labels
)
logging.info(f"Best Model Score: {best_model_score}")

# Beep again when best model is ready
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# # 7 Considering Upsampling and Pseudo-Labeling
#
# Being unable to increase my model's balanced-accuracy metric above about 0.8 using hyperparameter tuning or better data pre-processing (I have tinkered with both for a while now). What could be the reason is the imbalance existing in the data - there are about 17 times more "bad" labels than "good" ones, and I expect about the same imbalance in the unlabeled corpus. I can attempt to improve the data in two (possibly combined) methods:
# * **Upsampling (mixing-in copies of) "good"-labeled policies in the training data**: This might make the model 'understand' "good" policies better, as the data will be less biased (need to watch for overfitting of the "good" policies)
# * **Pseudo-labeling unlabeled training data using high/low thresholds for prediction**: This would make the scoring more robust as model summary vectors (specifically the "good" one) are averaged over more samples, enabling finer tuning according to score. This could also potentially enable supervised classification later on.
#
# If upsampling appears to be improving the metric, I could iteratively label more polcies (good & bad) and upsample further, as long as things keep improving.

# %% [markdown]
# ## 7.1 Pseudo-Labeling
#
# ### 7.1.1 Score Thresholds
# First, let's see how many candidates I have for pseudo-labeling. Taking a look again at the histograms in [5.3](#sec-5-3), I can "safely" set the thresholds at below 0.46 for "bad" policies and above 0.55 for "good" policies. Let's see how many new labeles I would be able to obtain this way. Let's try plotting the score distributions for the training set.
#
# We begin by inferring vectors for all training policies and getting the scores (this might take couple of minutes):

# %%
# get indices for labeled training samples
train_labels, labeled_train_idxs = classifier.valid_labels(train_set.labels)

# getting the
y_train_scores_labeled = classifier.decision_function(
    train_set.sample(idxs=np.nonzero(labeled_train_idxs)[0])
)
y_train_scores_unlabeled_sample = classifier.decision_function(
    train_set.sample(5_000, idxs=np.nonzero(~labeled_train_idxs)[0])
)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
with Plotter(
    figsize=(10, 6),
    xlabel="Similarity Scores",
    ylabel="Bin Density",
    suptitle="Distribution of Similarity Scores",
) as ax:

    # Plot the distribution of similarity scores
    ax.hist(
        good_scores := y_train_scores_labeled[train_labels == -1],
        bins=int(np.sqrt(good_scores.size)),
        density=True,
        alpha=0.6,
        label=f"Good ({good_scores.size:,})",
    )
    ax.hist(
        bad_scores := y_train_scores_labeled[train_labels == 1],
        bins=int(np.sqrt(bad_scores.size)),
        density=True,
        alpha=0.6,
        label=f"Bad ({bad_scores.size:,})",
    )
    ax.hist(
        y_train_scores_unlabeled_sample,
        bins=int(np.sqrt(len(y_train_scores_unlabeled_sample))),
        density=True,
        alpha=0.3,
        label=f"Unlabeled (sample of {len(y_train_scores_unlabeled_sample):,})",
        zorder=1,
    )

    ax.legend(title="Label (Num.)")
    ax.grid(True)

# %% [markdown]
# Reassuringly, the distributions look similar to the test distributions ([5.3](#sec-5-3)).
#
# Now, let's estimate how many new good/bad pseudo-labeled policies we can hope for, in a preliminary iteration. For this, we can just use the sample we just inferred for the above plot, and divide it by the sampling ratio.
#
# Now let's set the thresholds to "safe" values according to the "good"/"bad" distributions above:

# %%
print(
    f"Unlabeled score range: {min(y_train_scores_unlabeled_sample):.2f}-{max(y_train_scores_unlabeled_sample):.2f}"
)

THRESH_GOOD = 0.57
THRESH_BAD = 0.42

sampling_ratio = y_train_scores_unlabeled_sample.size / len(train_set)

print(
    f"Potential 'good' pseudo-labels estimate: {sum(y_train_scores_unlabeled_sample > THRESH_GOOD) / sampling_ratio:.0f}"
)
print(
    f"Potential 'bad' pseudo-labels estimate: {sum(y_train_scores_unlabeled_sample < THRESH_BAD) / sampling_ratio:.0f}"
)

# %% [markdown]
# So it appears we can squeeze out more labeles if necessary, but it remains to be seen how accurate they will be.

# %% [markdown]
# ### 7.1.2 Most-Similar Labeled Policies
#
# A different Idea, utilizing Doc2Vec's prized similarity methods, could be checking out the most similar labeled policies for each unlabeled policy - the general idea is that if for a specific unlabeled policiy, the N most similar labeled documents are of a specific label (and above a certain similarity score), we could label it the same. We should try it out with a small sample of unlabeld policies first:

# %% [markdown]
# # TODO: TRY THIS (below is old code for a starting point/idea)

# %%
# from gensim.models import Doc2Vec
# import gensim
# from gensim.models.doc2vec import TaggedDocument

# # Load your pre-trained Doc2Vec model
# model = Doc2Vec.load("your_model_path")

# # Tokenize your test privacy policy, for example:
# test_policy_tokens = ["list", "of", "tokenized", "words", "in", "the", "test", "policy"]

# # Infer the vector for your test document
# test_vector = model.infer_vector(test_policy_tokens)

# # Find the top N most similar documents
# similar_documents = model.docvecs.most_similar([test_vector], topn=N)

# # Initialize a list to store the labels of the most similar documents
# nearest_labels = []

# # Loop through the most similar documents
# for doc_id, similarity in similar_documents:
#     # Retrieve the TaggedDocument based on the doc_id
#     similar_doc = TaggedDocument.load("your_tagged_documents_path/" + doc_id + ".tagged")

#     # Check if the similar document has a label
#     if similar_doc.tags[0] in labeled_documents:
#         nearest_labels.append(labeled_documents[similar_doc.tags[0]])

#     # If you have reached the desired number of labeled neighbors
#     if len(nearest_labels) >= N:
#         break

# # Now you have the labels from the nearest neighbors
# # If none of them have labels, you can set a default category

# # Example: If all the neighbors are unclassified, set a default label
# if not nearest_labels:
#     default_label = "Unclassified"
# else:
#     # Choose the label from the first neighbor (the most similar one)
#     default_label = nearest_labels[0]

# print("Predicted Label:", default_label)

# %% [markdown]
# Let's try improving the model with upsampling first.

# %% [markdown]
# ## 7.2 Upsampling
# Before pseudo-labeling, let's try the more simple upsampling technique, and see if we can get a better model before scraping for more labels. I have added a feature which allows all samples of a certain label ("good", in this case) to repeat `upsampling_factor` times in SampleGenerator iterations. Let's try refitting the model with the upsampled training corpus, and see if it scores better on the test set.
#
# # NOTE - This does not help accoring to CV
