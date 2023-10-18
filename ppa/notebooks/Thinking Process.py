# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# ### TODO: try removing PPs of extreme length (according to histogram) - see how it effects  clustering - it definitely affects training time! Compare say cutting at 5000 token vs. 2000 tokens.
# ### TODO: Consider better preprocessing of documents - a better understanding of the document structure might be needed (use HTML instead of markdown? or identify headers etc. with special tokens?)
# ### TODO: Implement cross-validation training, once a metric is devised
# ### TODO: consider memory-mapping the train/test data - check the size in MBytes of the array (say 5000 integers per row, and 100K rows... sounds like alot)

# %%
# # %reload_ext autoreload
# # %autoreload 2

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
# I then tried to look for publicly available databases of privacy policies, and surprisingly found a great one immediately, collected by the authors of this [paper](https://www.linkedin.com/jobs/view/3698341388/). Not only does it contain about a million policies, it also containg archived older versions of each! (which I may not need, but are nice to have). The data is available from [GitHub](https://github.com/citp/privacy-policy-historical/tree/master):

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
# get all privacy policy markdown file paths in a (random) list
print("Loading all privacy policy paths to memory... ", end="")
policy_paths = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name != "README.md"]
print(f"Found {len(policy_paths):,} privacy policy files.")

# # TESTEST - use only N paths!
# N_PATHS = 1_000
# print(f"\nWARNING! USING JUST {N_PATHS:,} PATHS!")
# policy_paths = policy_paths[:N_PATHS]

# %% [markdown]
# Create a fresh CorpusProcessor instance, build a `gensim.corpora import Dictionary` and process the entire corpus, all while streaming to/from disk.

# %%
from ppa.utils import timer
from ppa.ppa import CorpusProcessor

# SHOULD_REPROCESS = True
SHOULD_REPROCESS = False

SEED = 42
MODEL_DIR_PATH = Path.cwd().parent / "models"

# create a document processor with the paths
cp = CorpusProcessor(
    policy_paths,
    MODEL_DIR_PATH,
    should_filter_stopwords=True,
    min_tokens=40,
    max_tokens=5000,
    seed=SEED,
)

# build and save dictionary from all documents, process all documents and serialize (compressed) the TaggedDocument objects to disk
cp.process(force=SHOULD_REPROCESS)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens). It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset:

# %%
from ppa.display import Plotter
import numpy as np

N = cp.total_samples // 10
pp_lengths = np.array([len(tagged_doc.words) for tagged_doc in cp.generate_samples(N)])

print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

with Plotter() as ax:
    ax.hist(pp_lengths, int(np.sqrt(pp_lengths.size)))

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

# %% [markdown]
# Now, let's take a look at the `gensim.corpora.Dictionary` we created from the entire corpus:

# %%
print(
    f"Dictionary was created by processing {cp.dct.num_pos:,} tokens from a corpus of {cp.dct.num_docs:,} documents."
)
print(f"It contains {len(cp.dct):,} unique tokens.")
print(f"Each document, on average, contains {cp.dct.num_nnz // cp.dct.num_docs:,} unique tokens.")

# %% [markdown]
# We can use it to visualize the data a bit.
#
# First, let's visualize the most frequent words in the entire corpus as a word-cloud:

# %%
from ppa.display import display_wordcloud

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
display_wordcloud(cp.dct)

# %% [markdown]
# Now, let's also visualize the most frequent words document-wise:

# %%
print("Document Frequency Word-Cloud:")
display_wordcloud(cp.dct, per_doc=True)

# %% [markdown]
# Notice how the images are quite similar.
#
# We can try filtering out the least common and most common words from the Dictionary - words like "privacy", "information" or "data" are present in almost all PPs, and are probably not good differentiators between them.

# %%
from copy import deepcopy

filtered_dct = deepcopy(cp.dct)
filtered_dct.filter_extremes()

print(f"After filtering extremens, Dictionary contains {len(filtered_dct):,} unique tokens")

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
display_wordcloud(filtered_dct)
print("Document Frequency Word-Cloud:")
display_wordcloud(filtered_dct, per_doc=True)

# %% [markdown]
# What immediately stands out is the difference between the "total frequency" and "per-document frequency" before and after filtering the most common words. In the "total frequency" picture, we are just seeing less common words (the most common ones being ignored). In the "per-document frequency" picture, this enables us to see past the noise.
#
# Look for the frequency of specific words (such as "url"):

# %%
# The token you want to search for
target_token = "url"

# Iterate through each document and check if the token is in the document
for tagged_doc in cp.generate_samples(N):
    if target_token in tagged_doc.words:
        print(f"{tagged_doc.tags[0]}:\n")
        print(" ".join(tagged_doc.words))
        break

# %% [markdown]
# # 3. Modeling
#
# Next, we want to transform our documents into some vector space. There are many techniques which could be used, but a well established one which captures in-document token relationships (important for semantic context) is Doc2Vec. Training a Doc2Vec model over our data will provide a vector embedding of each document in our corpus. This would facillitate several pathways:
# 1) Enabling cluster analysis and visualization
# 2) Similarity comparison between PPs
# 3) Inferring vector embedding for non-corpus policies

# %% [markdown]
# ## 3.1 Training the (unsupervised) Doc2Vec model (or loading the last trained one)
#
# First, let's split the data to train/test sets

# %%
N = cp.total_samples // 10
TEST_FRAC = 0.2
train_data, test_data = cp.generate_train_test_sets(N, test_frac=TEST_FRAC)
print(f"Using {N:,} Samples ({N/cp.total_samples:.1%} of available samples, {TEST_FRAC:.1%} test).")

# %% [markdown]
# ## 3.2 Training using the training set

# %%
import time
import multiprocessing as mp
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# define save/load path
MODEL_PATH = MODEL_DIR_PATH / "privacy_policy_doc2vec.model"

# SHOULD_RETRAIN = True
SHOULD_RETRAIN = False

if not SHOULD_RETRAIN:
    # load the last trained model
    model = Doc2Vec.load(str(MODEL_PATH))

    # Done
    print("Model Loaded")
    # TODO: print some model details

else:
    # Initialize and train the Doc2Vec model
    model_kwargs = {
        "vector_size": 300,
        "window": 5,
        "hs": 1,
        "epochs": 10
        #         "workers": mp.cpu_count(),
    }
    model = Doc2Vec(**model_kwargs)

    # Build vocabulary
    print("Building vocabulary... ", end="")
    model.build_vocab(train_data)

    # Train the model
    print(f"Training model... ", end="")
    tic = time.perf_counter()
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the trained model for future use
    print(f"Saving to '{MODEL_PATH}'... ", end="")
    model.save(str(MODEL_PATH))
    print("Done!")

    # Done!
    print(f"Training timing: {(time.perf_counter() - tic)/60:.1f} mins")
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms


# %% [markdown]
# ## 3.2 Sanity check
# check that documents are most similar to themselves. I do not expect/desire to optimize the fraction of model-inferred documents which are most similar to themselves, as this might mean the model is overfitting. Instead, this is just a test to see that the model does something logical.
#
# It is worth mentioning that while purposfully attempting to overfit a small subset of 1000 documents and using a complex model (increased vector size), I was only able to reach about 80% - I attribute this to noise in the training data.

# %%
import collections
from winsound import Beep

# Set the number of top similar documents to consider
SAMPLE_SIZE = N // 10
TOP_N = 10

ranks = []
second_ranks = []
for idx, tagged_doc in enumerate(train_data):

    # Estimate percentage using first (random) `SAMPLE_SIZE` documents
    if idx + 1 == SAMPLE_SIZE:
        break

    # keep track
    if not (idx + 1) % (SAMPLE_SIZE // 10):
        print(f"{(idx+1)/(SAMPLE_SIZE):.0%}... ", end="")

    # Calculate similarities only for the TOP_N similar documents for the current inferred vector
    inferred_vec = model.infer_vector(tagged_doc.words)
    sims = model.dv.most_similar([inferred_vec], topn=TOP_N)

    # Find the rank of the tag in the top N
    try:
        rank = [docid for docid, sim in sims].index(tagged_doc.tags[0])
    except ValueError:
        # Handle the case where the tag is not found in sims
        rank = -1  # Or any other value that indicates "not found"
    ranks.append(rank)

    second_ranks.append(sims[1])
print(" Done.")

counter = collections.Counter(ranks)
print("counter: ", counter)

# Done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# # 4 Model Evaluation
#
# Doc2Vec is an unsupervised model, so finding a metric to evaluate it is not a straightforward task, given the fact that we also do not have any labeled data.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %% [markdown]
# ## 4.1 Inferring Vectors for Test Data

# %%
from ppa.display import Plotter, display_dim_reduction

N_samples = 1000

# Infer document vectors for the test data
print("Inferring vectors for test documents... ", end="")
document_vectors = [
    model.infer_vector(doc.words) for idx, doc in enumerate(test_data) if idx < N_samples
]
print("Done.")

# Convert document vectors to a numpy array
document_vectors_array = np.array(document_vectors)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# ## 4.2 Visualizing the Inferred Documents

# %% [markdown]
# PCA

# %%
from sklearn.decomposition import PCA

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(document_vectors_array)

with Plotter(figsize=(10, 8)) as ax:
    display_dim_reduction(pca_result, "PCA", parent_ax=ax)

    # TODO: create a wrapper function for 'display_dim_reduction' which includes the annotation with 'test_data' argument
    # Annotate the points with document labels
    for idx, tagged_doc in enumerate(test_data):
        if idx == N_samples:
            break

        # Annotate only a subset of points for clarity (adjust as needed)
        if idx % 10 == 0:  # Annotate every 10th point
            ax.annotate(tagged_doc.tags[0], (pca_result[idx, 0], pca_result[idx, 1]), fontsize=8)

# %% [markdown]
# Let's try t-SNE as well

# %%
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=15,
    learning_rate=200,
    n_iter=1000,
    n_iter_without_progress=500,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(document_vectors_array)
with Plotter(figsize=(10, 8)) as ax:
    display_dim_reduction(tsne_result, "t-SNE", parent_ax=ax)

    # TODO: create a wrapper function for 'display_dim_reduction' which includes the annotation with 'test_data' argument
    # Annotate the points with document labels
    for idx, tagged_doc in enumerate(test_data):
        if idx == N_samples:
            break

        # Annotate only a subset of points for clarity (adjust as needed)
        if idx % 10 == 0:  # Annotate every 10th point
            ax.annotate(tagged_doc.tags[0], (tsne_result[idx, 0], tsne_result[idx, 1]), fontsize=8)

# %% [markdown]
# We need to get some clue as to what the above means. Let's try gathering several "good" and "bad" privacy policies, and see where they stand in the PCA picture.

# %% [markdown]
# # TODO: this will only be relevant once a metric is devised
# Attempting to implement hyperparameter search

# %%
# # Sample tagged documents training data
# print("Preparing training data... ", end="")
# N = 10_000
# train_data = cp.generate_samples(N)
# print(f"Training data ready.")

# %%
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.metrics import make_scorer
# from functools import partial
# from sklearn.base import BaseEstimator
# import numpy as np
# import collections
# from gensim.models import Doc2Vec
# from scipy.stats import randint, uniform

# # Create a larger parameter grid with more combinations
# param_dist = {
#     "vector_size": randint(50, 401),  # Random integer between 50 and 200
#     "epochs": randint(10, 41),  # Random integer between 10 and 40
#     "dm": [0, 1],  # Distributed Memory (PV-DM) vs. Distributed Bag of Words (PV-DBOW)
#     "window": randint(3, 11),  # Random integer between 3 and 10 for the window size
#     "min_count": randint(1, 11),  # Random integer between 1 and 10 for minimum word count
#     "sample": uniform(1e-7, 1e-3),  # Random float between 0.0001 and 0.001 for downsampling
#     "hs": [0, 1],
# }


# class Doc2VecEstimator(BaseEstimator):
#     """Doc."""

#     def __init__(self, vector_size, epochs, dm, window, min_count, sample, hs):
#         self.vector_size = vector_size
#         self.epochs = epochs
#         self.dm = dm
#         self.window = window
#         self.min_count = min_count
#         self.sample = sample
#         self.hs = hs

#     def fit(self, X, y=None):
#         model = Doc2Vec(
#             vector_size=self.vector_size,
#             epochs=self.epochs,
#             dm=self.dm,
#             window=self.window,
#             min_count=self.min_count,
#             sample=self.sample,
#             hs=self.hs,
#         )
#         model.build_vocab(train_data)
#         model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
#         self.model = model
#         return self

#     def predict(self, X):
#         # This is a dummy predict method since it's not relevant for Doc2Vec models
#         return None


# # Define a scoring function for HalvingRandomSearchCV to maximize documents at rank 0
# def custom_scorer(estimator, X, y):
#     train_data = X  # Assuming X contains the train_data
#     fraction_size = N // 10

#     ranks = []
#     for idx, tagged_doc in enumerate(train_data):
#         # Estimate percentage using first (random) `SAMPLE_SIZE` documents
#         if idx + 1 == fraction_size:
#             break

#         inferred_vec = estimator.model.infer_vector(tagged_doc.words)
#         sims = estimator.model.dv.most_similar([inferred_vec], topn=TOP_N)
#         try:
#             rank = [docid for docid, sim in sims].index(tagged_doc.tags[0])
#         except ValueError:
#             rank = -1
#         ranks.append(rank)
#     counter = collections.Counter(ranks)
#     return counter[0]  # Maximize the number of documents at rank 0


# # Create a custom scorer function with a fixed model and training data
# scorer = partial(custom_scorer, train_data=train_data)

# # Create HalvingRandomSearchCV object with the custom estimator
# halving_random_search = HalvingRandomSearchCV(
#     estimator=Doc2VecEstimator(
#         vector_size=100, epochs=10, dm=1, window=5, min_count=1, sample=1e-5, hs=0
#     ),
#     param_distributions=param_dist,
#     n_candidates="exhaust",
#     verbose=1,
#     scoring=make_scorer(scorer, greater_is_better=False),
#     random_state=42,
#     cv=2,
# )

# # Fit the hyperparameter search on your training data
# tic = time.perf_counter()
# halving_random_search.fit(np.zeros(N), np.zeros(N))
# print(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# # Get the best hyperparameters and model
# best_params = halving_random_search.best_params_
# best_model = halving_random_search.best_estimator_

# # Print the best hyperparameters
# print("Best Hyperparameters:", best_params)
