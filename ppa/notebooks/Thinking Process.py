# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# ### TODO: Implement cross-validation training, once a metric is devised
# ### TODO: Try [UMAP](https://github.com/lmcinnes/umap) visualization, for speed if anything else

# %%
# # %reload_ext autoreload
# # %autoreload 2

from winsound import Beep
from ppa.utils import config_logging
from ppa.display import Plotter
import numpy as np

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
import random

# TESTEST - use only N paths!
N_PATHS = 100_000_000
print(f"\nWARNING! LIMITING TO {N_PATHS:,} PATHS!")

# get all privacy policy markdown file paths in a (random) list
print("Loading all privacy policy paths to memory... ", end="")
all_policy_paths = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name != "README.md"]
SEED = 42
rng = np.random.default_rng(seed=SEED)
rng.shuffle(all_policy_paths)

# get `N_PATHS` random policy paths
policy_paths = [fpath for idx, fpath in enumerate(all_policy_paths) if idx < N_PATHS]
print(f"Loaded {len(policy_paths):,}/{len(all_policy_paths):,} privacy policy files.")

# %% [markdown]
# # Processing TODOs:
# ## TODO: for individual policies - try seeing if the URL appears as is or as consecutive separate words (use word-ninja) in the policy, more than once. if so, convert all appearances to \<COMPANY> tokens.
# ## TODO: find strange tokens and take care of them. e.g:
# ### 1) I noticed some tokens containing underscores '_' - these should not exist!
# ### 2) I noticed some \<URL> tokens concatenated to some other word/token.
# ### 3) I noticed \<URL>_\<URL> structures
# ## Start by establishing a way to search for patterns and get at least the first policy containing them so that their removal could be tested.
# ## TODO: implement a second processing step after Dictionary object is created, which should possibly contain:
# ### 1) try to compose a list of privacy-domain words/n-grams, perhaps by inspecting the dictionary or the corpus itself (for n-grams), and incorporate 'Privacy Term Highlighting' (see ChatGPT conversation) for converting them into special tokens (such as by adding square brackes around these expressions). Consider using the full data using the ToS;DR API for extracting important features existing in general in PPs so that these could be used for feature engineering (selecting best tokens) for all PPs. This could better embed the privacy-oriented properties of PPs (and not themes)
# #### 1.1) examples for privacy terms - third-party

# %%
# TEST

# from copy import deepcopy

# # get and check out token document counts (in how many documents each token appears)
# id_doc_count_dict = dict(sorted({id_: count for id_, count in dct.dfs.items()}.items(), key=lambda x: x[1]))
# token_doc_count_dict = {dct[id_]: count for id_, count in id_doc_count_dict.items()}

# # consider tokens appearing in N or less documents
# min_docs = 3
# rare_doc_count_ids = [id_ for id_, count in id_doc_count_dict.items() if count <= min_docs]
# rare_doc_count_tokens = [cp.dct[id_] for id_ in rare_doc_count_ids]

# # get and check out collection counts for the rare doc-count tokens
# comp_names_potential_ids = [id_ for id_ in rare_doc_count_ids if cp.dct.cfs[id_] >= id_doc_count_dict[id_] * 2]
# comp_names_potential_tokens = [cp.dct[id_] for id_ in comp_names_potential_ids]

# %%
# print(
#     f"Dictionary was created by processing {cp.dct.num_pos:,} tokens from a corpus of {cp.dct.num_docs:,} documents."
# )
# print(f"It contains {len(cp.dct):,} unique tokens.")
# print(f"Each document, on average, contains {cp.dct.num_nnz // cp.dct.num_docs:,} unique tokens.")

# %%
# print(len(comp_names_potential_tokens))
# comp_names_potential_tokens

# %%
# print(len(rare_doc_count_tokens))
# rare_doc_count_tokens

# %%
# len(rare_doc_count_tokens)

# %% [markdown]
# Create a fresh CorpusProcessor instance, build a `gensim.corpora import Dictionary` and process the entire corpus, all while streaming to/from disk.

# %%
from ppa.ppa import CorpusProcessor

# SHOULD_REPROCESS = True
SHOULD_REPROCESS = False

MODEL_DIR_PATH = Path.cwd().parent / "models"

# create a document processor with the paths
cp = CorpusProcessor(
    policy_paths,
    MODEL_DIR_PATH,
    seed=SEED,
)

# build and save dictionary from all documents, process all documents and serialize (compressed) the TaggedDocument objects to disk
cp.process(
    force=SHOULD_REPROCESS,
    min_tokens=40,
    max_tokens=5000,
    lemmatize=True,
    should_filter_stopwords=True,
)

Beep(1000, 500)

# %%
# TEST

IDX = 3

sg = cp.generate_samples()
print(sg[IDX].tags, len(sg[IDX].words))
" ".join(sg[IDX].words)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens). It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset:

# %%
N = cp.total_samples // 10
pp_lengths = np.array(
    [len(tagged_doc.words) for tagged_doc in cp.generate_samples(n_samples=N, shuffled_idx=True)]
)

print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

with Plotter() as ax:
    ax.hist(pp_lengths, int(np.sqrt(pp_lengths.size)))

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

Beep(1000, 500)

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
# TODO: change this so that it finds a pattern, not a token

# The token you want to search for
target_token = "url"

# Iterate through each document and check if the token is in the document
for tagged_doc in cp.generate_samples(n_samples=N):
    if target_token in tagged_doc.words:
        print(f"{tagged_doc.tags[0]}:\n")
        print(" ".join(tagged_doc.words))
        break

# %% [markdown]
# # 3. Aqcuiring and incorporating labels
# ## 3.1 ETL for Policy Ratings
# Getting all tags, for searching the ToS;DR database

# %%
from itertools import chain

# N = 10
N = np.inf

print(f"Getting URLs/tags... ", end="")
# tags = [tagged_doc.tags[0] for idx, tagged_doc in enumerate(chain(train_data, test_data)) if idx < N]
tags = [fpath.stem for idx, fpath in enumerate(policy_paths) if idx < N]
print(f"{len(tags):,} tags obtained.")

# %% [markdown]
# ETL

# %%
import asyncio
from ppa.ppa import ToSDRDataLoader
import pandas as pd

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
ratings_df = pd.DataFrame()  # TESTESTEST - to shut mypy up
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
# ## 3.2 Exploration
# ### 3.2.1 Checking for duplicates in data according to rating IDs
#
# Let's try to check what policies with duplicate IDs look like - are they really the same? Note that ToS;DR rates terms-of-service together with privacy policies - I don't really know what same IDs mean!
#
# To do this, let's take the ID with, say, 5 entries in `ratings_df`:

# %%
# ratings_df['id'].value_counts()[ratings_df['id'].value_counts() == 5]

# %% [markdown]
# choosing one of them

# %%
# result = ratings_df[ratings_df['id'] == 9465]
# result

# %% [markdown]
# and find all matching tags (URLs) in the corpus (both training and testing):

# %%
# corpus, _ = cp.generate_train_test_sets(test_frac=0)

# training_samples = [tagged_doc for tagged_doc in corpus if tagged_doc.tags[0] in result["tag"].tolist()]

# training_samples

# %% [markdown]
# It appears that at least for one ID, the privacy policies are different. For now, we will disregard the IDs.

# %% [markdown]
# ### 3.2.2 Checking for Bias in Labeled Data

# %%
# with Plotter() as ax:
#     ax.hist(sorted(ratings_df["rating"]))

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


ratings_df["rating"] = ratings_df["rating"].apply(relabel_rating)

with Plotter() as ax:
    ax.hist(sorted(ratings_df["rating"]))

# %% [markdown]
# Perhaps this classification could work as anomaly detection ('good' policies being the anomaly)?

# %% [markdown]
# Finally, let's update the corpus with labeled data, and save it separately:

# %%
# SHOULD_FORCE_LABELING = True
SHOULD_FORCE_LABELING = False

url_rating_dict = ratings_df.set_index("tag")["rating"].to_dict()
cp.add_label_tags(url_rating_dict, force=SHOULD_FORCE_LABELING)

# %% [markdown]
# # 4. Modeling
#
# Next, we want to transform our documents into some vector space. There are many techniques which could be used, but a well established one which captures in-document token relationships (important for semantic context) is Doc2Vec. Training a Doc2Vec model over our data will provide a vector embedding of each document in our corpus. This would facillitate several pathways:
# 1) Enabling cluster analysis and visualization
# 2) Similarity comparison between PPs
# 3) Inferring vector embedding for non-corpus policies
# 4) Ultimately, this will enable using standard classification methods (tabular data)

# %% [markdown]
# Split to train/test sets in a stratified fashion, i.e. keep the same label ratio (in this case the percentages of "good" and "bad" policies) in the data.

# %%
N = cp.total_samples
TEST_FRAC = 0.2

train_set, test_set = cp.generate_train_test_sets(
    #     n_samples=N, test_frac=TEST_FRAC, labeled=False, shuffled_idx=True
    n_samples=N,
    test_frac=TEST_FRAC,
    labeled=True,
    shuffled_idx=True,
)

# TEST - check percentages in train/test splits
from collections import Counter

print(Counter([doc.tags[1] if len(doc.tags) > 1 else "unlabeled" for doc in train_set]))
print(Counter([doc.tags[1] if len(doc.tags) > 1 else "unlabeled" for doc in test_set]))

Beep(1000, 500)

# %% [markdown]
# Train the Doc2Vec model (semi-supervised):

# %%
import time
from gensim.models.doc2vec import Doc2Vec

# define save/load path
MODEL_PATH = MODEL_DIR_PATH / "privacy_policy_doc2vec.model"

# SHOULD_RETRAIN = True
SHOULD_RETRAIN = False

if not SHOULD_RETRAIN:
    # load the last trained model
    semi_supervised_model: Doc2Vec = Doc2Vec.load(str(MODEL_PATH))

    # Done
    print("Semi-supervised model loaded")
    # TODO: print some model details

else:
    # Initialize and train the Doc2Vec model
    semi_supervised_model_kwargs = {
        #         'dm': 1, 'epochs': 9, 'hs': 1, 'vector_size': 443, 'window': 6
        "vector_size": 400,
        "window": 6,
        "hs": 1,
        "negative": 0,
        "epochs": 9,
        #         "workers": mp.cpu_count(),
    }
    semi_supervised_model = Doc2Vec(**semi_supervised_model_kwargs)

    # Build vocabulary
    print("Building vocabulary... ", end="")
    semi_supervised_model.build_vocab(train_set)
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

    # Train the model
    print(f"Training semi-supervised model... ", end="")
    tic = time.perf_counter()
    semi_supervised_model.train(
        train_set,
        total_examples=semi_supervised_model.corpus_count,
        epochs=semi_supervised_model.epochs,
    )

    # Save the trained model for future use
    print(f"Saving to '{MODEL_PATH}'... ", end="")
    semi_supervised_model.save(str(MODEL_PATH))
    print("Done!")

    # Done!
    print(f"Training timing: {(time.perf_counter() - tic)/60:.1f} mins")
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# # 5 Doc2Vec Model Evaluation
#
# Doc2Vec, using secondary tags, is a weakly semi-supervised model, so finding a metric to evaluate it is not a straightforward task.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %% [markdown]
# ## 5.1 Sanity check
# As a first test of the model, a reasonable sanity check (adapted from that suggested by [Radim Řehůřek](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py) himself) would be to see if most vectors inferred for the policies the model was trained upon are most similar to the corresponding document vectors of the model itself.

# %%
from collections import Counter

if SHOULD_RETRAIN:
    model = semi_supervised_model

    # Set the number of top similar documents to consider
    SAMPLE_SIZE = max(N // 100, 100)
    TOP_N = 10

    ranks = []
    second_ranks = []
    for idx, tagged_doc in enumerate(train_set):

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

    counter = Counter(ranks)
    print("counter: ", counter)

    # Done
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

else:
    print("Skipping.")

# %% [markdown]
# ## 5.2 Visualizing the results using dimensionallity reduction to 2D
# We begin by inferring vectors for all labeled samples in the test set

# %%
# define the model
model = semi_supervised_model

# Infer document vectors for the test data
print("Inferring vectors for labeled test policies... ", end="")
labeled_test_vectors, labeled_test_tags = zip(
    *[
        (model.infer_vector(td.words), (td.tags if len(td.tags) > 1 else td.tags + [label]))
        for td, label in zip(test_set, test_set.labels)
        if label != "unlabeled"
    ]
)
labeled_test_vectors = np.array(labeled_test_vectors)
print("Done.")

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# ### 5.2.1 PCA

# %%
from sklearn.decomposition import PCA
from ppa.display import display_dim_reduction

test_labels = [tags[1] for tags in labeled_test_tags]

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(labeled_test_vectors)

# annots = [
#     tagged_doc.tags[0]
#     for tagged_doc in test_set
#     if len(tagged_doc.tags) > 1 and tagged_doc.tags[1] == "good"
# ]
display_dim_reduction(pca_result, "PCA", labels=test_labels, figsize=(10, 8))

# %% [markdown]
# ### 5.2.2 t-SNE

# %%
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=1,
    learning_rate=200,
    n_iter=1000,
    n_iter_without_progress=500,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(labeled_test_vectors)

# annots = [
#     tagged_doc.tags[0]
#     for tagged_doc in test_set
#     if len(tagged_doc.tags) > 1 and tagged_doc.tags[1] == "good"
# ]
display_dim_reduction(tsne_result, "t-SNE", labels=test_labels, figsize=(10, 8))

# %% [markdown]
# I cannot see any pattern separating "good" policies from "bad" ones. This doesn't mean the model isn't sensitive to the labels, only that the 2D visualization doesn't appear to capture it.

# %% [markdown]
# ## 5.3 Devising a metric
# Perhaps the similarity between like-labled policies is lost in the dimensionality reduction. Let's try measuring the cosine similarity between the vectors directly.

# %%
from sklearn.metrics.pairwise import cosine_similarity

print("Calculating mean good/bad training-set (model) vectors... ", end="")
# train

labeled_train_vectors, labeled_train_tags = zip(
    *[
        (model.dv[td.tags[0]], (td.tags if len(td.tags) > 1 else td.tags + [label]))
        for td, label in zip(train_set, train_set.labels)
        if label != "unlabeled"
    ]
)

# train_vectors, train_tags = zip(*[(model.dv[td.tags[0]], td.tags) for td in train_set])
mean_good_train_vector = np.array(
    [vec for vec, tags in zip(labeled_train_vectors, labeled_train_tags) if tags[1] == "good"]
).mean(axis=0)
mean_bad_train_vector = np.array(
    [vec for vec, tags in zip(labeled_train_vectors, labeled_train_tags) if tags[1] == "bad"]
).mean(axis=0)
print("Done.")

print("Calculating similarites... ", end="")
good_similarities = {}
for test_tag, test_policy_vector in zip(labeled_test_tags, labeled_test_vectors):
    good_sim = cosine_similarity([test_policy_vector], [mean_good_train_vector])[0][0]
    bad_sim = cosine_similarity([test_policy_vector], [mean_bad_train_vector])[0][0]
    good_similarities[test_tag[0]] = (
        (good_sim - bad_sim),
        test_tag[1],
    )
print("Done.")

# Checkout the URLs and scores
# print(dict(sorted(good_similarities.items(), key=lambda item: item[1][0], reverse=True)))

# Collect predicted scores and true labels for "good" and "bad" policies
print("Calculating test sample scores... ", end="")
good_true_labels, good_similarity_scores = zip(
    *[(true_label == "good", score) for score, true_label in good_similarities.values()]
)
print("Done.")

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# We can check out the distribution of my custom similarity scores separately for "good" and "bad" policies:

# %%
# Separate positive and negative instances
positive_scores = [
    score for score, label in zip(good_similarity_scores, good_true_labels) if label == 1
]
negative_scores = [
    score for score, label in zip(good_similarity_scores, good_true_labels) if label == 0
]

with Plotter(
    figsize=(10, 6),
    xlabel="Similarity Scores",
    ylabel="Frequency",
    suptitle="Distribution of Similarity Scores",
) as ax:

    # Plot the distribution of similarity scores
    ax.hist(
        positive_scores,
        bins=int(np.sqrt(len(positive_scores))),
        density=True,
        alpha=0.7,
        color="blue",
        label="Positive Instances",
    )
    ax.hist(
        negative_scores,
        bins=int(np.sqrt(len(negative_scores))),
        density=True,
        alpha=0.7,
        color="orange",
        label="Negative Instances",
    )

    ax.legend()
    ax.grid(True)

# %% [markdown]
# The significant overlap apparent in the above figure could mean that the model is struggling to separate good from bad policies, especially in the range (-0.15, 0.05)

# %% [markdown]
# ### 5.3.1 ROC AUC

# %%
from sklearn.metrics import roc_auc_score, roc_curve

# Display ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(good_true_labels, good_similarity_scores)
roc_auc = roc_auc_score(good_true_labels, good_similarity_scores)
with Plotter(
    suptitle="Receiver Operating Characteristic (ROC) Curve",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
) as ax:
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.legend()


# %% [markdown]
# ### 5.3.2 AUC-PR

# %%
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(good_true_labels, good_similarity_scores)
auc_pr = auc(recall, precision)

with Plotter(
    xlabel="Recall",
    ylabel="Precision",
) as ax:
    ax.plot(recall, precision, label=f"Precision-Recall Curve (AUC-PR = {auc_pr:.2f})")
    ax.legend()

# %%
raise RuntimeError("STOP HERE!")

# %% [markdown]
# ## 5.4 Hyperparameter Search

# %% [markdown]
# Define a custom Doc2Vec estimator which would work with my SampleGenerator class

# %%
from sklearn.base import BaseEstimator, TransformerMixin
import logging


class Doc2VecEstimator(BaseEstimator, TransformerMixin):
    """Doc."""

    def __init__(self, vector_size, epochs, dm, window, min_count, sample, hs, negative):
        self.vector_size = vector_size
        self.epochs = epochs
        self.dm = dm
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.hs = hs
        self.negative = negative
        self.model = None
        self.mean_good_train_vector = None
        self.mean_bad_train_vector = None

    def fit(self, X, y=None):
        """Doc."""

        logging.info(f"[Doc2VecEstimator.fit] len(X): {len(X)}")
        label_counter = Counter([td.tags[1] if len(td.tags) > 1 else "unlabeled" for td in X])
        logging.info(f"[Doc2VecEstimator.fit] label_counter: {label_counter}")

        model = Doc2Vec(
            vector_size=self.vector_size,
            epochs=self.epochs,
            dm=self.dm,
            window=self.window,
            min_count=self.min_count,
            sample=self.sample,
            hs=self.hs,
            negative=self.negative,
        )
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

        #         # Compute mean vectors from the training set
        #         train_vectors, train_tags = zip(*[(self.model.dv[td.tags[0]], td.tags) for td in X])
        #         #         logging.info(f"{train_tags}")
        #         self.mean_good_train_vector = np.array(
        #             [
        #                 vec
        #                 for vec, tags in zip(train_vectors, train_tags)
        #                 if len(tags) > 1 and tags[1] == "good"
        #             ]
        #         ).mean(axis=0)
        #         self.mean_bad_train_vector = np.array(
        #             [
        #                 vec
        #                 for vec, tags in zip(train_vectors, train_tags)
        #                 if len(tags) > 1 and tags[1] == "bad"
        #             ]
        #         ).mean(axis=0)

        return self

    def transform(self, X, y=None):
        """Doc."""

        labeled_vectors = np.array([model.infer_vector(td.words) for td in X])
        logging.info(f"[Doc2VecEstimator.transform] labeled_vectors.shape: {labeled_vectors.shape}")
        return labeled_vectors

    def fit_transform(self, X, y=None):
        """Doc."""

        self.fit(X, y)
        return self.transform(X, y)

    def score(self, X, y=None):
        """Compute the AUC-PR score on the test set."""

        logging.info(f"[Doc2VecEstimator.score] len(X): {len(X)}")

        # Use self.mean_good_train_vector and self.mean_bad_train_vector to compute scores
        labeled_test_vectors, labeled_test_tags = zip(
            *[(self.model.infer_vector(td.words), td.tags) for td in X if len(td.tags) > 1]
        )
        labeled_test_vectors = np.array(labeled_test_vectors)

        # Calculating similarities
        good_similarities = {}
        for test_tag, test_policy_vector in zip(labeled_test_tags, labeled_test_vectors):
            good_sim = cosine_similarity([test_policy_vector], [mean_good_train_vector])[0][0]
            bad_sim = cosine_similarity([test_policy_vector], [mean_bad_train_vector])[0][0]
            good_similarities[test_tag[0]] = (
                (good_sim - bad_sim),
                test_tag[1],
            )

        # Calculating test sample scores
        y_true, y_scores = zip(
            *[(true_label == "good", score) for score, true_label in good_similarities.values()]
        )

        # Compute AUC-PR
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall, precision)
        return auc_pr


# %% [markdown]
# Perform search to find best set of hyperparametes

# %%
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
# from scipy.stats import randint, uniform
# # import psutil

# # Create a larger parameter grid with more combinations
# param_dist = {
#     "vector_size": randint(150, 601),  # Random integer between 50 and 400
#     "epochs": randint(5, 21),  # Random integer between 10 and 40
#     "dm": [0, 1],  # Distributed Memory (PV-DM) vs. Distributed Bag of Words (PV-DBOW)
#     "window": randint(4, 13),
# #     "min_count": randint(0, 6),
# #     "sample": uniform(1e-7, 1e-5),
#     "hs": [0, 1],
# }

# # Create HalvingRandomSearchCV object with the custom estimator
# halving_random_search = HalvingRandomSearchCV(
#     estimator=Doc2VecEstimator(
#         vector_size=100, epochs=10, dm=1, window=5, min_count=1, sample=1e-5, hs=0, negative=0,
#     ),
#     param_distributions=param_dist,
#     n_candidates="exhaust",
#     verbose=1,
#     random_state=cp.seed,
#     cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cp.seed),
#     min_resources=15000,
# #     n_jobs=psutil.cpu_count(logical=False) - 1,
# )

# # Fit the hyperparameter search on your training data
# tic = time.perf_counter()
# # print("Loading training set to memory... ", end="")
# # hps_train_set = deepcopy(train_set)
# # hps_train_set.load_to_memory()
# # print("Done.")
# print("Starting search...")
# # halving_random_search.fit(hps_train_set, hps_train_set.labels)
# halving_random_search.fit(train_set, train_set.labels)
# print(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# # Get the best hyperparameters and model
# best_params = halving_random_search.best_params_
# best_model = halving_random_search.best_estimator_

# # Print the best hyperparameters
# print("Best Hyperparameters:", best_params)

# # Beep when done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# ## 5.5 Final Evaluation of Doc2Vec Model Using Best Hyperparameters

# %%
# # Get the AUC-PR score of the best model on the test set
# best_model_score = best_model.score(test_set)
# print("Best Model Score:", best_model_score)

# %% [markdown]
# # 6. Attaching a Classifier
# It seems that the Doc2Vec model by itself is not doing a good job separating good privacy policies from bad ones. I can try using the vector embeddings as features for a binary classifier. Since I already have an estimator class for Doc2Vec, it should have been relatively easy to create a pipeline and attach more estimators. Unfortunately, since sklearn.pipeline.Pipeline doens't transform y (targets/labels) during fitting, I would have to implement a custom pipeline.

# %%
# # TEST

# # Create the pipeline
# pipeline = Pipeline([
#     ('vect', Doc2VecEstimator(vector_size=100, epochs=10, dm=1, window=5, min_count=1, sample=1e-5, hs=1, negative=0)),
# #     ('sel', SelectKBest(score_func=f_classif, k=50)),
#     ('clf', OneClassSVM(gamma="scale"))
# ])

# # Retrieve the steps in the pipeline
# steps = pipeline.named_steps

# # Inspect transformed data between pipeline steps
# data_step1 = steps['vect'].fit_transform(train_set[:10_000], train_set.labels[:10_000])  # Output from Doc2VecEstimator
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
# # data_step2 = steps['sel'].fit_transform(*data_step1)  # Output from SelectFromModel
# # Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
# # Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
# data_step3 = steps['clf'].fit_predict(data_step1)  # Output from OneClassSVM
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# # Log the shapes or any information you need to inspect
# logging.info(f"Data after step 1 shape: {data_step1.shape}")
# logging.info(f"Data after step 2 shape: {data_step2.shape}")
# logging.info(f"Data after step 3 shape: {data_step3.shape}")

# %%
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, OneClassSVM
from scipy.stats import randint, uniform
from sklearn.feature_selection import SelectKBest, f_classif


def custom_auc_pr_score(estimator, X, y):

    #     logging.info(f"[custom_auc_pr_score] estimator: {estimator}")

    #     logging.info(f"[custom_auc_pr_score] type(X): {type(X)}")
    #     logging.info(f"[custom_auc_pr_score] len(X): {len(X)}")
    #     logging.info(f"[custom_auc_pr_score] X[0]: {X[0]}")

    #     logging.info(f"[custom_auc_pr_score] type(y): {type(y)}")
    #     logging.info(f"[custom_auc_pr_score] len(y): {len(y)}")
    #     logging.info(f"[custom_auc_pr_score] y[0]: {y[0]}")

    # Calculate scores
    try:
        y_scores = estimator.score(X)
    except AttributeError:
        y_scores = estimator.decision_function(X)
        logging.info(f"[custom_auc_pr_score] y_scores[:10]: {y_scores[:10]}")

    # ignore unlabeled samples
    y_scores = np.array([score for idx, score in enumerate(y_scores) if y[idx] != "unlabeled"])
    logging.info(f"[custom_auc_pr_score] y_scores[:10]: {y_scores[:10]}")
    label_to_bool_dict = dict(good=1, bad=0)
    y = np.array([label_to_bool_dict[label] for label in y if label in label_to_bool_dict])

    # Compute the precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_scores)

    # Calculate the AUC-PR
    auc_pr = auc(recall, precision)
    return auc_pr


# Create the pipeline
pipeline = Pipeline(
    [
        (
            "vect",
            Doc2VecEstimator(
                vector_size=100,
                epochs=10,
                dm=1,
                window=5,
                min_count=1,
                sample=1e-5,
                hs=1,
                negative=0,
            ),
        ),
        #     ('sel', SelectKBest(score_func=f_classif, k=50)),
        ("clf", OneClassSVM(gamma="scale")),
    ]
)

# # Fit the pipeline on your training data
# print("Fitting pipeline...")
# pipeline.fit(train_set[:10_000], train_set.labels[:10_000])

# Create a larger parameter grid with more combinations
param_dist = {
    "vect__vector_size": randint(150, 601),  # Random integer between 50 and 400
    "vect__epochs": randint(5, 21),  # Random integer between 10 and 40
    "vect__dm": [0, 1],  # Distributed Memory (PV-DM) vs. Distributed Bag of Words (PV-DBOW)
    "vect__window": randint(4, 13),
    "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "clf__degree": randint(2, 6),  # Assuming higher-degree polynomials might be beneficial
    "clf__gamma": [
        "scale",
        "auto",
        uniform(0.1, 1.0),
    ],  # Consider both predefined and float gamma values
    "clf__coef0": uniform(-1, 1),  # Coef0 for 'poly' and 'sigmoid'
    "clf__tol": uniform(1e-5, 1e-2),  # Varying tolerance values
    "clf__nu": uniform(0.05, 0.95),  # Considering a range for nu
    "clf__shrinking": [True, False],  # To explore the effect of shrinking heuristic
}

# Update the hyperparameter search to use the pipeline
halving_random_search = HalvingRandomSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_candidates="exhaust",
    verbose=1,
    random_state=cp.seed,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cp.seed),
    scoring=custom_auc_pr_score,
    min_resources=15000,
)

# Fit the hyperparameter search on your training data
tic = time.perf_counter()
print("Starting search...")
halving_random_search.fit(train_set, train_set.labels)
# halving_random_search.fit(train_set, np.zeros((len(train_set.labels),)))
print(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# Get the best hyperparameters and model
best_params = halving_random_search.best_params_
best_model = halving_random_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# %% [markdown]
# # Label test policies according to nearest labeld policy from training coprus - check this out if classification using the available true labels is insufficient

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
