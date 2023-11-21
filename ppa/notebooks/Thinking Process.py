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
# ### TODO: Try feature selection?
# ### TODO: Attempt to separate the estimator - the epochs are the issue
# ### TODO: consider "pseudo-labelling" - predicting labels from model then re-training, testing each iteration.
# ### TODO: Consider/try/test using vec_model.dv.get_mean_vector("good") instead of calculating the mean in vec_score method
# ### TODO: try removing PPs of extreme length (according to histogram) - see how it effects score - it definitely affects training time! Compare say cutting at 5000 token vs. 2000 tokens.
# ### TODO: Figure out why there seems to be a few more vectors in the model.dv then there are training samples???
# ### TODO: Try [UMAP](https://github.com/lmcinnes/umap) visualization, for speed if anything else
#
# ### TODO: try these suggestions:
# * Topic Modeling: Consider using topic modeling techniques (e.g., Latent Dirichlet Allocation - LDA) to identify underlying topics within the documents. Visualize the topics and their prevalence in the dataset.
# * Named Entity Recognition (NER): If applicable, perform NER to extract entities like names, organizations, locations, and dates from the text. Explore the frequency and distribution of entities in the documents.

# %%
# # %reload_ext autoreload
# # %autoreload 2

from winsound import Beep
from ppa.utils import config_logging
from ppa.display import Plotter
import numpy as np
from collections import Counter
import psutil

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
    n_samples=N,
    test_frac=TEST_FRAC,
    labeled=True,
    shuffled_idx=True,
)

print(Counter(train_set.labels))
print(Counter(test_set.labels))

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
        "workers": psutil.cpu_count(logical=False) - 1,
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
mean_good_train_vector = model.dv["good"]
mean_bad_train_vector = model.dv["bad"]
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
positive_scores = [score for score, label in zip(good_similarity_scores, good_true_labels) if label]
negative_scores = [
    score for score, label in zip(good_similarity_scores, good_true_labels) if not label
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

# %% [markdown]
# # 6. Attaching a Classifier
# It seems that the Doc2Vec model by itself is not doing a good job separating good privacy policies from bad ones. I can try using the vector embeddings as features for a binary classifier. Since I already have an estimator class for Doc2Vec, it should have been relatively easy to create a pipeline and attach more estimators. Unfortunately, since sklearn.pipeline.Pipeline doens't transform y (targets/labels) during fitting, I would have to implement a custom estimator combining Doc2Vec with an unsupervised classifier.

# %% [markdown]
# Let's create reusable train/test sets (stratified)

# %%
N = 40_000
TEST_FRAC = 0.2

toy_train_set, toy_test_set = cp.generate_train_test_sets(
    n_samples=N,
    test_frac=TEST_FRAC,
    labeled=True,
    shuffled_idx=True,
)

print("Keeping labels in sets... ", end="")
print(Counter(toy_train_set.labels))
print(Counter(toy_test_set.labels))
print("Done.")

Beep(1000, 500)

# %% [markdown]
# Define a custom Doc2Vec + IsolationForest estimator which would work with my SampleGenerator class

# %%
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from scipy.special import expit
import psutil
from typing import List, Set, Tuple
import logging
from gensim.models.doc2vec import TaggedDocument

from ppa.utils import timer


class Doc2VecIsolationForestEstimator(BaseEstimator):
    """Doc."""

    implemented_metrics: Set[str] = {"auprc", "bal_acc"}

    def __init__(
        self,
        # General
        epochs: int,
        onlne_learning: bool,
        metric: str,
        train_score: bool = False,
        random_state: int = None,
        # Doc2Vec kwargs
        vector_size: int = 300,
        dm: int = 1,
        window: int = 5,
        min_count: int = 0,
        sample: float = 0.0,
        hs: int = 1,
        negative: float = 0.0,
        workers: int = psutil.cpu_count(logical=False) - 1,
        # IsolationForest kwargs
        n_estimators: int = 300,
        contamination: str | float = "auto",
        max_samples: int | str = "auto",
        max_features: int | float = 1.0,
        bootstrap: bool = False,
        verbose: int = 0,
        warm_start: bool = True,
        n_jobs: int = psutil.cpu_count(logical=False) - 1,
    ):

        if metric not in self.implemented_metrics:
            raise ValueError(
                f"Unknown metric '{self.metric}'. Must be in {self.implemented_metrics}"
            )

        # Get the local variables as a dictionary
        arguments = locals()
        del arguments["self"]  # Remove 'self' from the dictionary

        # Loop through the arguments and set attributes dynamically
        for key, value in arguments.items():
            setattr(self, key, value)

    @timer(1000)
    def fit(self, X, y):
        """Doc."""

        # Initialize both models
        self.vec_model = Doc2Vec(
            vector_size=self.vector_size,
            dm=self.dm,
            window=self.window,
            min_count=self.min_count,
            sample=self.sample,
            hs=self.hs,
            negative=self.negative,
            seed=self.random_state,
            workers=self.workers,
        )

        self.clf_model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
        )

        # Count and display the amount of samples for each label
        logging.info(f"[Estimator.fit] label_counter: {Counter(y)}")

        # build vocabulary for Doc2Vec
        self.vec_model.build_vocab(X)

        # Training begins
        if self.onlne_learning:
            # get learning rate ranges for each epoch
            alpha_edges = np.geomspace(
                self.vec_model.alpha, self.vec_model.min_alpha, self.epochs + 1
            )
            self.alpha_ranges = np.array([alpha_edges[:-1], alpha_edges[1:]]).T

            # get increasing number of IsolationForest estimators in logarithmic fashion fo each epoch, e.g. [ 500,  575,  643,  707,  766,  820,  871,  917,  960, 1000]
            start_val = self.clf_model.n_estimators // 2
            stop_val = self.clf_model.n_estimators
            self.n_estimators_list = np.hstack(
                (
                    [start_val],
                    np.cumsum(
                        np.flip(
                            np.diff(
                                np.flip(np.geomspace(stop_val, start_val, self.epochs, dtype=int))
                            )
                        )
                    )
                    + start_val,
                )
            )

            # iterative training
            #             self.vec_loss = [] # NOTE Not implemented in Gensim
            for epoch in range(self.epochs):
                # train a Doc2Vec model on the entire, sparsely labeled dataset
                # set the alpha decay range according to the pre-defined ranges
                self.vec_model.alpha, self.vec_model.min_alpha = self.alpha_ranges[epoch]
                logging.info(
                    f"[Estimator.fit] [epoch {epoch}] Training {self.vec_model.lifecycle_events[0]['params']}"
                )
                #                 logging.info(f"[Estimator.fit] [epoch {epoch}] vec_model.alpha: {self.vec_model.alpha:.2e}, vec_model.alpha: {self.vec_model.min_alpha:.2e}")
                self.vec_model.train(
                    X, total_examples=self.vec_model.corpus_count, epochs=1, compute_loss=True
                )
                #                 self.vec_loss.append(self.vec_model.get_latest_training_loss()) # NOTE: Not implemented, returns 0.0
                #                 logging.info(f"[Estimator.fit] [epoch {epoch}] vec_loss: {self.vec_loss[epoch]:.2e}") # NOTE: Not implemented in Gensim, returns 0.0

                # transform the documents into vectors
                logging.info(
                    f"[Estimator.fit] [epoch {epoch}] Using model vectors as X_train_vec..."
                )
                X_train_vec = self.vec_model.dv.vectors

                # Increase the number of trees in IsolationForest according to predetermined list of values
                self.clf_model.set_params(n_estimators=self.n_estimators_list[epoch])
                # Train the IsolationForest model using the all samples
                logging.info(
                    f"[Estimator.fit] Fitting IsolationForest model `{self.clf_model.get_params()}` to {X_train_vec.shape[0]:,} vector samples..."
                )
                self.clf_model.fit(X_train_vec)

                # calculate training score (optional)
                if self.train_score:
                    logging.info(
                        f"[Estimator.fit] [epoch {epoch}] Training score: {self.score(X, y, verbose=False)}"
                    )

        # Training vectorizer first, than classifier
        else:
            # train a Doc2Vec model on the entire, sparsely labeled dataset
            logging.info(f"[Estimator.fit] Training {self.vec_model.lifecycle_events[0]['params']}")
            self.vec_model.train(
                X, total_examples=self.vec_model.corpus_count, epochs=self.epochs, compute_loss=True
            )
            #             self.vec_loss = [self.vec_model.get_latest_training_loss()]
            #             logging.info(f"[Estimator.fit] vec_loss: {self.vec_loss[0]:.2e}")

            # transform the documents into vectors
            logging.info(f"[Estimator.fit] Using model vectors as X_train_vec...")
            X_train_vec = self.vec_model.dv.vectors

            # Train the IsolationForest model using the all samples
            logging.info(
                f"[Estimator.fit] Fitting IsolationForest model `{self.clf_model.get_params()}` to {X_train_vec.shape[0]:,} vector samples..."
            )
            self.clf_model.fit(X_train_vec)

            # calculate training score (optional)
            if self.train_score:
                logging.info(f"[Estimator.fit] Training score: {self.score(X, y, verbose=False)}")

        return self

    #     @timer(1000)
    def predict(self, X=None, X_vec=None, **kwargs):
        """Doc."""

        # transform X if X_vec not supplied
        X_vec = X_vec if X_vec is not None else self.vec_transform(X, **kwargs)

        # return classifier prediction
        logging.info(f"[Estimator.predict] Predicting...")
        return self.clf_model.predict(X_vec)

    #     @timer(1000)
    def score(self, X: List[TaggedDocument], y: List[str], verbose=True):
        """Doc."""

        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
        y_true, labeled_idxs = self.valid_labels(y)

        if verbose:
            logging.info(f"[Estimator.score] Filtering labeled samples...")
        X_test = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]

        # transform X, y
        X_test_vec = self.vec_transform(X_test)

        #         # Calculate _vec_score
        #         vec_score = self._vec_score(X_test_vec, y_true)
        #         if verbose:
        #             logging.info(f"[Estimator.score] vec_score: {vec_score:.2f}")

        # Calculate scores
        if verbose:
            logging.info(f"[Estimator.score] Calculating score...")
        y_scores = self.clf_model.decision_function(X_test_vec)

        #         # Normalize scores using min-max scaling, using labeled + unlabeled samples
        #         scaler = MinMaxScaler((-6, 6))
        #         y_scores_norm = scaler.fit_transform(y_scores.reshape(-1, 1)).flatten()
        #         logging.info(f"[Estimator.score] y_scores_norm[:10]: {y_scores_norm[:10]}")
        #         # Transform normalized scores to probabilities using sigmoid function
        #         y_scores = expit(y_scores_norm)

        # Compute the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # Calculate the AUC-PR
        auc_pr = auc(recall, precision)

        # predict and calculate balanced accuracy
        y_pred = self.predict(X_vec=X_test_vec)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        if verbose:
            # TESTESTEST
            RocCurveDisplay.from_predictions(y_true, y_scores, name="ROC-AUC")
            PrecisionRecallDisplay.from_predictions(y_true, y_scores, name="AUPRC")
            print(classification_report(y_true, y_pred))
            # calculate individual accuracies
            good_idxs = y_true == -1
            bad_idxs = y_true == 1
            good_accuracy = sum(y_pred[good_idxs] == y_true[good_idxs]) / y_true[good_idxs].size
            bad_accuracy = sum(y_pred[bad_idxs] == y_true[bad_idxs]) / y_true[bad_idxs].size

            logging.info(f"[Estimator.score] AUPRC: {auc_pr:.2f}")
            logging.info(f"[Estimator.score] ACC: Good={good_accuracy:.2f}, Bad={bad_accuracy:.2f}")
            logging.info(f"[Estimator.score] Balanced ACC: {bal_acc}")
            # /TESTESTEST

        if verbose:
            logging.info(f"[Estimator.score] Returning '{self.metric}' score.")

        if self.metric == "auprc":
            return auc_pr
        elif self.metric == "bal_acc":
            return bal_acc

    def valid_labels(self, y) -> Tuple[np.array, np.array]:
        """Doc."""

        conv_dict = dict(unlabeled=np.nan, good=-1, bad=1)
        y_arr = np.array([conv_dict[label] for label in y])
        labeled_idxs = ~np.isnan(y_arr)
        return y_arr[labeled_idxs], labeled_idxs

    def vec_transform(self, X: List[TaggedDocument], normalized=False):
        """Doc."""

        inference_params = {}
        if self.onlne_learning:
            inference_params = dict(
                epochs=self.epochs,
                alpha=self.alpha_ranges[0][0],  # initial learning rate
                min_alpha=self.alpha_ranges[-1][-1],  # final learning rate
            )
        logging.info(f"[Estimator.vec_transform] inference_params: {inference_params}")

        logging.info(
            f"[Estimator.vec_transform] Inferring{' normalized ' if normalized else ' '}vector embeddings..."
        )
        if normalized:
            return np.array(
                [
                    (vec := self.vec_model.infer_vector(td.words, **inference_params))
                    / np.linalg.norm(vec)
                    for td in X
                ]
            )
        else:
            return np.array([self.vec_model.infer_vector(td.words, **inference_params) for td in X])


#     def _vec_score(self, X_test_vec, y_test):
#         """Compute the AUC-PR score specifically for the Doc2Vec part of the model"""

#         # Compute mean vectors from the training set
#         mean_good_train_vector = self.vec_model.dv["good"]
#         mean_bad_train_vector = self.vec_model.dv["bad"]

#         # Use mean_good_train_vector and mean_bad_train_vector to compute scores
#         good_sims = cosine_similarity(X_test_vec, mean_good_train_vector.reshape(1, -1))
#         bad_sims = cosine_similarity(X_test_vec, mean_bad_train_vector.reshape(1, -1))
#         similarity_diff = good_sims - bad_sims
#         all_similarities = list(zip(similarity_diff.ravel(), np.nan_to_num(y_test)))

#         # Extract scores for labeled and unlabeled vectors
#         all_scores = np.array([sim[0] for sim in all_similarities]).reshape(-1, 1)
#         logging.info(
#             f"[Estimator.vec_score] raw all scores min/max: {(min(all_scores), max(all_scores))}"
#         )

#         # Fit MinMaxScaler for normalization using both labeled and unlabeled scores
#         scaler = MinMaxScaler((-6, 6)).fit(all_scores)

#         # Extract y_true and normalized_labeled_scores for labeled vectors
#         labeled_logical_idxs = ~np.isnan(y_test)
#         y_labeled_true = y_test[labeled_logical_idxs]
#         labeled_scores = all_scores[labeled_logical_idxs.ravel()]
#         logging.info(
#             f"[Estimator.vec_score] raw labeled_scores min/max: {(min(labeled_scores), max(labeled_scores))}"
#         )

#         # Apply normalization to labeled scores
#         normalized_labeled_scores = scaler.transform(labeled_scores)

#         # Transform normalized scores using sigmoid function
#         y_labeled_probs = expit(normalized_labeled_scores)
#         logging.info(f"[Estimator.vec_score] y_labeled_probs[:10]: {y_labeled_probs[:10]}")

#         # Compute AUC-PR using only normalized scores for labeled vectors
#         precision, recall, _ = precision_recall_curve(y_labeled_true, y_labeled_probs)
#         auc_pr = auc(recall, precision)
#         return auc_pr


# %% [markdown]
# Trying to fit a fraction of the data for testing

# %%
estimator = Doc2VecIsolationForestEstimator(
    random_state=cp.seed,
    onlne_learning=False,
    epochs=1,
    metric="bal_acc",
    train_score=True,
)
estimator.fit(toy_train_set, toy_train_set.labels)

# %%
score = estimator.score(toy_test_set, toy_test_set.labels)
print("score: ", score)

# %% [markdown]
# ## Hyperparameter Search
# Perform search to find best set of hyperparametes

# %%
# TESTESTEST

# halving_random_search.cv_results_

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

# Create a larger parameter grid with more combinations
param_dist = {
    "vector_size": randint(50, 601),  # Random integer between 150 and 600
    "epochs": randint(5, 21),  # Random integer between 5 and 20
    "dm": [0, 1],  # Distributed Memory (PV-DM) vs. Distributed Bag of Words (PV-DBOW)
    "window": randint(4, 13),
    "n_estimators": randint(50, 601),  # Random integer between 50 and 250
    "contamination": ["auto", 0.06],
    "max_features": uniform(0.5, 0.5),  # Uniform distribution between 0.5 and 1.0
    "bootstrap": [True, False],
    "onlne_learning": [True, False],
}

# Update the hyperparameter search to use the pipeline
halving_random_search = HalvingRandomSearchCV(
    estimator=Doc2VecIsolationForestEstimator(
        random_state=cp.seed,
        onlne_learning=True,
        epochs=5,
        #         n_jobs=1,
        #         workers=1,
        metric="bal_acc",
    ),
    param_distributions=param_dist,
    n_candidates="exhaust",
    verbose=1,
    random_state=cp.seed,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cp.seed),
    min_resources=15000,
    #     n_jobs = psutil.cpu_count(logical=False) - 1,
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

# %%
# Get the AUC-PR score of the best model on the test set
best_model_score = best_model.score(test_set, test_set.labels)
print("Best Model Score:", best_model_score)

# %% [markdown]
# # Visualize the decision boundary in 2D

# %%

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
