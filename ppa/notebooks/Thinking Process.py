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
# get all privacy policy markdown file paths in a (random) list
print("Loading all privacy policy paths to memory... ", end="")
policy_paths = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name != "README.md"]
print(f"Found {len(policy_paths):,} privacy policy files.")

# # TESTEST - use only N paths!
# N_PATHS = 1_000
# print(f"\nWARNING! USING ONLY {N_PATHS:,} PATHS!")
# policy_paths = policy_paths[:N_PATHS]

# %% [markdown]
# # WIP - Better privacy-policy tokenization
#
# Load a sample raw policy (Wikimedia foundation):

# %%
# wikimedia_foundation_pp_path = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name == "wikivoyage.org.md"][0]

# %% [markdown]
# Current pre-processing

# %%
# URL_PATTERN = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
# EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

# def preprocess_document_v1(fpath: Path, url_pattern=URL_PATTERN, email_pattern=EMAIL_PATTERN, should_filter_stopwords=True):

#     # Read all but the header
#     with open(fpath, "r", encoding="utf-8") as f:
#         _, *doc_lines = f.readlines()
#         doc = "\n".join(doc_lines)[2:]

#     # Replace URLs with "<URL>" and email addresses with "<EMAILADD>"
#     if url_pattern:
#         doc = re.sub(url_pattern, "<URL>", doc)
#     if email_pattern:
#         doc = re.sub(email_pattern, "<EMAILADD>", doc)

#     # Tokenize the text
#     doc = gensim.utils.simple_preprocess(doc)

#     # Remove stopwords and return
#     if should_filter_stopwords:
#         return gensim.parsing.preprocessing.remove_stopword_tokens(doc)
#     else:
#         return doc

# tokens = preprocess_document_v1(wikimedia_foundation_pp_path)
# print("len(tokens): ", len(tokens))
# print(" ".join(tokens))

# %% [markdown]
# Now with keeping of hyphenated expressions, and with lemmatization:

# %%
# import re
# import gensim
# from nltk.stem import WordNetLemmatizer

# def tokenize_text(text: str, filter_stopwords: bool=True, lemmatize: bool=True):
#     """Doc."""

#     # Use regular expressions to find hyphenated words and replace hyphens with " hyph "
#     text_with_hyph = re.sub(r'([A-Za-z0-9]+)-([A-Za-z0-9]+)', r'\1 hyph \2', text)

#     # Tokenize the text using simple_preprocess
#     tokens = gensim.utils.simple_preprocess(text_with_hyph, min_len=2, max_len=20)

#     # Lemmatize (optional)
#     if lemmatize:
#         # Initialize the WordNet lemmatizer
#         lemmatizer = WordNetLemmatizer()
#         # Lemmatize the tokens
#         tokens = [lemmatizer.lemmatize(token) for token in tokens]

#     # Merge tokens with "hyph" between them into hyphenated tokens
#     hyphenated_tokens = []
#     for i in range(len(tokens) - 2):
#         if tokens[i+1] == "hyph":
#             hyphenated_token = tokens[i] + "-" + tokens[i + 2]
#             hyphenated_tokens.append(hyphenated_token)
#         elif tokens[i] == "hyph" or tokens[i-1] == "hyph":
#             pass
#         else:
#             hyphenated_tokens.append(tokens[i])

#     # Remove stopwords (optional)
#     if filter_stopwords:
#         hyphenated_tokens = gensim.parsing.preprocessing.remove_stopword_tokens(hyphenated_tokens)


#     # Remove consecutive duplicates
#     tokens = [hyphenated_tokens[0]]
#     for i in range(1, len(hyphenated_tokens)):
#         if hyphenated_tokens[i] != hyphenated_tokens[i - 1]:
#             tokens.append(hyphenated_tokens[i])

#     return tokens

# def preprocess_document_v2(fpath: Path, **kwargs):

#     # Read all but the header
#     with open(fpath, "r", encoding="utf-8") as f:
#         _, *doc_lines = f.readlines()
#         doc = "\n".join(doc_lines)[2:]

#     # Find and replace links with the text inside the square brackets
#     md_link_pattern = r'\[([^\]]+)\]\([^\)]+\)'
#     doc = re.sub(link_pattern, r'\1', doc)

#     # Replace URLs with "<URL>" and email addresses with "<EMAILADD>"
#     url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
#     doc = re.sub(url_pattern, "URL", doc)

#     # Tokenize the text
#     tokens = tokenize_text(doc, **kwargs)

#     return tokens

# tokens = preprocess_document_v2(wikimedia_foundation_pp_path, lemmatize=True, filter_stopwords=True)
# print("len(tokens): ", len(tokens))
# print(" ".join(tokens))

# %% [markdown]
# # TODO: try identifying the name of the company/URL in the policy (using its filepath or header?) and convert to a special token \<company>
# # TODO: implement a second processing step after Dictionary object is created, which should possibly contain:
# ## 1) iterating over all documents, removing any tokens which appear only in them and only once.
# ## 2) iterating over all documents, noting any tokens which appear only in them more than once and replacing them with a special token such as \<topic> or \<company>?..
# ## 3) try to compose a list of privacy-domain words/n-grams, perhpas by inspecting the dictionary or the corpus itself (for n-grams), and incorporate 'Privacy Term Highlighting' (see ChatGPT conversation) for converting them into special tokens (such as by adding square brackes around these expressions)
# ## 4) Consider using the full data using the ToS;DR API for extracting important features existing in general in PPs so that these could be used for feature engineering (selecting best tokens) for all PPs. This could better embed the privacy-oriented properties of PPs (and not themes)

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

Beep(1000, 500)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens). It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset:

# %%
N = cp.total_samples // 10
pp_lengths = np.array(
    [len(tagged_doc.words) for tagged_doc in cp.generate_samples(n_samples=N, shuffled=True)]
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
# The token you want to search for
target_token = "url"

# Iterate through each document and check if the token is in the document
for tagged_doc in cp.generate_samples(n_samples=N):
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
# N = cp.total_samples
N = cp.total_samples
TEST_FRAC = 0.2
train_set, test_set = cp.generate_train_test_sets(n_samples=N, test_frac=TEST_FRAC, shuffled=True)
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
    unsupervised_model = Doc2Vec.load(str(MODEL_PATH))

    # Done
    print("Unsupervised Model Loaded")
    # TODO: print some model details

else:
    # Initialize and train the Doc2Vec model
    unsupervised_model_kwargs = {
        "vector_size": 300,
        "window": 5,
        "hs": 1,
        "negative": 0,
        "epochs": 10,
        #         "workers": mp.cpu_count(),
    }
    unsupervised_model = Doc2Vec(**unsupervised_model_kwargs)

    # Build vocabulary
    print("Building vocabulary... ", end="")
    unsupervised_model.build_vocab(train_set)
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

    # Train the model
    print(f"Training unsupervised model... ", end="")
    tic = time.perf_counter()
    unsupervised_model.train(
        train_set, total_examples=unsupervised_model.corpus_count, epochs=unsupervised_model.epochs
    )

    # Save the trained model for future use
    print(f"Saving to '{MODEL_PATH}'... ", end="")
    unsupervised_model.save(str(MODEL_PATH))
    print("Done!")

    # Done!
    print(f"Training timing: {(time.perf_counter() - tic)/60:.1f} mins")
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms


# %% [markdown]
# ## 3.2 Sanity check
# ### TODO: Admit this check into the codebase
# check that documents are most similar to themselves. I do not expect/desire to optimize the fraction of model-inferred documents which are most similar to themselves, as this might mean the model is overfitting. Instead, this is just a test to see that the model does something logical.
#
# It is worth mentioning that while purposfully attempting to overfit a small subset of 1000 documents and using a complex model (increased vector size), I was only able to reach about 80% - I attribute this to noise in the training data.

# %%
from collections import Counter

if SHOULD_RETRAIN:
    model = unsupervised_model

    # Set the number of top similar documents to consider
    SAMPLE_SIZE = N // 10
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
    print("Skipping...")

# %% [markdown]
# # 4 Model Evaluation
#
# Doc2Vec is an unsupervised model, so finding a metric to evaluate it is not a straightforward task, given the fact that we also do not have any labeled data.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %% [markdown]
# ## 4.1 Inferring Vectors for Test Data

# %%
from ppa.display import Plotter, display_dim_reduction

model = unsupervised_model

# Infer document vectors for the test data
print("Inferring vectors for test documents... ", end="")
document_vectors_array = np.array([model.infer_vector(doc.words) for doc in test_set])
print("Done.")

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# ## 4.2 Visualizing the Inferred Documents

# %% [markdown]
# PCA

# %%
from sklearn.decomposition import PCA

N_samples = 10_000

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(document_vectors_array)

annots = [
    tagged_doc.tags[0]
    for idx, tagged_doc in enumerate(test_set)
    if (idx < N_samples) and (idx % 10 == 0)
]
display_dim_reduction(pca_result, "PCA", annots=annots, figsize=(10, 8))

# %% [markdown]
# Let's try t-SNE as well

# %%
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=5,
    learning_rate=200,
    n_iter=500,
    n_iter_without_progress=200,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(document_vectors_array)

annots = [
    tagged_doc.tags[0]
    for idx, tagged_doc in enumerate(test_set)
    if (idx < N_samples) and (idx % 10 == 0)
]
display_dim_reduction(tsne_result, "t-SNE", annots=annots, figsize=(10, 8))

# %% [markdown]
# We need to get some clue as to what the above means. Let's try gathering several "good" and "bad" privacy policies, and see where they stand in the PCA picture.

# %% [markdown]
# # 5 Incorprating labeled data
# ## 5.1 ETL for Policy Ratings
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
# ## 5.2 Exploration
# ### 5.2.1 Checking for duplicates in data according to rating IDs
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
# ### 5.2.2 Checking for Bias in Labeled Data

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
# ### 5.2.3 Visualizing Labeled Data Using Unsupervised Model
# Let's visualize the all labeled data over the unsupervised model we have trained.
#
# First, let's update the corpus with labeled data, and save it separately:

# %%
# SHOULD_FORCE_LABELING = True
SHOULD_FORCE_LABELING = False

url_rating_dict = ratings_df.set_index("tag")["rating"].to_dict()
cp.add_label_tags(url_rating_dict, force=SHOULD_FORCE_LABELING)

# %% [markdown]
# Next, let's infer vectors for all the policies for which we have labels:

# %%
from ppa.utils import get_file_index_path
from typing import Dict, List
import json

labeled_corpus_index_path = get_file_index_path(cp.labeled_corpus_path)

index_dict: Dict[str, List[int]] = {"good": [], "bad": [], "unlabeled": []}
with open(labeled_corpus_index_path, "r") as idx_file:
    while True:
        try:
            start_pos, note = json.loads(idx_file.readline())
            index_dict[note].append(start_pos)
        except json.JSONDecodeError:
            break

labeled_start_pos = index_dict["good"] + index_dict["bad"]
print(f"{len(labeled_start_pos)} labeled policies indexed (good + bad)")

# %%
from ppa.display import Plotter, display_dim_reduction
from ppa.ppa import SampleGenerator

# define the model
model = unsupervised_model

print("Gathering all rated policies... ", end="")
labeled_policies = [
    tagged_doc for tagged_doc in SampleGenerator(cp.labeled_corpus_path) if len(tagged_doc.tags) > 1
]
labels = [td.tags[1] for td in labeled_policies]
print("Done.")

# Infer document vectors for the test data
print("Inferring vectors for labeled policies... ", end="")
document_vectors = [model.infer_vector(doc.words) for idx, doc in enumerate(labeled_policies)]
print("Done.")

# Convert document vectors to a numpy array
document_vectors_array = np.array(document_vectors)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# And now, let's visualize them, with only the "good" policies annotated by URL:

# %% [markdown]
# # TODO: try visualizing with labels!!!

# %%
from sklearn.decomposition import PCA

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(document_vectors_array)

annots = [
    tagged_doc.tags[0]
    for tagged_doc in labeled_policies
    if ratings_df.loc[ratings_df["tag"] == tagged_doc.tags[0], "rating"].iloc[0] == "good"
]
# display_dim_reduction(pca_result, "PCA", labels=labels, annots=annots, figsize=(10, 8))
display_dim_reduction(pca_result, "PCA", labels=labels, figsize=(10, 8))

# %%
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=5,
    learning_rate=200,
    n_iter=1000,
    n_iter_without_progress=500,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(document_vectors_array)

annots = [
    tagged_doc.tags[0]
    for tagged_doc in labeled_policies
    if ratings_df.loc[ratings_df["tag"] == tagged_doc.tags[0], "rating"].iloc[0] == "good"
]
# display_dim_reduction(tsne_result, "t-SNE", annots=annots, figsize=(10, 8))
display_dim_reduction(tsne_result, "t-SNE", labels=labels, figsize=(10, 8))

# %% [markdown]
# So, in both PCA and t-SNE visualizations, we see that no pattern emerges for "good" or "bad" policies. Essentially, this means that the current model does not capture what separates "good"/"bad" policies.
# I will now try retraining the model with the new labels

# %% [markdown]
# # 6 Retrainning the Model with Some Labeled Data (Semi-Supervised?)

# %% [markdown]
# Split to train/test sets in a stratified fashion, i.e. keep the same label ratio (in this case the percentages of "good" and "bad" policies) in the data.

# %%
N = cp.total_samples

train_set, test_set = cp.generate_train_test_sets(
    n_samples=N, test_frac=TEST_FRAC, labeled=True, shuffled=True
)

# TEST - check percentages in train/test splits
from collections import Counter

print(Counter([doc.tags[1] if len(doc.tags) > 1 else "unlabeled" for doc in train_set]))
print(Counter([doc.tags[1] if len(doc.tags) > 1 else "unlabeled" for doc in test_set]))

Beep(1000, 500)

# %% [markdown]
# Re-train the model (now semi-supervised):

# %%
# define save/load path
MODEL_PATH = MODEL_DIR_PATH / "privacy_policy_doc2vec.model"

SHOULD_RETRAIN = True
# SHOULD_RETRAIN = False

if not SHOULD_RETRAIN:
    # load the last trained model
    semi_supervised_model: Doc2Vec = Doc2Vec.load(str(MODEL_PATH))

    # Done
    print("Semi-supervised model loaded")
    # TODO: print some model details

else:
    # Initialize and train the Doc2Vec model
    semi_supervised_model_kwargs = {
        "vector_size": 300,
        "window": 5,
        "hs": 1,
        "negative": 0,
        "epochs": 10,
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
        total_examples=unsupervised_model.corpus_count,
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
# Sanity check

# %%
from collections import Counter

if SHOULD_RETRAIN:
    model = semi_supervised_model

    # Set the number of top similar documents to consider
    SAMPLE_SIZE = N // 10
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
    print("Skipping...")

# %% [markdown]
# Visualizing the results

# %%
# define the model
model = semi_supervised_model

# Infer document vectors for the test data
print("Inferring vectors for test policies... ", end="")
test_vectors, test_tags = zip(*[(model.infer_vector(td.words), td.tags) for td in test_set])
test_vectors = np.array(test_vectors)
print("Done.")

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# PCA

# %%
# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(test_vectors)

annots = [
    tagged_doc.tags[0]
    for tagged_doc in test_set
    if len(tagged_doc.tags) > 1 and tagged_doc.tags[1] == "good"
]
display_dim_reduction(pca_result, "PCA", annots=annots, figsize=(10, 8))

# %% [markdown]
# t-SNE

# %%
tsne = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate=200,
    n_iter=1000,
    n_iter_without_progress=500,
    random_state=SEED,
)
tsne_result = tsne.fit_transform(test_vectors)

annots = [
    tagged_doc.tags[0]
    for tagged_doc in test_set
    if len(tagged_doc.tags) > 1 and tagged_doc.tags[1] == "good"
]
display_dim_reduction(tsne_result, "t-SNE", annots=annots, figsize=(10, 8))

# %% [markdown]
# Devising a metric: Perhaps the similarity between like-labled policies is lost in the dimensionality reduction. Let's try measuring the cosine similarity between the vectors directly

# %% [markdown]
# # TODO: I need to figure out how to make this into a metric (ask ChatGPT) - I have two dictionaries of similarities (good, bad). I could decide on some threshold above which policies are predicted as good/bad, then calculate some accuracy score (biased for bad!!!) which basically counts how many predicted policies are actually labeled as predicted.
# # TODO: once the metric is ready, I should test my new preprocessing routine and see if it improves the metric

# %%
# from sklearn.metrics.pairwise import cosine_similarity

# print("Gathering good and bad vector lists: ", end="")
# # train
# train_vectors, train_tags = zip(*[(model.dv[td.tags[0]], td.tags) for td in train_set])
# mean_good_train_vector = np.array([vec for vec, tags in zip(train_vectors, train_tags) if len(tags) > 1 and tags[1] == "good"]).mean(axis=0)
# mean_bad_train_vector = np.array([vec for vec, tags in zip(train_vectors, train_tags) if len(tags) > 1 and tags[1] == "bad"]).mean(axis=0)

# # test
# labeled_test_vectors, labeled_test_tags = zip(*[(vec, tags) for vec, tags in zip(test_vectors, test_tags) if len(tags) > 1])
# print("Done.")

# # Beep when done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# similarities_good = {}
# similarities_bad = {}
# for test_tag, test_policy_vector in zip(labeled_test_tags, labeled_test_vectors):
#     # Calculate similarity with "good" policies
#     similarity_good = cosine_similarity([test_policy_vector], [mean_good_train_vector])
#     similarities_good[test_tag[0]] = (similarity_good[0][0], test_tag[1])

#     # Calculate similarity with "bad" policies
#     similarity_bad = cosine_similarity([test_policy_vector], [mean_bad_train_vector])
#     similarities_bad[test_tag[0]] = (similarity_bad[0][0], test_tag[1])

# # Beep when done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
from sklearn.metrics.pairwise import cosine_similarity

print("Gathering good and bad vector lists: ", end="")
# train
train_vectors, train_tags = zip(*[(model.dv[td.tags[0]], td.tags) for td in train_set])
mean_good_train_vector = np.array(
    [vec for vec, tags in zip(train_vectors, train_tags) if len(tags) > 1 and tags[1] == "good"]
).mean(axis=0)
mean_bad_train_vector = np.array(
    [vec for vec, tags in zip(train_vectors, train_tags) if len(tags) > 1 and tags[1] == "bad"]
).mean(axis=0)

# test
labeled_test_vectors, labeled_test_tags = zip(
    *[(vec, tags) for vec, tags in zip(test_vectors, test_tags) if len(tags) > 1]
)
print("Done.")

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

similarities = {}
for test_tag, test_policy_vector in zip(labeled_test_tags, labeled_test_vectors):

    if test_tag[1] == "good":
        similarity = cosine_similarity([test_policy_vector], [mean_good_train_vector])
    else:  # bad
        similarity = cosine_similarity([test_policy_vector], [mean_bad_train_vector])
    similarities[test_tag[0]] = (similarity[0][0], test_tag[1])

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

# %%
from sklearn.metrics import roc_auc_score

# Collect predicted scores and true labels for "good" and "bad" policies
good_true_labels, good_similarity_scores = zip(
    *[(true_label == "good", score) for score, true_label in similarities.values()]
)
bad_true_labels, bad_similarity_scores = zip(
    *[(true_label == "bad", score) for score, true_label in similarities.values()]
)

# Calculate ROC AUC for "good"/"bad" policies
roc_auc_good = roc_auc_score(good_true_labels, good_similarity_scores, average="weighted")
roc_auc_bad = roc_auc_score(bad_true_labels, bad_similarity_scores, average="weighted")

print(f"ROC AUC (good): {roc_auc_good:.2f}")
print(f"ROC AUC (bad): {roc_auc_bad:.2f}")


# %%
# good_threshold = 0.26  # Adjust the threshold as needed
# bad_threshold = -0.5  # Adjust the threshold as needed

# # Initialize counters for evaluation metrics
# true_good = 0
# predicted_good = 0
# correctly_predicted_good = 0
# true_bad = 0
# predicted_bad = 0
# correctly_predicted_bad = 0

# # Iterate through test policies and their predicted labels
# for (similarity_score, true_label) in similarities_good.values():

#     if true_label == "good":
#         true_good += 1

#     if similarity_score >= good_threshold:
#         predicted_good += 1
#         if true_label == "good":
#             correctly_predicted_good += 1

# for (similarity_score, true_label) in similarities_bad.values():

#     if true_label == "bad":
#         true_bad += 1

#     if similarity_score >= bad_threshold:
#         predicted_bad += 1
#         if true_label == "bad":
#             correctly_predicted_bad += 1

# # Calculate precision and recall
# precision_good = correctly_predicted_good / predicted_good if predicted_good > 0 else 0
# recall_good = correctly_predicted_good / true_good if true_good > 0 else 0
# f1_good = 2 / (1 / precision_good + 1 / recall_good)

# precision_bad = correctly_predicted_bad / predicted_bad if predicted_bad > 0 else 0
# recall_bad = correctly_predicted_bad / true_bad if true_bad > 0 else 0
# f1_bad = 2 / (1 / precision_bad + 1 / recall_bad)

# # Calculate bias-adjusted accuracy for "bad" policies
# bias_adjusted_accuracy_bad = (precision_bad + recall_bad) / 2

# print(f"Precision (good): {precision_good:.2f}")
# print(f"Recall (good): {recall_good:.2f}")
# print(f"F1 (good): {f1_good:.2f}")
# print()
# print(f"Precision (bad): {precision_bad:.2f}")
# print(f"Recall (bad): {recall_bad:.2f}")
# print(f"F1 (bad): {f1_bad:.2f}")
# print(f"Bias-adjusted Accuracy (bad): {bias_adjusted_accuracy_bad:.2f}")


# %% [markdown]
# # Label test policies according to nearest labeld policy from training coprus

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
# # TODO: this will only be relevant once a metric is devised
# Attempting to implement hyperparameter search

# %%
# # Sample tagged documents training data
# print("Preparing training data... ", end="")
# N = 10_000
# train_data = cp.generate_samples(n_samples=N)
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
