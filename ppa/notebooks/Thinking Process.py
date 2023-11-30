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
# ### TODO: try oversampling the known good policies - this might be a problem since I have very little labeled good policies - this might work together with pseudo-labeling using high thresholds?
# ### TODO: separate files for corpus before bigrams and before filtering? could save time trying different pre-processings
# ### TODO: Rethink online training - perhaps the Doc2Vec should train alone for a few epochs before beginning training on the IsolationForest? Perhaps also a more linear increase in n_estimators is more appropriate for the forest. Another option - perhaps the number of trees trained each epoch should accelarate instead of deaccelarating? i.e. keep the Doc2Vec epochs as is but flip the n_estimators increments?
# ### TODO: Try feature selection
# ### TODO: consider "pseudo-labeling" - predicting labels from model then re-training, testing each iteration.
# ### TODO: Figure out why there seems to be a few more vectors in the model.dv then there are training samples??? - Perhaps these are the 'summary' vectors for the labels?
# ### TODO: Try [UMAP](https://github.com/lmcinnes/umap) visualization, for speed if anything else
# ### TODO: try these suggestions:
# * Topic Modeling: Consider using topic modeling techniques (e.g., Latent Dirichlet Allocation - LDA) to identify underlying topics within the documents. Visualize the topics and their prevalence in the dataset.
# * Named Entity Recognition (NER): If applicable, perform NER to extract entities like names, organizations, locations, and dates from the text. Explore the frequency and distribution of entities in the documents.

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

# TESTEST - use only N paths!
N_PATHS = 100_000_000
# N_PATHS = 2_000
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
# Create a fresh CorpusProcessor instance, build a `gensim.corpora import Dictionary` and process the entire corpus, all while streaming to/from disk.

# %%
from ppa.processing import CorpusProcessor

SHOULD_REPROCESS = True
# SHOULD_REPROCESS = False

MODEL_DIR_PATH = Path.cwd().parent / "models"

# build and save dictionary from all documents, process all documents and serialize (compressed) the TaggedDocument objects to disk
if SHOULD_REPROCESS:
    # create a document processor with the paths
    cp = CorpusProcessor(
        policy_paths,
        MODEL_DIR_PATH,
        seed=SEED,
    )
    cp.process_corpus(
        lemmatize=True,
        should_filter_stopwords=True,
        bigrams=True,
        n_below=None,
        no_above=1.0,
        min_percentile=1,
        max_percentile=99,
        threshold=0.5,
    )
else:
    cp = CorpusProcessor.load(MODEL_DIR_PATH / "corpus_processor.pkl")

Beep(1000, 500)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens). It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset:

# %%
pp_lengths = np.array(
    [
        len(tagged_doc.words)
        for tagged_doc in cp.generate_samples(n_samples=5_000, shuffled_idx=True, shuffled_gen=True)
    ]
)

print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

with Plotter() as ax:
    ax.hist(pp_lengths, int(np.sqrt(pp_lengths.size)))

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

# %% [markdown]
# Now, let's take a look at the `gensim.corpora.Dictionary` we created from the entire corpus:

# %%
cp.dict_info()

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
# import re

# # The pattern you want to search for using regular expression
# # target_pattern = r'\b[a-zA-Z]<|>\b[a-zA-Z]\b'
# target_pattern = "><"  # twisternederland.com

# # Compile the regular expression pattern
# pattern = re.compile(target_pattern)

# # Iterate through each document and check if the pattern matches the document
# for tagged_doc in cp.generate_samples():
#     text = " ".join(tagged_doc.words)
#     if pattern.search(text):
#         print(text)
#         break

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
# Finally, let's update the corpus with labeled data, and save it separately:

# %%
SHOULD_FORCE_LABELING = True
# SHOULD_FORCE_LABELING = False

if SHOULD_FORCE_LABELING:
    url2label = ratings_df.set_index("tag")["label"].to_dict()
    cp.add_label_tags(url2label)

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
# Train the Doc2Vec model (semi-supervised), and save:

# %%
from ppa.estimators import D2VClassifier
from ppa.utils import timer
from datetime import datetime
import time

SHOULD_FIT_MODEL = True
# SHOULD_FIT_MODEL = False

if SHOULD_FIT_MODEL:
    # initialize classifier
    classifier = D2VClassifier(
        random_state=cp.seed,
        window=5,
        vector_size=100,
        epochs=10,
        train_score=True,
        #     iterative_training=True,
        workers=psutil.cpu_count(logical=False) - 1,
    )

    # fit the model
    classifier.fit(train_set, train_set.labels, X_test=test_set, y_test=test_set.labels)

    # score the model
    print("Balanced ACC: ", classifier.score(test_set, test_set.labels))

    # save
    dt_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    classifier.model.save(f"D:/MEGA/Programming/ML/PPA/ppa/models/pp_d2v_{dt_str}.model")

# %%
raise RuntimeError("STOP HERE!")

# %% [markdown]
# Load existing model to estimator

# %%
from gensim.models.doc2vec import Doc2Vec

if not SHOULD_FIT_MODEL:
    classifier = D2VClassifier(
        random_state=cp.seed,
        window=5,
        vector_size=100,
        epochs=10,
        train_score=True,
        #     iterative_training=True,
        workers=psutil.cpu_count(logical=False) - 1,
    )

    MODEL_DT = "27112023_154831"
    model_fname = f"pp_d2v_{MODEL_DT}.model"
    classifier.model = Doc2Vec.load(f"D:/MEGA/Programming/ML/PPA/ppa/models/{model_fname}")

    # score the model
    print("Balanced ACC: ", classifier.score(test_set, test_set.labels))

    # Beep when done
    Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# # 5 Doc2Vec Model Evaluation
#
# Doc2Vec, using secondary tags, is at best a weakly semi-supervised model, so finding a metric to evaluate it is not a straightforward task.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %% [markdown]
# ## 5.1 Sanity check
# As a first test of the model, a reasonable sanity check (adapted from that suggested by [Radim Řehůřek](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py) himself) would be to see if most vectors inferred for the policies the model was trained upon are most similar to the corresponding document vectors of the model itself.

# %%
classifier.sanity_check(train_set, max_rank=10, plot=True)

# %% [markdown]
# ## 5.2 Visualizing the results using dimensionallity reduction to 2D
# We begin by inferring vectors for all labeled samples in the test set

# %%
# Infer document vectors for the test data
# convet labeles to an array and keep only "good"/"bad" elements, and their indices
labeled_test_tags, labeled_test_idxs = classifier.valid_labels(test_set.labels)
test_set_labeled = test_set.sample(idxs=np.nonzero(labeled_test_idxs)[0])
test_vectors_labeled = classifier.transform(test_set_labeled, normalized=True)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# ### 5.2.1 PCA

# %%
from sklearn.decomposition import PCA
from ppa.display import display_dim_reduction

test_labels = ["good" if val == -1 else "bad" for val in labeled_test_tags]

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(test_vectors_labeled)

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
tsne_result = tsne.fit_transform(test_vectors_labeled)

# annots = [
#     tagged_doc.tags[0]
#     for tagged_doc in test_set
#     if len(tagged_doc.tags) > 1 and tagged_doc.tags[1] == "good"
# ]
display_dim_reduction(tsne_result, "t-SNE", labels=test_labels, figsize=(10, 8))

# %% [markdown]
# I cannot see any pattern separating "good" policies from "bad" ones. This doesn't mean the model isn't sensitive to the labels, only that the 2D visualizations don't seem to capture it.

# %% [markdown]
# ## 5.3 Devising a metric <a id='sec-5-3'></a>
# Perhaps the similarity between like-labled policies is lost in the dimensionality reduction. Let's try measuring the cosine similarity between the vectors directly. We can check out the distribution of my custom similarity scores separately for "good" and "bad" policies, as well as for the rest of the (unlabeled) policies:

# %%
y_test_scores_labeled = classifier.decision_function(test_set_labeled)
y_test_scores_unlabeled = classifier.decision_function(
    test_set.sample(1_000, idxs=np.nonzero(~labeled_test_idxs)[0])
)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
with Plotter(
    figsize=(10, 6),
    xlabel="Similarity Scores",
    ylabel="Bin Counts (Density)",
    suptitle="Distribution of Similarity Scores",
) as ax:

    # Plot the distribution of similarity scores
    ax.hist(
        good_scores := y_test_scores_labeled[labeled_test_tags == -1],
        bins=int(np.sqrt(good_scores.size)),
        density=True,
        alpha=0.6,
        label=f"Good ({good_scores.size})",
    )
    ax.hist(
        bad_scores := y_test_scores_labeled[labeled_test_tags == 1],
        bins=int(np.sqrt(bad_scores.size)),
        density=True,
        alpha=0.6,
        label=f"Bad ({bad_scores.size})",
    )
    ax.hist(
        y_test_scores_unlabeled,
        bins=int(np.sqrt(len(y_test_scores_unlabeled))),
        density=True,
        alpha=0.3,
        label=f"Unlabeled (sample of {len(y_test_scores_unlabeled):,})",
        zorder=1,
    )

    ax.legend(title="Label (Num.)")
    ax.grid(True)

# %% [markdown]
# This shows how good the model is at separating "good" policies from "bad" ones - the less overlap between the two histograms, the less errors it will make when classifying. We will, for now, score the model using balanced accuracy, using the above method to predict "good" policies for those scoring above a threshold (default 0.5). It is worth mentioning that this overlap unavoidable if there are incorrect labels or otherwise noisy data (bad preprocessing etc.)

# %% [markdown]
# ### 5.3.1 Investigating Specific Policies
# In the distributions above we can see significant overlap between good/bad policies, including some outliers which score far beyond their group mean. Lets try to understand why - this could be the key to devise a better model.

# %% [markdown]
# Let's get the URLs, labels and scores in a `DataFrame`:

# %%
df = pd.DataFrame(
    {
        "url": [td.tags[0] for td in test_set_labeled],
        "label": test_set_labeled.labels,
        "score": y_test_scores_labeled,
    }
)

# %% [markdown]
# What seems interesting to test now is what the actual 'letter score' (from ToS;DR) is for each of the so-called outliers among the top scores - if the outliers are all 'C's, this could mean that them being labeled "bad" is a consequence of the binary labeling forced unto the 5 letter ratings:

# %%
merged_df = pd.merge(df, ratings_df[["tag", "rating"]], left_on="url", right_on="tag", how="left")
df["letter"] = merged_df["rating"]
df.sort_values(by="score", ascending=False).iloc[:20]

# %% [markdown]
# So, while "bad"-labled policies with high scores are indeed 'C's, there are some 'D' there too. Let's check out the distributions as in 5.3, this time separated and colored by letter rating:

# %%
with Plotter(
    suptitle="Histograms of Scores by Letter Rating",
    xlabel="Score",
    ylabel="Bin Counts (Density)",
) as ax:

    # Iterate through unique letters and plot histograms for each
    for letter in sorted(df["letter"].unique(), reverse=True):
        scores_for_letter = df[df["letter"] == letter]["score"]
        ax.hist(
            scores_for_letter,
            bins=int(np.sqrt(len(scores_for_letter))),
            density=True,
            alpha=0.25,
            label=f"{letter} ({len(scores_for_letter):,})",
        )

    ax.hist(
        y_test_scores_unlabeled,
        bins=int(np.sqrt(len(y_test_scores_unlabeled))),
        density=True,
        alpha=0.5,
        label=f"Unlabeled (sample of {len(y_test_scores_unlabeled):,})",
        zorder=-1,
    )

    ax.legend(title="Letter (Num.)")

# %% [markdown]
# We can see that all but the 'E'-rated policies have some tail in the high-scores, and even more disturbing is the fact that most 'A'-rated policies have fairly low scores!

# %% [markdown]
# One solution for this could be simply wrong labels. Let's check out the worst-scored "good" policies. Since there are so little labeled policies in thee test set, we can check them all out together in one table:

# %%
df[df["label"] == "good"].sort_values(by="score")

# %%
" ".join(test_set["vpl.ca"].words)

# %% [markdown]
# Since I recognize some of the URLs as having legitimately "good" privacy policies, and the worst scored one is that of Vancouver's public library, I have no reason to suspect the labels, and I therefore blame my model. It could be that there's just not enough good privacy models in my data for Doc2Vec to pick up on the qualities of good privacy policies.

# %% [markdown]
# # 6 Considering Upsampling and Pseudo-Labeling
#
# Being unable to increase my model's balanced-accuracy metric above about 0.8 using hyperparameter tuning or better data pre-processing (I have tinkered with both for a while now). What could be the reason is the imbalance existing in the data - there are about 17 times more "bad" labels than "good" ones, and I expect about the same imbalance in the unlabeled corpus. I can attempt to improve the data in two (possibly combined) methods:
# * **Upsampling (mixing-in copies of) "good"-labeled policies in the training data**: This might make the model 'understand' "good" policies better, as the data will be less biased (need to watch for overfitting of the "good" policies)
# * **Pseudo-labeling unlabeled training data using high/low thresholds for prediction**: This would make the scoring more robust as model summary vectors (specifically the "good" one) are averaged over more samples, enabling finer tuning according to score. This could also potentially enable supervised classification later on.

# %% [markdown]
# ## 6.1 Pseudo-Labeling
#
# ### 6.1.1 Score Thresholds
# First, let's see how many candidates I have for pseudo-labeling. Taking a look again at the histograms in [5.3](#sec-5-3), I can "safely" set the thresholds at below 0.46 for "bad" policies and above 0.55 for "good" policies. Let's see how many new labeles I would be able to obtain this way. Let's try plotting the score distributions for the training set.
#
# <!--
# We begin by getting the scores for the training set vectors, which are essentially the model's 'document vectors' (no need to infer):
# # TODO: instead of iterating over the disk-bound corpus for getting merely the relevant URLs, use the index file!
# # TODO: Figure out why I get normal-like distributions for all classes (good, bad, unlabeled) when using the model vectors instead of inferred vectors
#
# CODE (for using model's 'document vectors' instead of inference):
# # convet labeles to an array and keep only "good"/"bad" elements, and their indices
# train_labels, labeled_train_idxs = classifier.valid_labels(train_set.labels)
#
# # get the labeled/unlabeled training set
# train_set_labeled = train_set.sample(idxs=np.nonzero(labeled_train_idxs)[0])
# train_set_unlabeled = train_set.sample(idxs=np.nonzero(~labeled_train_idxs)[0])
#
# # Get transformed train set (labeled)
# train_vec_labeled = np.empty((len(train_set_labeled), classifier.model.vector_size))
# for idx, td in enumerate(train_set_labeled):
#     train_vec_labeled[idx] = classifier.model.dv[td.tags[0]]
#
# # Get transformed train set (unlabeled)
# train_vec_unlabeled = np.empty((len(train_set_unlabeled), classifier.model.vector_size))
# for idx, td in enumerate(train_set_unlabeled):
#     train_vec_unlabeled[idx] = classifier.model.dv[td.tags[0]]
#
# # get the scores
# y_train_scores_labeled = classifier.decision_function(X_vec=train_vec_labeled)
# y_train_scores_unlabeled = classifier.decision_function(X_vec=train_vec_unlabeled)
#
# # Beep when done
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
#
# -->
#
# We begin by inferring vectors for all training policies and getting the scores (THIS MIGHT TAKE A WHILE!):

# %%
# get indices for labeled training samples
train_labels, labeled_train_idxs = classifier.valid_labels(train_set.labels)

# getting the
y_train_scores_labeled = classifier.decision_function(
    train_set.sample(idxs=np.nonzero(labeled_train_idxs)[0])
)
y_train_scores_unlabeled_sample = classifier.decision_function(
    train_set.sample(1_000, idxs=np.nonzero(~labeled_train_idxs)[0])
)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %%
with Plotter(
    figsize=(10, 6),
    xlabel="Similarity Scores",
    ylabel="Bin Counts (Density)",
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
# Now, let's count how many new good/bad pseudo-labeled policies we can hope for, in a preliminary iteration. For this, we need to get the scores for the entire unlabeled training corpus (for displaying the distribution, only a small sample was used):

# %%
y_train_scores_unlabeled = classifier.decision_function(
    train_set.sample(idxs=np.nonzero(~labeled_train_idxs)[0])
)

# Beep when done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# Now let's set the thresholds to "safe" values according to the "good"/"bad" distributions above:

# %%
print(
    f"Unlabeled score range: {min(y_train_scores_unlabeled):.2f}-{max(y_train_scores_unlabeled):.2f}"
)

THRESH_GOOD = 0.55
THRESH_BAD = 0.44

print("Potential 'good' pseudo-labels: ", sum(y_train_scores_unlabeled > THRESH_GOOD))
print("Potential 'bad' pseudo-labels: ", sum(y_train_scores_unlabeled < THRESH_BAD))

# %% [markdown]
# It appears that we can barely squeeze out about 20 new "good" labels...

# %% [markdown]
# ### 6.1.2 Most-Similar Labeled Policies
#
# A different Idea, utilizing Doc2Vec's prized similarity methods, could be checking out the most similar labeled policies for each unlabeled policy - the general idea is that if for a specific unlabeled policiy, the N most similar labeled documents are of a specific label (and above a certain similarity score), we could label it the same. We should try it out with a small sample of unlabeld policies first:

# %% [markdown]
# # TODO: TRY THIS

# %%

# %% [markdown]
# Let's try improving the model with upsampling first.

# %% [markdown]
# ## 6.2 Upsampling
# Before pseudo-labeling, let's try the more simple upsampling technique, and see if we can get a better model before scraping for more labels. I have (NOT YET) added a feature which allows all samples of a certain label ("good", in this case) to repeat `upsampling_factor` times in SampleGenerator iterations. Let's try refitting the model with the upsampled training corpus, and see if it scores better on the test set:
# # TODO: IMPLEMENT UPSAMPLING

# %%
raise RuntimeError("STOP HERE!")

# %% [markdown]
# # 6 Hyperparameter Search
# Perform search to find best set of hyperparametes

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, GridSearchCV
from scipy.stats import randint, uniform

# Create a larger parameter grid with more combinations
param_dist = {
    "epochs": [5, 25],
    "vector_size": [150, 300],
    "window": [5, 15],
}

N_SPLITS = 3
HRS_SEED = randint(0, 2**32).rvs()

# Update the hyperparameter search to use the pipeline
search = GridSearchCV(
    estimator=D2VClassifier(
        random_state=cp.seed,
        epochs=5,
    ),
    #     param_distributions=param_dist,
    param_grid=param_dist,
    #     n_candidates="exhaust",
    verbose=1,
    #     random_state=HRS_SEED,
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cp.seed),
    #     min_resources=15000,
    n_jobs=min(N_SPLITS, psutil.cpu_count(logical=False) - 1),
    refit=False,
)

# # Update the hyperparameter search to use the pipeline
# search = HalvingRandomSearchCV(
#     classifier=Doc2VecIsolationForestEstimator(
#         random_state=cp.seed,
#         onlne_learning=True,
#         epochs=5,
#         #         n_jobs=1,
#         #         workers=1,
#         metric="bal_acc",
#     ),
#     param_distributions=param_dist,
#     n_candidates="exhaust",
#     verbose=1,
#     random_state=cp.seed,
#     cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cp.seed),
#     min_resources=15000,
#     #     n_jobs = psutil.cpu_count(logical=False) - 1,
# )

# Fit the hyperparameter search on your training data
tic = time.perf_counter()
logging.info("Starting search...")
search.fit(train_set, train_set.labels)
logging.info(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# Print the best hyperparameters
logging.info(f"Best Hyperparameters: {search.best_params_}")

# display the entire CV results, too
display(pd.DataFrame(search.cv_results_))

# Beep when search is done
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# Refit the model using the best found hyperparameters, with multiprocessing
best_model = D2VClassifier(
    **search.best_params_,
    random_state=cp.seed,
    workers=psutil.cpu_count(logical=False) - 1,
)
best_model.fit(train_set, train_set.labels)

# Get the score of the best model on the test set
best_model_score = best_model.score(test_set, test_set.labels)
logging.info("Best Model Score:", best_model_score)

# Beep again when best model is ready
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# Trying to fit a fraction of the data for testing

# %%
from sklearn.model_selection import cross_validate

classifier = D2VClassifier(
    random_state=cp.seed,
    window=25,
    vector_size=200,
    epochs=5,
    train_score=True,
    workers=psutil.cpu_count(logical=False) - 1,
)

# CV = 4

# tic = time.perf_counter()
# logging.info("Starting CV...")
# scores = cross_validate(
#     classifier,
#     toy_train_set,
#     toy_train_set.labels,
#     cv=CV,
#     return_train_score=True,
#     verbose=1,
#     n_jobs=min(CV, psutil.cpu_count(logical=False) - 1),
# )
# logging.info(f"CV timing: {(time.perf_counter() - tic)/60:.1f} mins")
# print("np.nanmean(scores['test_score']): ", np.nanmean(scores["test_score"]))
# scores

# tic = time.perf_counter()
# logging.info("Fitting...")
# classifier.fit(toy_train_set, toy_train_set.labels)
# logging.info(f"Timing: {(time.perf_counter() - tic)/60:.1f} mins")
# classifier.score(toy_test_set, toy_test_set.labels)

# %%
# classifier.score(toy_test_set, toy_test_set.labels, threshold=0.492)

# %%
# BEST PARAMS SO FAR:

# {
#     'dm': 1,
#     'epochs': 17,
#     'hs': 1,
#     'min_count': 0,
#     'negative': 0.0,
#     'prob_threshold': 0.5,
#     'random_state': 42,
#     'sample': 0.0,
#     'train_score': False,
#     'vector_size': 682,
#     'window': 8,
# }

# %% [markdown]
# # 6. Attaching a Classifier
# It seems that the Doc2Vec model by itself is not doing a good job separating good privacy policies from bad ones. I can try using the vector embeddings as features for a binary classifier. Since I already have an classifier class for Doc2Vec, it should have been relatively easy to create a pipeline and attach more estimators. Unfortunately, since sklearn.pipeline.Pipeline doens't transform y (targets/labels) during fitting, I would have to implement a custom classifier combining Doc2Vec with an unsupervised classifier.

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
