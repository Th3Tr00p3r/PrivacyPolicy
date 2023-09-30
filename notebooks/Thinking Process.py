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
# ### TODO: try removing PPs of extreme length (according to histogram) - see how it effects  clustering - it definitely affects training time!
# ### TODO: modify "generate_samples" method so that it creates a train/test split - or rather add such method to "TrainDataGenerator" (change name to SampleGenerator?)
# ### TODO: find out what the best metric would be for optimization

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

# # TESTEST - use only 1000 paths!
# print("\nWARNING! USING JUST 1000 PATHS!")
# policy_paths = policy_paths[:1000]

# %% [markdown]
# Define document processing class

# %%
import gzip
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Iterable
import time
import random
from winsound import Beep

import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import Dictionary

# define save/load path
PROJECT_ROOT = Path.cwd().parent
MODEL_DIR_PATH = PROJECT_ROOT / "models"

# TODO: I should separate between two cases for URL-like strings - companyname.com should be <COMPANYNAME>, www.companyname.com should be <URL>.
# Also, can I use the special token method of the Dictionary for them?
URL_PATTERN = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"


@dataclass
class TrainDataGenerator:
    """Doc."""

    save_dir_path: Path
    n_samples: int

    def __iter__(self):
        """Doc."""

        with gzip.open(self.save_dir_path / "train_data.pkl.gz", "rb") as input_file:
            try:
                for _ in range(self.n_samples):
                    # Deserialize and yield one document at a time
                    yield pickle.load(input_file)
            except EOFError:
                pass  # End of file


@dataclass
class CorpusProcessor:
    """Doc."""

    fpaths: List[Path]
    save_dir_path: Path
    should_filter_stopwords: bool = False
    url_pattern: str = URL_PATTERN
    email_pattern: str = EMAIL_PATTERN
    max_tokens: int = 100_000
    min_tokens: int = 0

    def __len__(self):
        return len(self.fpaths)

    def process(self, force=False):
        """Doc."""
        # TODO: implement saving/loading from LineSentence files? (performance boost promised?)

        if not force:
            # try loading existing dict
            dct = Dictionary.load(str(self.save_dir_path / "dictionary.pkl"))
            print("Loaded existing dictionary.")
            print("Using existing training data.")

        else:
            print(
                f"Processing and saving {(n_samples := len(self.fpaths)):,} TaggedDocument objects to disk: ",
                end="",
            )
            # initialize a Dictionary object
            dct = Dictionary()
            with gzip.open(self.save_dir_path / "train_data.pkl.gz", "wb") as output_file:
                # re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, fpath in enumerate(self.fpaths):
                    # track visually
                    if not (fidx + 1) % (n_samples // 100):
                        print("o", end="")
                    # open and process each file
                    tokenized_doc = self._preprocess_document(fpath)
                    # ignore very short/long documents
                    if self.min_tokens <= len(tokenized_doc) <= self.max_tokens:
                        # add to dictionary
                        dct.add_documents([tokenized_doc])
                        # create a TaggedDocument instance
                        tagged_doc = TaggedDocument(words=tokenized_doc, tags=[fpath.stem])
                        # Serialize the document tokens using pickle and write to the compressed file
                        pickle.dump(tagged_doc, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                print(" - Done.")

            print("Saving Dictionary... ", end="")
            dct.save(str(self.save_dir_path / "dictionary.pkl"))
            print("Done.")

        return dct

    def generate_samples(self, n_samples: int = None, randomized=False):
        """Doc."""

        # initialize re-generator
        sample_loader = TrainDataGenerator(self.save_dir_path, n_samples)

        if n_samples is not None:
            if randomized:
                return self._reservoir_sampling(sample_loader, n_samples)
            else:
                return sample_loader

    def _reservoir_sampling(self, iterable: Iterable, n_samples: int):
        """Doc."""

        sample = []
        for idx, item in enumerate(iterable):
            if idx < n_samples:
                sample.append(item)
            else:
                switch_idx = random.randint(0, idx)
                if switch_idx < n_samples:
                    sample[switch_idx] = item
        return sample

    def _preprocess_document(self, fpath: Path):
        """Doc."""

        # read all but the header (probably not the most efficient method)
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.read().split("\n")
            doc = "\n".join(doc_lines)

        # Replace URLs with "<URL>", email addresses with "<EMAILADD>"
        if self.url_pattern:
            doc = re.sub(self.url_pattern, "<URL>", doc)
        if self.email_pattern:
            doc = re.sub(self.email_pattern, "<EMAILADD>", doc)

        # tokenize the text
        simp_proc_doc = gensim.utils.simple_preprocess(doc)

        # remove stopwords and return
        if self.should_filter_stopwords:
            return gensim.parsing.preprocessing.remove_stopword_tokens(simp_proc_doc)
        else:
            return simp_proc_doc


# %% [markdown]
# Create a fresh CorpusProcessor instance, build a `gensim.corpora import Dictionary` and process the entire corpus, all while streaming to/from disk.

# %%
# SHOULD_REPROCESS = True
SHOULD_REPROCESS = False

# create a document processor with the paths
cp = CorpusProcessor(
    policy_paths, MODEL_DIR_PATH, should_filter_stopwords=True, min_tokens=40, max_tokens=5000
)

# build and save dictionary from all documents, process all documents and serialize
# the TaggedDocument objects to disk, compressed
tic = time.perf_counter()
dct = cp.process(force=SHOULD_REPROCESS)
print(f"Processing time: {(time.perf_counter() - tic)/60:.1f} mins")
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it. We can try using only a subset of the database for now (say 10K documents)
#
# Let's take a look at the distribution of PP lengths (number of tokens):

# %%
import matplotlib.pyplot as plt
import numpy as np

N = 10_000
pp_lengths = np.array([len(tagged_doc.words) for tagged_doc in cp.generate_samples(N)])

print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

_ = plt.hist(pp_lengths, int(np.sqrt(pp_lengths.size)))

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

# %% [markdown]
# It might prove wise to trim the ends of this distribution, as those very short or very long PPs might not represent the general case, and are definitely outliers in the dataset.

# %%
# # TODO: need to re-process everything for this - I should be able to do this without reprocessing e.g. load,
# # ignore short/long docs and remove their unique tokens from the Dictionary (filter_tokens), then re-save both Dictionary
# # and training data.

# # rebuild the database, with length limits
# cp = CorpusProcessor(policy_paths, MODEL_DIR_PATH, min_tokens= 40, max_tokens=5000)
# tic = time.perf_counter()
# dct = cp.process(force=True)
# print(f"Processing time: {(time.perf_counter() - tic)/60:.1f} mins")
# Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# pp_lengths = np.array([len(tagged_doc.words) for tagged_doc in cp.generate_samples(N)])

# print(f"Sampled corpus of {pp_lengths.size:,} privacy policies.")

# _ = plt.hist(pp_lengths, int(np.sqrt(pp_lengths.size)))

# print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
# print(f"median PP length: {np.median(pp_lengths):,.0f} tokens")

# %% [markdown]
# Now, let's take a look at the `gensim.corpora.Dictionary` we created from the entire corpus:

# %%
print(
    f"Dictionary was created by processing {dct.num_pos:,} tokens from a corpus of {dct.num_docs:,} documents."
)
print(f"It contains {len(dct):,} unique tokens.")
print(f"Each document, on average, contains {dct.num_nnz // dct.num_docs:,} unique tokens.")

# %% [markdown]
# We can use it to visualize the data a bit.
#
# First, let's visualize out the most frequent words in the entire corpus as a word-cloud:

# %%
from typing import Dict

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def display_wordcloud(word_freq_dict: Dict[str, int]):
    """Doc."""

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white")

    # Generate the word cloud from your word frequency dictionary
    wordcloud.generate_from_frequencies(word_freq_dict)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# create a {word: total_freq} dictionary using built-in methods of
word_totalfreq_dict = {token: dct.cfs[id] for token, id in dct.token2id.items()}

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
display_wordcloud(word_totalfreq_dict)

# %% [markdown]
# Now, let's also visualize the most frequent words document-wise:

# %%
# create a {word: doc_freq} dictionary using built-in methods of
word_docfreq_dict = {token: dct.dfs[id] for token, id in dct.token2id.items()}

# display the word frequencies as a word-cloud
print("Document Frequency Word-Cloud:")
display_wordcloud(word_docfreq_dict)

# %% [markdown]
# Notice how the images are quite similar.
#
# We can try filtering out the least common and most common words from the Dictionary - words like "privacy", "information" or "data" are present in almost all PPs, and are probably not good differentiators between them.

# %%
from copy import deepcopy

filtered_dct = deepcopy(dct)
filtered_dct.filter_extremes()

print(f"After filtering extremens, Dictionary contains {len(filtered_dct):,} unique tokens")

word_totalfreq_dict = {token: filtered_dct.cfs[id] for token, id in filtered_dct.token2id.items()}
word_docfreq_dict = {token: filtered_dct.dfs[id] for token, id in filtered_dct.token2id.items()}

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
display_wordcloud(word_totalfreq_dict)
print("Document Frequency Word-Cloud:")
display_wordcloud(word_docfreq_dict)

# %% [markdown]
# What immediately stands out is the difference between the "total frequency" and "per-document frequency" before and after filtering the most common words. In the "total frequency" picture, we are just seeing less common words (the most common ones being ignored). In the "per-document frequency" picture, this enables us to see past the noise.
#
# Look for the frequency of specific words (such as "url"):

# %%
# # The token you want to search for
# target_token = "url"

# # Iterate through each document and check if the token is in the document
# for doc in corpus.sample(10000):
#     if target_token in doc:
#         print(" ".join(doc))
#         break

# %% [markdown]
# # 3. Modeling
#
# Next, we want to transform our documents into some vector space. There are many techniques which could be used, but a well established one which captures in-document token relationships (important for semantic context) is Doc2Vec. Training a Doc2Vec model over our data will provide a vector embedding of each document in our corpus. This would facillitate several pathways:
# 1) Enabling cluster analysis and visualization
# 2) Similarity comparison between PPs
# 3) Inferring vector embedding for non-corpus policies

# %% [markdown]
# Training the (unsupervised) Doc2Vec model (or loading the last trained one)

# %%
import multiprocessing as mp
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# define save/load path
MODEL_PATH = PROJECT_ROOT / "models" / "privacy_policy_doc2vec.model"

SHOULD_RETRAIN = True
# SHOULD_RETRAIN = False

if not SHOULD_RETRAIN:
    # load the last trained model
    model = Doc2Vec.load(str(MODEL_PATH))

    # Done
    print("Model Loaded")
    # TODO: print some model details

else:

    # Sample tagged documents training data
    print("Preparing training data... ", end="")
    N = 10_000
    train_data = cp.generate_samples(N)
    print(f"Training data ready.")

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
# Sanity check - check that documents are most similar to themselves. I do not expect/desire to optimize the fraction of model-inferred documents which are most similar to themselves, as this might mean the model is overfitting. Instead, this is just a test to see that the model does something logical.
#
# It is worth mentioning that while purposfully attempting to overfit a small dataset of 1000 documents, I was only able to reach about 80%. I attribute this to noise in the training data.

# %%
import collections

# Set the number of top similar documents to consider
SAMPLE_SIZE = N // 10
TOP_N = 100

ranks = []
second_ranks = []
for idx, tagged_doc in enumerate(train_data):

    # Estimate percentage using first (random) `SAMPLE_SIZE` documents
    if idx + 1 == SAMPLE_SIZE:
        break

    # keep track
    if not (idx + 1) % (SAMPLE_SIZE // 10):
        print(f"{(idx+1)/(SAMPLE_SIZE):.1%}... ", end="")

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
# # Model Evaluation
#
# Doc2Vec is an unsupervised model, so finding a metric to evaluate it is not a straightforward task, given the fact that we also do not have any labeled data.
# Since I am still basically doing EDA, let's take a look at the test data in the learned vector embeddings, and see if any clusters emerge. My current short-term goal is to classify policies as "good" or "bad" (for the end-user, of course!), so I'm hoping to be able to see some clear boundries in the data.

# %%
from sklearn.decomposition import PCA

N_samples = 1000

# Prepare your test data as TaggedDocuments
# Assuming test_data is a list of tokenized documents
test_documents = train_data  # TODO: this should be the test data

# Infer document vectors for the test data
document_vectors = [
    model.infer_vector(doc.words) for idx, doc in enumerate(test_documents) if idx < N_samples
]

# Convert document vectors to a numpy array
document_vectors_array = np.array(document_vectors)

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(document_vectors_array)

# Create a scatter plot for visualization
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)

# Annotate the points with document labels (optional)
for idx, tagged_doc in enumerate(test_documents):
    if idx == N_samples:
        break

    # Annotate only a subset of points for clarity (adjust as needed)
    if idx % 10 == 0:  # Annotate every 10th point
        plt.annotate(tagged_doc.tags[0], (pca_result[idx, 0], pca_result[idx, 1]), fontsize=8)

plt.title("PCA Visualization of Doc2Vec Document Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# %%
Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# %% [markdown]
# Attempting to implement hyperparameter search

# %%
# Sample tagged documents training data
print("Preparing training data... ", end="")
N = 10_000
train_data = cp.generate_samples(N)
print(f"Training data ready.")

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from functools import partial
from sklearn.base import BaseEstimator
import numpy as np
import collections
from gensim.models import Doc2Vec
from scipy.stats import randint, uniform

# Create a larger parameter grid with more combinations
param_dist = {
    "vector_size": randint(50, 401),  # Random integer between 50 and 200
    "epochs": randint(10, 41),  # Random integer between 10 and 40
    "dm": [0, 1],  # Distributed Memory (PV-DM) vs. Distributed Bag of Words (PV-DBOW)
    "window": randint(3, 11),  # Random integer between 3 and 10 for the window size
    "min_count": randint(1, 11),  # Random integer between 1 and 10 for minimum word count
    "sample": uniform(1e-7, 1e-3),  # Random float between 0.0001 and 0.001 for downsampling
    "hs": [0, 1],
}


class Doc2VecEstimator(BaseEstimator):
    """Doc."""

    def __init__(self, vector_size, epochs, dm, window, min_count, sample, hs):
        self.vector_size = vector_size
        self.epochs = epochs
        self.dm = dm
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.hs = hs

    def fit(self, X, y=None):
        model = Doc2Vec(
            vector_size=self.vector_size,
            epochs=self.epochs,
            dm=self.dm,
            window=self.window,
            min_count=self.min_count,
            sample=self.sample,
            hs=self.hs,
        )
        model.build_vocab(train_data)
        model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model
        return self

    def predict(self, X):
        # This is a dummy predict method since it's not relevant for Doc2Vec models
        return None


# Define a scoring function for HalvingRandomSearchCV to maximize documents at rank 0
def custom_scorer(estimator, X, y):
    train_data = X  # Assuming X contains the train_data
    fraction_size = N // 10

    ranks = []
    for idx, tagged_doc in enumerate(train_data):
        # Estimate percentage using first (random) `SAMPLE_SIZE` documents
        if idx + 1 == fraction_size:
            break

        inferred_vec = estimator.model.infer_vector(tagged_doc.words)
        sims = estimator.model.dv.most_similar([inferred_vec], topn=TOP_N)
        try:
            rank = [docid for docid, sim in sims].index(tagged_doc.tags[0])
        except ValueError:
            rank = -1
        ranks.append(rank)
    counter = collections.Counter(ranks)
    return counter[0]  # Maximize the number of documents at rank 0


# Create a custom scorer function with a fixed model and training data
scorer = partial(custom_scorer, train_data=train_data)

# Create HalvingRandomSearchCV object with the custom estimator
halving_random_search = HalvingRandomSearchCV(
    estimator=Doc2VecEstimator(
        vector_size=100, epochs=10, dm=1, window=5, min_count=1, sample=1e-5, hs=0
    ),
    param_distributions=param_dist,
    n_candidates="exhaust",
    verbose=1,
    scoring=make_scorer(scorer, greater_is_better=False),
    random_state=42,
    cv=2,
)

# Fit the hyperparameter search on your training data
tic = time.perf_counter()
halving_random_search.fit(np.zeros(N), np.zeros(N))
print(f"Hyperparameter search timing: {(time.perf_counter() - tic)/60:.1f} mins")

# Get the best hyperparameters and model
best_params = halving_random_search.best_params_
best_model = halving_random_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
