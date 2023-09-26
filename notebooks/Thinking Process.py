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
# # TODOs:
# ### 1) CHECK THAT BLAS IS USED AND THAT C COMPILER WAS INSTALLED PRIOR TO GENSIM?

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
# ## 1.1 ETL (ELT)
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
# ### 1.1.2 Data Processing (Transforming)
#
# Now, since we are going to use some NLP tools later on, the few basic steps would be to clean, tokenize and remove stop-words from each policy. It would be smart to keep the data at this state, since at this point we are unsure about the requirements for furthur processing.

# %% [markdown]
#
# The cloned repository contains the texts in Markdown format, each in a seperate file in some alphabetically ordered subfolder. The company name/URL is contained in the filename. The following steps are taken:
#
# 1) All paths to PPs are kept in a generator
# 2) All PP texts are lazily transformed into (URL, processed PP) tuples

# %%
import random
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Generator, Iterable

import gensim
import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset (if not already downloaded)
nltk.download("stopwords")

# Get the list of english stopwords
STOP_WORDS = set(stopwords.words("english"))


@dataclass
class TagDocumentPairIterator:
    """Doc."""

    policy_paths: InitVar[Iterable[Path]]
    n: int = None
    seed: int = None

    def __post_init__(self, policy_paths):
        random.seed(self.seed)
        self.policy_paths = list(policy_paths)
        if self.n is None:
            self.n = len(self.policy_paths)
        self.policy_paths = random.sample(self.policy_paths, self.n)

    def __iter__(self):
        for fpath in self.policy_paths:
            yield (fpath.stem, self._preprocess_document(fpath))

    def _preprocess_document(self, fpath: Path):
        """Doc."""

        # read all but the header (probably not the most efficient method)
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.read().split("\n")
            doc = "\n".join(doc_lines)

        # tokenize the text
        simp_proc_doc = gensim.utils.simple_preprocess(doc)

        # remove stopwords and return
        return [word for word in simp_proc_doc if word not in STOP_WORDS]


# get all privacy policy markdown file paths in a generator
policy_path_gen = [fpath for fpath in REPO_PATH.rglob("*.md") if fpath.name != "README.md"]

print("Done.")

# %% [markdown]
# ### 1.1.3 Saving the transformed data (Loading)
#
# Since we already have the raw data locally, and since we do not expect more data to come in, we can avoid saving the data and just lazily transform it as needed. This means the loaded data becomes just the file paths.
#
# Let's create a function for sampling from our database:

# %%
N = 10
nl = "\n"
SEED = 17
print(
    f"sampling {N} companies/URLs:{nl}{nl.join([item[0] for item in TagDocumentPairIterator(policy_path_gen, N, SEED)])}"
)

# %% [markdown]
# # 2. Preliminary EDA
# Now that we have an easily accesible basic database, let's try exploring it.
#
# We can try using only a subset of the database for now (say 10K documents)

# %%
N = 10_000
# N = 10
company_urls, corpus = list(zip(*TagDocumentPairIterator(policy_path_gen, N, SEED)))

print(f"Loaded and transformed {N} privacy policies.")

# %% [markdown]
# Let's take a look at the distribution of PP lengths (number of tokens):

# %%
import matplotlib.pyplot as plt
import numpy as np

pp_lengths = np.array([len(tokens) for tokens in corpus])

_ = plt.hist(pp_lengths, 100)

print(f"PP length range: {pp_lengths.min()} - {pp_lengths.max():,} tokens")
print(f"median PP length: {np.median(pp_lengths):,} tokens")

# %% [markdown]
# It might prove wise, later on, to trim the ends of this distribution, as those very short or very long PPs might not represent the general case.
#
# Now, let's create a `gensim.corpora.Dictionary` from our corpus sample, so that we can visualize the data a bit.

# %%
from gensim.corpora import Dictionary

dct = Dictionary(corpus)

print(
    f"Dictionary was created by processing {dct.num_pos:,} tokens from a corpus of {dct.num_docs:,} documents."
)
print(f"It contains {len(dct):,} unique tokens.")
print(f"Each document, on average, contains {dct.num_nnz // dct.num_docs:,} unique tokens.")

# %% [markdown]
# First, let's visualize out the most frequent words in the entire corpus as a word-cloud:
#
# from typing import Dict
#
# import matplotlib.pyplot as plt

from typing import Dict

# %%
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
# We can try filtering out the least common and most common words from the Dictionary - words like "privacy", "information" or "data" are present in almost all PPs, and are probably not good differentiators between them.

# %%
from copy import deepcopy

filtered_dct = deepcopy(dct)
filtered_dct.filter_extremes()

print(f"After filtering extremens, Dictionary contains {len(filtered_dct)} unique tokens")

word_totalfreq_dict = {token: filtered_dct.cfs[id] for token, id in filtered_dct.token2id.items()}
word_docfreq_dict = {token: filtered_dct.dfs[id] for token, id in filtered_dct.token2id.items()}

# display the word frequencies as a word-cloud
print("Total Frequency Word-Cloud:")
display_wordcloud(word_totalfreq_dict)
print("Document Frequency Word-Cloud:")
display_wordcloud(word_docfreq_dict)

# %% [markdown]
# What immediately stands out is the difference between the "total frequency" and "per-document frequency" before and after filtering the most common words. In the "total frequency" picture, we are just seeing less common words (the most common ones being ignored). In the "per-document frequency" picture, this enables us to see past the noise.

# %% [markdown]
# # 3. Modeling
#
# Next, we want to transform our documents into some vector space. There are many techniques which could be used, but a well established one which captures in-document token relationships (important for semantic context) is Doc2Vec. Training a Doc2Vec model over our data will provide a vector embedding of each document in our corpus. This would facillitate several pathways:
# 1) Enabling cluster analysis and visualization
# 2) Similarity comparison between PPs
# 3) Inferring vector embedding for non-corpus policies
#
# TODO: It remains to be seen if this is feasible for the complete dataset on my maching

# %%
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


@dataclass
class TaggedDocumentIterator:

    url_doc_corpus: Iterable

    def __iter__(self):
        for url, doc in self.url_doc_corpus:
            # Tag each document with a unique identifier
            yield TaggedDocument(words=doc, tags=[url])

    def __len__(self):
        return sum(1 for _ in self)


# sample 10,000 PPs as a first attempt
N = 10_000
url_doc_corpus = TagDocumentPairIterator(policy_path_gen, N, SEED)

# Create tagged documents
train_data = TaggedDocumentIterator(url_doc_corpus)

# Initialize and train the Doc2Vec model
model = Doc2Vec()

# Build vocabulary
model.build_vocab(train_data)

# Train the model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the trained model for future use
PROJECT_ROOT = Path.cwd().parent
model.save(PROJECT_ROOT / "models" / "privacy_policy_doc2vec.model")

# %% [markdown]
# Sanity check - check that most documents are most similar to themselves

# %%
import collections

ranks = []
second_ranks = []
for tagged_doc in train_data:
    inferred_vector = model.infer_vector(tagged_doc.words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(tagged_doc.tags[0])
    ranks.append(rank)

    second_ranks.append(sims[1])

counter = collections.Counter(ranks)
print(counter)
