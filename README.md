---

title: Privacy Policy Analyzer
emoji: 👁️‍🗨️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false

---

# Privacy Policy Analyzer (PPA)

[![Maintenance](https://img.shields.io/maintenance/yes/2023)](https://github.com/Th3Tr00p3r/PrivacyPolicy)
[![License](https://img.shields.io/badge/license-MIT-l)](/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)
[![HuggingFace/Gradio](https://img.shields.io/badge/deployment-gradio-orange)](https://huggingface.co/spaces/molehillnest/PPA)

Have you ever read a privacy policy? In a world inundated with digital agreements and privacy statements, understanding the fine print often becomes a daunting task. PPA aims to be your ally in this realm, deciphering the intricacies of these policies and empowering you to make informed decisions.

Imagine a tool that swiftly dissects the language of privacy policies, unraveling their content to categorize them simply as 'good' or 'bad.' PPA does just that by harnessing the capabilities of data extraction, text processing, and a unique Doc2Vec-based classification system.



## Thinking Process

The project begins with data gathering from the paper available at [Princeton's PrivacyPolicies](https://privacypolicies.cs.princeton.edu/), which provides a substantial database of privacy policies. These policies are gathered and processed, leveraging techniques in ETL (Extract, Transform, Load) to clean, tokenize, and create vector embeddings for analysis. While most policies lack explicit labels, a small subset (approximately 1%) has secondary tags denoting their quality as 'good' or 'bad'.

The D2VClassifier employs a hybrid unsupervised-learning/supervised inference approach, based on Doc2Vec. First, the Doc2Vec model is trained on the entire corpus. For the training set policies which do posses labeles, it computes mean vectors for each class ('good' and 'bad'). When presented with a new document, the Doc2Vec model infers a vector representation, and the classifier calculates its similarity to the mean vectors. The resulting score reflects the closeness of the document to the learned 'good' and 'bad' representations.

The classification decision is determined by thresholding this similarity score. If the similarity score exceeds a predefined threshold, the document is classified as 'good'; otherwise, it's classified as 'bad'. This methodology enables the model to infer the quality of policies based on their similarity to the limited labeled data available, allowing a binary classification output.

In order to differentiate between privacy policies and other types of documents a few heuristics are employed, including distribution of common words, document length, and token filtering ratio against the training corpus dictionary.

## Features

- **CorpusProcessor**: Handles text processing, tokenization, and indexing for the Doc2Vec model.
- **SampleGenerator**: Assists in handling the training and testing data, ensuring balanced representation for model training while ensuring a one-at-a time presence of documents in RAM.
- **D2VClassifier**: Integrates with scikit-learn for hyperparameter tuning and pipeline connectivity, utilizing specialized text corpus retrieval for Doc2Vec.
- **IndexedFile**: Facillitates working with on-disk data, namely enabeling external shuffling via file start position indexing of data samples. Used with the above classes.
- **CLI Script**: Includes a CLI script named `ppa_cli.py` for running the trained model to classify content fetched from URLs. The script should be placed within the same directory as the `ppa` package.
- **Deployed App**: The model is deployed using [Gradio](https://www.gradio.app/) via the `app.py` file, offering a user-friendly interface for policy analysis.

## Usage

### Requirements

- See `requirements.txt` for necessary dependencies.
- Refer to the [Thinking Process.py](https://github.com/Th3Tr00p3r/PrivacyPolicy/blob/master/ppa/notebooks/Thinking%20Process.py) Jupyter notebook for a detailed walkthrough of the project's development stages.

### Running the Trained Model

The trained model can classify the content fetched from a provided URL using the CLI script. To use the script:

1. Place the `ppa_cli.py` script file within the same directory as the `ppa` package.
2. Ensure dependencies from `requirements.txt` are installed.
3. Run the CLI script by executing the following command:
   
   ```bash
    python ppa_cli.py <URL>
   ```
   
    Replace `<URL>` with the URL you want to classify.

The CLI script fetches the document text from the provided URL using trafilatura, processes the document using a trained CorpusProcessor, and then classifies it using a loaded D2VClassifier model. The classification result (label) and score are printed to the terminal.

## Future Work

The current project is continuously evolving, with several improvements and avenues for enhancement under consideration or in progress.

### Consideration of Downstream Supervised Classification

Exploration is underway to transition towards downstream supervised classification, contingent on acquiring a more extensive set of labeled data. A larger dataset may obviate the need for secondary labels while using Doc2Vec as a pure transformer model.

### Proposed Enhancements

#### Upsampling of 'Good'-Labeled Policies

One potential enhancement involves strategic up-sampling or mixing in copies of 'good'-labeled policies in the training data. The intent is to balance the dataset, aiding the model's comprehension of 'good' policies. Careful consideration is required to prevent overfitting while enhancing the model's generalization.

#### Pseudo-Labeling Unlabeled Training Data

Another avenue being explored is pseudo-labeling unlabeled training data with high/low thresholds for prediction. This strategy aims to bolster scoring robustness by averaging model summary vectors, particularly the 'good' ones, across a wider sample set. This could enable finer-tuning based on scores and potentially lay the groundwork for supervised classification in the future.

These enhancements and strategies are designed to refine the model's accuracy, robustness, and applicability. They particularly aim to address the challenge of representing 'good' and 'bad' policies beyond crude mean single-vector representations offered by Doc2Vec. The goal is to enable a more nuanced understanding of privacy policies by leveraging more complex classification models capable of capturing the intricate features embedded within the data.

### License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Th3Tr00p3r/PrivacyPolicy/blob/master/LICENSE) for more details.
