# Privacy Policy Analyzer (PPA)
[![Maintenance](https://img.shields.io/maintenance/yes/2023)](https://github.com/Th3Tr00p3r/PrivacyPolicy)
[![License](https://img.shields.io/badge/license-MIT-l)](/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)

## Overview
PPA is an ongoing project aimed at simplifying the complexities of privacy policies through Natural Language Processing (NLP) and Machine Learning techniques. The goal is to facilitate a quick assessment of privacy policies, aiding users in understanding the policies they often consent to without a thorough read. This project explores data extraction, text processing, and Doc2Vec-based classification to categorize policies as either 'good' or 'bad' in terms of their privacy content. This repository showcases various stages of the development process, from data acquisition to model evaluation.

## Thinking Process
The project begins with data gathering from the paper available at [Princeton's PrivacyPolicies](https://privacypolicies.cs.princeton.edu/), which provides a substantial database of privacy policies. These policies are gathered and processed, leveraging techniques in ETL (Extract, Transform, Load) to clean, tokenize, and create vector embeddings for analysis. While most policies lack explicit labels, a small subset (approximately 1%) has secondary tags denoting their quality as 'good' or 'bad'.

The D2VClassifier employs an unsupervised learning approach using Doc2Vec, trained with secondary tags where available. For these labeled policies, it computes mean vectors for each tag ('good' and 'bad'). When presented with a new document, the model infers a vector representation and calculates its similarity to the mean vectors. The resulting score reflects the closeness of the document to the learned 'good' and 'bad' representations.

The classification decision is determined by thresholding this similarity score. If the similarity score exceeds a predefined threshold, the document is classified as 'good'; otherwise, it's classified as 'bad'. This methodology enables the model to infer the quality of policies based on their similarity to the limited labeled data available, allowing a binary classification output.

## Features
- **CorpusProcessor**: Handles text processing, tokenization, and indexing for the Doc2Vec model.
- **SampleGenerator (IndexedFile)**: Assists in handling the training and testing data, ensuring balanced representation for model training.
- **D2VClassifier**: Integrates with scikit-learn for hyperparameter tuning and pipeline connectivity, utilizing specialized text corpus retrieval for Doc2Vec.
- **CLI Script**: Includes a CLI script named `ppa_cli.py` for running the trained model to classify content fetched from URLs. The script should be placed within the same directory as the `ppa` package.

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
