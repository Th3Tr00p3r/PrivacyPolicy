# Privacy Policy Analyzer (PPA)
[![Maintenance](https://img.shields.io/maintenance/yes/2023)](https://github.com/Th3Tr00p3r/PrivacyPolicy)
[![License](https://img.shields.io/badge/license-MIT-l)](/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)

## Overview
PPA is an ongoing project aimed at simplifying the complexities of privacy policies through Natural Language Processing (NLP) and Machine Learning techniques. The goal is to facilitate a quick assessment of privacy policies, aiding users in understanding the policies they often consent to without a thorough read. This project explores data extraction, text processing, and Doc2Vec-based classification to categorize policies as either 'good' or 'bad' in terms of their privacy content. This repository showcases various stages of the development process, from data acquisition to model evaluation.

## Thinking Process
The project begins with data gathering from the paper available at [Princeton's PrivacyPolicies](https://privacypolicies.cs.princeton.edu/), which provides a substantial database of privacy policies. These policies are gathered and processed, leveraging techniques in ETL (Extract, Transform, Load) to clean, tokenize, and create vector embeddings for analysis. Additionally, labels are acquired through scraping the ToS;DR API, enabling the classification of policies based on their quality.

## Features
- **CorpusProcessor**: Handles text processing, tokenization, and indexing for the Doc2Vec model.
- **SampleGenerator (IndexedFile)**: Assists in handling the training and testing data, ensuring balanced representation for model training.
- **D2VClassifier**: Integrates with scikit-learn for hyperparameter tuning and pipeline connectivity, utilizing specialized text corpus retrieval for Doc2Vec.
- **CLI Script**: Includes a CLI script named `ppa_cli.py` for running the trained model to classify content fetched from URLs. The script should be placed within the same directory as the `ppa` package.

## Directory Structure
- `/data`: Stores acquired privacy policy documents.
- `/models`: Contains trained models.
- `/notebooks`: Jupyter notebooks detailing the project's various stages.
- `/scripts`: Contains CLI scripts for utilizing the trained models.

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

The CLI script fetches the document text from the provided URL using [trafilatura](https://github.com/adbar/trafilatura), processes the document using a trained CorpusProcessor, and then classifies it using a loaded D2VClassifier model. The classification result (label) and score are printed to the terminal.

### License
This project is licensed under the MIT License. See [LICENSE](https://github.com/Th3Tr00p3r/PrivacyPolicy/blob/master/LICENSE) for more details.
