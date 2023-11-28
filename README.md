# Privacy Policy Analyzer (PPA)

The Privacy Policy Analyzer (PPA) is an ongoing project aimed at simplifying the comprehension and assessment of privacy policies. Its goal is to provide initial labels and potentially, later on, a more comprehensive scoring system for privacy policies. Although it's designed for user empowerment, its primary aim is to ease the understanding of complex privacy policies.

### Project Structure
- `data/`: Stores privacy policy-related data.
- `models/`: Contains trained models and embeddings generated from privacy policy documents.
- `notebooks/`: Includes the "Thinking Process.py (jupytext)" notebook, highlighting the iterative development.
- `display.py`: Manages display and visualization functions.
- `estimators.py`: Houses the `D2VClassifier` class integrating with scikit-learn and specialized text corpus retrieval for Doc2Vec.
- `privacy_policy_spider.py`: Module for web scraping privacy policy documents.
- `processing.py`: Manages various data processing tasks specific to privacy policy text.
- `utils.py`: Holds utility functions and helper methods.
- `__init__.py`: Python package initializer.

### Getting Started
1. **Requirements:** Check `requirements.txt` for necessary dependencies.
2. **Thinking Process Notebook:** Explore the "Thinking Process.py (jupytext)" Jupyter notebook for detailed insights into the project's development.
3. **Early Stages:** The project is a work in progress (WIP), open to suggestions and feedback.
4. **Contribution:** No formalized contribution process is established yet, but suggestions and ideas are highly appreciated.
5. **License:** Distributed under the MIT License.

### CorpusProcessor and SampleGenerator
The core classes, `CorpusProcessor` and `SampleGenerator` (alongside `IndexedFile`), handle the collection and processing of privacy policy documents. These classes facilitate data retrieval, transformation, and partitioning for further analysis and model training.

### D2VClassifier
The `D2VClassifier` class integrates with scikit-learn, enabling connectivity with hyperparameter tuning and pipeline usage. It utilizes a specialized text corpus retrieval class for Doc2Vec model training within the scikit-learn ecosystem.

Key Features:
- **Integration with scikit-learn:** Seamlessly integrates within scikit-learn pipelines for efficient data processing and model training workflows.
- **Text Corpus Retrieval:** Uses a specialized mechanism for retrieving and preparing text data, optimizing it for Doc2Vec model training.
- **Hyperparameter Tuning Support:** Provides functionalities to fine-tune Doc2Vec model parameters for improved performance.
- **Pipeline Connectivity:** Allows direct integration into scikit-learn pipelines, enabling smooth data transformation and model training sequences.

This class serves as a bridge between scikit-learn and Doc2Vec models, simplifying the incorporation of text-based data processing and analysis, specifically tailored for privacy policies.

### Thinking Process Overview
The project initiation involves:

#### 1. Data Collection and Processing
- Scraping privacy policies from various sources.
- Extracting, cleaning, tokenizing, and creating vector embeddings from these policies.

#### 2. Preliminary Exploratory Data Analysis (EDA)
- Understanding the distribution of policy lengths.
- Visualizing word frequencies using `gensim` corpora.

#### 3. Incorporating Labels
- Gathering tags and ratings from the ToS;DR database.
- Checking biases in labeled data and classifying policies as 'good' or 'bad.'

#### 4. Modeling with Doc2Vec
- Training a Doc2Vec model for document embeddings.
- Preparing data for training and testing sets in a stratified manner.

#### 5. Doc2Vec Model Evaluation
- Evaluating the model using various methods such as similarity comparisons and visualization.

#### 6. Upsampling and Pseudo-Labeling
- Attempting upsampling and pseudo-labeling techniques to balance the data and improve model accuracy.

The "Thinking Process.py (jupytext)" notebook provides a detailed narrative of the project's evolution, capturing key stages from data acquisition to model evaluation and improvement strategies.
