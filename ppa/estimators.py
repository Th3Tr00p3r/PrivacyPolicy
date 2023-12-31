import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

# from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    balanced_accuracy_score,
    classification_report,
    precision_recall_curve,
)

from ppa.display import Plotter
from ppa.processing import CORPUS_FPATH, SampleGenerator
from ppa.utils import timer


class ScoredXGBClassifier(xgb.XGBClassifier):
    """
    Extends xgboost.XGBClassifier to provide scoring and evaluation metrics.
    """

    def fit(self, X, y, *args, **kwargs):
        """Doc."""
        # transform y to [0, 1]
        y_trans = np.array([1 if label == "good" else 0 for label in y])
        # continue with super
        return super().fit(X, y_trans, *args, **kwargs)

    def score(self, X: np.ndarray, y: List[str], plot=False, **kwargs):
        """
        Compute the balanced accuracy score and other evaluation metrics.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : List[str]
            Target labels.
        plot : bool, optional
            Flag to enable plotting, by default False.

        Returns
        -------
        float
            Balanced accuracy score.
        """

        # transform y to [0, 1]
        y_trans = np.array([1 if label == "good" else 0 for label in y])

        # predict and get scores
        y_pred = self.predict(X)

        #        # calculate individual accuracies # TODO: the "good_accuracy" is really just the recall
        #        good_idxs = y_trans == 1
        #        bad_idxs = y_trans == 0
        #        good_accuracy = sum(y_pred[good_idxs] == y_trans[good_idxs]) / y_trans[good_idxs].size
        #        bad_accuracy = sum(y_pred[bad_idxs] == y_trans[bad_idxs]) / y_trans[bad_idxs].size
        #        logging.info(
        #            f"[{self.__class__.__name__}.score] ACC: Good={good_accuracy:.2f}, Bad={bad_accuracy:.2f}."
        #        )

        # Compute balanced accuracy
        return balanced_accuracy_score(y_trans, y_pred)


class ScoredIsolationForest(IsolationForest):
    """Doc."""

    def score(self, X: np.ndarray, y: List[str], plot=False, **kwargs):
        """
        Compute the balanced accuracy score and other evaluation metrics.

        Parameters
        ----------
        X : SampleGenerator
            Input documents.
        y : List[str]
            Target labels.
        plot : bool, optional
            Flag to enable plotting, by default False.

        Returns
        -------
        float
            Balanced accuracy score.
        """

        # convet labeles to an array
        y_true = np.array([-1 if label == "good" else 1 for label in y])

        # predict and get scores
        y_pred, y_scores = self.predict(X), self.score_samples(X)

        # Compute balanced accuracy and the precision-recall curve
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # Calculate the AUC-PR
        auc_pr = auc(recall, precision)

        # calculate individual accuracies # TODO: the "good_accuracy" is really just the recall
        good_idxs = y_true == -1
        bad_idxs = y_true == 1
        good_accuracy = sum(y_pred[good_idxs] == y_true[good_idxs]) / y_true[good_idxs].size
        bad_accuracy = sum(y_pred[bad_idxs] == y_true[bad_idxs]) / y_true[bad_idxs].size
        logging.info(
            f"[{self.__class__.__name__}.score] ACC: Good={good_accuracy:.2f}, Bad={bad_accuracy:.2f}. AUPRC: {auc_pr:.2f}"
        )

        if plot:
            # TESTESTEST
            RocCurveDisplay.from_predictions(y_true, y_scores, name="ROC-AUC")
            PrecisionRecallDisplay.from_predictions(y_true, y_scores, name="AUPRC")
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")
            print(classification_report(y_true, y_pred))
            # /TESTESTEST

        return bal_acc


class D2VTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer using Doc2Vec model for vectorizing documents.
    Built to integrate with scikit-learn as well as using custom data-loading structures for Gensim's Doc2Vec model.
    """

    def __init__(
        self,
        # Doc2Vec kwargs
        vector_size: int = 300,
        dm: int = 1,
        window: int = 5,
        min_count: int = 5,
        sample: float = 1e-3,
        hs: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        # General
        epochs: int = 10,
        seed: int = None,
        workers: int = 1,
    ):
        """
        Initialize the Doc2Vec Transformer.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        seed : int, optional
            Random seed for reproducibility, by default None.
        vector_size : int, optional
            Dimensionality of the feature vectors, by default 300.
        dm : int, optional
            Model architecture; distributed memory (DM) vs distributed bag of words (DBOW), by default 1.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence, by default 5.
        min_count : int, optional
            Ignores all words with a total frequency lower than this, by default 5.
        sample : float, optional
            Threshold for configuring which higher-frequency words are randomly downsampled, by default 1e-3.
        hs : int, optional
            If 1, hierarchical softmax will be used for model training, by default 0.
        negative : int, optional
            If > 0, negative sampling will be used, by default 5.
        workers : int, optional
            Number of worker threads to train the model, by default 1.
        """

        # randomize seed if not suppled
        seed = seed or np.random.randint(0, 2**31)

        # Loop through the arguments and set attributes dynamically
        arguments = locals()
        del arguments["self"]  # Remove 'self' from the dictionary
        for key, value in arguments.items():
            setattr(self, key, value)

    @timer(1000)
    def fit(self, X: SampleGenerator | List[TaggedDocument], y=None):
        """
        Fit the Doc2Vec Transformer.

        Parameters
        ----------
        X : SampleGenerator
            Training samples.

        Returns
        -------
        D2VTransformer
            Instance of the fitted classifier.
        """

        # Initialize both models
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            dm=self.dm,
            window=self.window,
            min_count=self.min_count,
            sample=self.sample,
            hs=self.hs,
            negative=self.negative,
            seed=self.seed,
            workers=self.workers,
        )

        # Count and display the amount of samples for each label
        logging.info(f"[{self.__class__.__name__}.fit] label_counter: {Counter(y)}")

        # Build vocabulary for Doc2Vec
        logging.info(f"[{self.__class__.__name__}.fit] Building vocabulary...")
        self.model.build_vocab(X)

        # Train a Doc2Vec model on the entire, sparsely labeled dataset
        logging.info(f"[{self.__class__.__name__}.fit] Training {self.get_params()}")
        self.model.train(X, total_examples=self.model.corpus_count, epochs=self.epochs)

        return self

    @timer(2000)
    def transform(
        self,
        X: SampleGenerator | List[TaggedDocument],
        epochs=None,
        alpha=None,
        min_alpha=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Transform the input documents into vector embeddings.

        Parameters
        ----------
        X : SampleGenerator
            Input documents.
        normalized : bool, optional
            Flag to normalize the vectors, by default False.
        epochs : int, optional
            Number of epochs for inference, by default None.
        alpha : float, optional
            Initial learning rate, by default None.
        min_alpha : float, optional
            Final learning rate, by default None.

        Returns
        -------
        np.ndarray
            Vector embeddings of the input documents.
        """

        logging.info(
            f"[{self.__class__.__name__}.transform] Inferring vector embeddings for {len(X):,} documents..."
        )
        X_vec = np.empty((len(X), self.model.vector_size), dtype=float)
        for idx, td in enumerate(X):
            X_vec[idx] = self.model.infer_vector(
                td.words, epochs=epochs, alpha=alpha, min_alpha=min_alpha
            )

        return X_vec

    @timer(2000)
    def sanity_check(self, X_train: SampleGenerator, n_samples=1_000, max_rank=None, plot=False):
        """
        Perform a sanity check to test model general behaviour.
        If most inferred training documents are most similar to their model representation, the model appears to work as expected.

        Parameters
        ----------
        X_train : SampleGenerator
            Training samples.
        n_samples : int, optional
            Number of samples for the check, by default 1_000.
        max_rank : int, optional
            Maximum rank for similarity evaluation, by default None.
        plot : bool, optional
            Flag to enable plotting, by default False.

        Returns
        -------
        Dict
            Sorted rank counts.
        """

        max_rank = max_rank or len(self.model.dv)
        ranks = []
        for idx, tagged_doc in enumerate(X_train.sample(n_samples)):

            # keep track
            if not (idx + 1) % (n_samples // 10):
                logging.info(f"{(idx+1)/(n_samples):.0%}... ")

            # Calculate similarities only for the TOP_N similar documents for the current inferred vector
            inferred_vec = self.model.infer_vector(tagged_doc.words)
            sims = self.model.dv.most_similar([inferred_vec], topn=max_rank)

            # Find the rank of the tag in the top N
            try:
                rank = [docid for docid, sim in sims].index(tagged_doc.tags[0])
            except ValueError:
                # Handle the case where the tag is not found in sims (worse than 10th most similar)
                rank = max_rank + 1
            ranks.append(rank)

        # count the similarity rank
        sorted_rank_counts: Dict[Any, int] = dict(sorted(Counter(ranks).items()))
        sorted_rank_counts[f">{max_rank-1}"] = sorted_rank_counts.pop(max_rank + 1)

        if plot:
            with Plotter(
                suptitle="Sanity Check",
                xlabel="Similarity Rank",
                ylabel="Num. Documents",
            ) as ax:
                ax.bar([str(k) for k in sorted_rank_counts.keys()], sorted_rank_counts.values())

        return sorted_rank_counts

    def save_model(self, filepath: Path) -> None:
        """
        Save the trained Doc2Vec model.

        Parameters
        ----------
        filepath : str
            filepath in which to save the object.
        """

        # saving the model
        self.model.save(str(filepath))
        logging.info(
            f"[{self.__class__.__name__}.save_model] Doc2Vec model successfully saved to {filepath}."
        )

    @classmethod
    def load_model(cls, filepath: Path):
        """Doc."""

        self = cls()
        self.model = Doc2Vec.load(str(filepath))
        logging.info(
            f"[{self.__class__.__name__}.load_model] Doc2Vec model successfully loaded from {filepath}."
        )
        return self

    def generate_train_test_sets(self, corpus_fpath=CORPUS_FPATH, **kwargs):
        """Doc."""

        return SampleGenerator(
            corpus_fpath, positive_keys=self.model.dv.index_to_key, **kwargs
        ), SampleGenerator(corpus_fpath, negative_keys=self.model.dv.index_to_key, **kwargs)


class D2VClassifier(D2VTransformer):
    """
    Classifier using Doc2Vec model for vectorizing and predicting document categories.
    Built to integrate with scikit-learn as well as using custom data-loading structures for Gensim's Doc2Vec model.
    """

    def __init__(
        self,
        # Doc2Vec kwargs
        vector_size: int = 300,
        dm: int = 1,
        window: int = 5,
        min_count: int = 5,
        sample: float = 1e-3,
        hs: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        # General
        epochs: int = 10,
        train_score: bool = False,
        seed: int = None,
        threshold: float = 0.505,
        workers: int = 1,
    ):
        """
        Initialize the Doc2Vec Classifier.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        train_score : bool, optional
            Flag to compute training score, by default False.
        seed : int, optional
            Random seed for reproducibility, by default None.
        threshold : float, optional
            Threshold for decision making, by default 0.5.
        vector_size : int, optional
            Dimensionality of the feature vectors, by default 300.
        dm : int, optional
            Model architecture; distributed memory (DM) vs distributed bag of words (DBOW), by default 1.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence, by default 5.
        min_count : int, optional
            Ignores all words with a total frequency lower than this, by default 5.
        sample : float, optional
            Threshold for configuring which higher-frequency words are randomly downsampled, by default 1e-3.
        hs : int, optional
            If 1, hierarchical softmax will be used for model training, by default 0.
        negative : int, optional
            If > 0, negative sampling will be used, by default 5.
        workers : int, optional
            Number of worker threads to train the model, by default 1.
        """

        # Loop through the arguments and set attributes dynamically
        kwargs = locals()
        # remove the unneeded standard locals
        del kwargs["self"]  # Remove 'self' from the dictionary
        del kwargs["__class__"]  # Remove 'class' from the dictionary
        for key, value in kwargs.items():
            setattr(self, key, value)

        # remove nonexistant parmaters for transformer
        del kwargs["train_score"]  # Remove 'class' from the dictionary
        del kwargs["threshold"]  # Remove 'class' from the dictionary
        # initialize the transformer base class with the same arguments
        super().__init__(**kwargs)

        # initialize empty _label2keys (obtained through property methods)
        self._label2keys: Dict[str, List[str]] = {}

    @property
    def label2keys(self):
        """Doc."""

        # regenerate if still empty
        if not self._label2keys:
            X, _ = self.generate_train_test_sets()
            for key, label in zip(X.keys, X.labels):
                try:
                    self._label2keys[label].append(key)
                except KeyError:
                    self._label2keys[label] = [key]
        return self._label2keys

    @timer(1000)
    def fit(self, X, y: List[str]):
        """
        Fit the Doc2Vec Transformer, then build good/bad mean inferred vectors for model.
        Optionally, score the training.

        Parameters
        ----------
        X : SampleGenerator
            Training samples.
        y : array-like
            Target labels.

        Returns
        -------
        D2VClassifier
            Instance of the fitted classifier.
        """

        # train the model using the parent transformer class
        super().fit(X, y, should_time=False)

        # calculate training score (optional)
        if self.train_score:
            logging.info(f"[{self.__class__.__name__}.fit] Calculating training score...")
            logging.info(
                f"[{self.__class__.__name__}.fit] Training score (Balanced ACC.): {self.score(X, y, verbose=False)}"
            )

        return self

    def decision_function(
        self, X: SampleGenerator | List[TaggedDocument] = None, X_vec: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        """
        Compute decision function scores based on similarity between vectors.

        Parameters
        ----------
        X : SampleGenerator, optional
            Input documents, by default None.
        X_vec : np.ndarray, optional
            Vector embeddings, by default None.

        Returns
        -------
        np.ndarray
            Decision function scores.
        """

        # Transform X if X_vec is not supplied (default)
        if X_vec is None:
            X_vec = self.transform(X, **kwargs)

        # get the mean vectors from the model using the collected keys
        mean_good = self.mean_labeled_model_vector("good")
        mean_bad = self.mean_labeled_model_vector("bad")

        # Use similaities between mean good/bad train vectors and samples to compute scores
        good_sims = self.model.dv.cosine_similarities(mean_good, X_vec)
        bad_sims = self.model.dv.cosine_similarities(mean_bad, X_vec)
        # scale to [0, 1] range and return scores
        return ((good_sims - bad_sims) + 2) / 4

    def predict(
        self,
        X: SampleGenerator | List[TaggedDocument],
        threshold: float = None,
        get_scores=False,
        **kwargs,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels based on decision function scores.

        Parameters
        ----------
        X : SampleGenerator
            Input documents.
        threshold : float, optional
            Threshold for decision making, by default None.
        get_scores : bool, optional
            Flag to return scores along with predictions, by default False.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predicted labels or labels with scores.
        """

        threshold = threshold if threshold is not None else self.threshold
        scores = self.decision_function(X, **kwargs)
        y_pred = np.where(scores > threshold, -1, 1)
        if get_scores:
            return y_pred, scores
        else:
            return y_pred

    def score(self, X: SampleGenerator | List[TaggedDocument], y: List[str], plot=False, **kwargs):
        """
        Compute the balanced accuracy score and other evaluation metrics.

        Parameters
        ----------
        X : SampleGenerator
            Input documents.
        y : List[str]
            Target labels.
        plot : bool, optional
            Flag to enable plotting, by default False.

        Returns
        -------
        float
            Balanced accuracy score.
        """

        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
        y_true, labeled_idxs = self.valid_labels(y)

        # scoring is only possible on labeled samples
        try:
            X_labeled: SampleGenerator | List[TaggedDocument] = cast(SampleGenerator, X).sample(
                idxs=np.nonzero(labeled_idxs)[0]
            )
        except AttributeError:
            # X is a list
            X_labeled = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]

        # predict and get scores
        y_pred, y_scores = self.predict(X_labeled, get_scores=True, **kwargs)

        # Compute balanced accuracy and the precision-recall curve
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # Calculate the AUC-PR
        auc_pr = auc(recall, precision)

        # calculate individual accuracies # TODO: the "good_accuracy" is really just the recall
        good_idxs = y_true == -1
        bad_idxs = y_true == 1
        good_accuracy = sum(y_pred[good_idxs] == y_true[good_idxs]) / y_true[good_idxs].size
        bad_accuracy = sum(y_pred[bad_idxs] == y_true[bad_idxs]) / y_true[bad_idxs].size
        logging.info(
            f"[{self.__class__.__name__}.score] ACC: Good={good_accuracy:.2f}, Bad={bad_accuracy:.2f}. AUPRC: {auc_pr:.2f}"
        )

        if plot:
            # TESTESTEST
            RocCurveDisplay.from_predictions(y_true, y_scores, name="ROC-AUC")
            PrecisionRecallDisplay.from_predictions(y_true, y_scores, name="AUPRC")
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")
            print(classification_report(y_true, y_pred))
            # /TESTESTEST

        return bal_acc

    def mean_labeled_model_vector(self, label: str):
        """Doc."""

        return np.array([self.model.dv[k] for k in self.label2keys[label]]).mean(axis=0)

    def valid_labels(self, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter and convert labels to valid numerical format.

        Parameters
        ----------
        y : array-like
            Target labels.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing valid labels and indices.
        """

        conv_dict = {"unlabeled": np.nan, "good": -1, "bad": 1}
        y_arr = np.array([conv_dict[label] for label in y])
        labeled_idxs = ~np.isnan(y_arr)
        return y_arr[labeled_idxs], labeled_idxs
