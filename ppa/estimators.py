import logging
from collections import Counter
from typing import List, Set, Tuple

import numpy as np
import psutil
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    balanced_accuracy_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# from ppa.utils import timer


class Doc2VecEstimator(BaseEstimator):
    """Doc."""

    def __init__(
        self,
        # General
        epochs: int,
        train_score: bool = False,
        random_state: int = None,
        prob_threshold: float = 0.5,
        # Doc2Vec kwargs
        vector_size: int = 300,
        dm: int = 1,
        window: int = 5,
        min_count: int = 0,
        sample: float = 0.0,
        hs: int = 1,
        negative: float = 0.0,
        workers: int = psutil.cpu_count(logical=False) - 1,
    ):

        # Get the local variables as a dictionary
        arguments = locals()
        del arguments["self"]  # Remove 'self' from the dictionary

        # Loop through the arguments and set attributes dynamically
        for key, value in arguments.items():
            setattr(self, key, value)

    #     @timer(1000)
    def fit(self, X, y):
        """Doc."""

        # Initialize both models
        self.model = Doc2Vec(
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

        # Count and display the amount of samples for each label
        logging.info(f"[Estimator.fit] label_counter: {Counter(y)}")

        # Build vocabulary for Doc2Vec
        self.model.build_vocab(X)

        # Train a Doc2Vec model on the entire, sparsely labeled dataset
        logging.info(f"[Estimator.fit] Training {self.model.lifecycle_events[0]['params']}")
        self.model.train(
            X, total_examples=self.model.corpus_count, epochs=self.epochs, compute_loss=True
        )

        # calculate training score (optional)
        if self.train_score:
            logging.info(f"[Estimator.fit] Training score: {self.score(X, y, verbose=False)}")

        return self

    def transform(self, X: List[TaggedDocument], normalized=False):
        """Doc."""

        logging.info(
            f"[Estimator.transform] Inferring{' normalized ' if normalized else ' '}vector embeddings..."
        )
        if normalized:
            return np.array(
                [(vec := self.model.infer_vector(td.words)) / np.linalg.norm(vec) for td in X]
            )
        else:
            return np.array([self.model.infer_vector(td.words) for td in X])

    #     @timer(1000)
    def predict(self, X=None, **kwargs):
        """Doc."""

        y_scores = self.score(X, **kwargs)
        return np.where(y_scores > self.prob_threshold, -1, 1)

    #     @timer(1000)
    def score(self, X: List[TaggedDocument], y: List[str], verbose=True):
        """Doc."""

        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
        y_true, labeled_idxs = self.valid_labels(y)

        if verbose:
            logging.info("[Estimator.score] Filtering labeled samples...")
        X_labeled = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]

        # transform X, y
        X_vec = self.transform(X_labeled)

        # Use similaities between mean good/bad train vectors and samples to compute scores
        good_sims = self.model.dv.cosine_similarities(self.model.dv["good"], X_vec)
        bad_sims = self.model.dv.cosine_similarities(self.model.dv["bad"], X_vec)
        # scale to [0, 1] range
        y_scores = ((good_sims - bad_sims) + 2) / 4

        # Compute the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # Calculate the AUC-PR
        auc_pr = auc(recall, precision)

        # predict and calculate balanced accuracy
        y_pred = np.where(y_scores > self.prob_threshold, -1, 1)
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

        return bal_acc

    def valid_labels(self, y) -> Tuple[np.array, np.array]:
        """Doc."""

        conv_dict = dict(unlabeled=np.nan, good=-1, bad=1)
        y_arr = np.array([conv_dict[label] for label in y])
        labeled_idxs = ~np.isnan(y_arr)
        return y_arr[labeled_idxs], labeled_idxs


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

    #     @timer(1000)
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
            logging.info("[Estimator.fit] Using model vectors as X_train_vec...")
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
        logging.info("[Estimator.predict] Predicting...")
        return self.clf_model.predict(X_vec)

    #     @timer(1000)
    def score(self, X: List[TaggedDocument], y: List[str], verbose=True):
        """Doc."""

        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
        y_true, labeled_idxs = self.valid_labels(y)

        if verbose:
            logging.info("[Estimator.score] Filtering labeled samples...")
        X_test = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]

        # transform X, y
        X_test_vec = self.vec_transform(X_test)

        #         # Calculate _vec_score
        #         vec_score = self._vec_score(X_test_vec, y_true)
        #         if verbose:
        #             logging.info(f"[Estimator.score] vec_score: {vec_score:.2f}")

        # Calculate scores
        if verbose:
            logging.info("[Estimator.score] Calculating score...")
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

    def _vec_score(self, X_test_vec, y_test):
        """Compute the AUC-PR score specifically for the Doc2Vec part of the model"""

        # Compute mean vectors from the training set
        mean_good_train_vector = self.vec_model.dv["good"]
        mean_bad_train_vector = self.vec_model.dv["bad"]

        # Use mean_good_train_vector and mean_bad_train_vector to compute scores
        good_sims = cosine_similarity(X_test_vec, mean_good_train_vector.reshape(1, -1))
        bad_sims = cosine_similarity(X_test_vec, mean_bad_train_vector.reshape(1, -1))
        similarity_diff = good_sims - bad_sims
        all_similarities = list(zip(similarity_diff.ravel(), np.nan_to_num(y_test)))

        # Extract scores for labeled and unlabeled vectors
        all_scores = np.array([sim[0] for sim in all_similarities]).reshape(-1, 1)
        logging.info(
            f"[Estimator.vec_score] raw all scores min/max: {(min(all_scores), max(all_scores))}"
        )

        # Fit MinMaxScaler for normalization using both labeled and unlabeled scores
        scaler = MinMaxScaler((-6, 6)).fit(all_scores)

        # Extract y_true and normalized_labeled_scores for labeled vectors
        labeled_logical_idxs = ~np.isnan(y_test)
        y_labeled_true = y_test[labeled_logical_idxs]
        labeled_scores = all_scores[labeled_logical_idxs.ravel()]
        logging.info(
            f"[Estimator.vec_score] raw labeled_scores min/max: {(min(labeled_scores), max(labeled_scores))}"
        )

        # Apply normalization to labeled scores
        normalized_labeled_scores = scaler.transform(labeled_scores)

        # Transform normalized scores using sigmoid function
        y_labeled_probs = expit(normalized_labeled_scores)
        logging.info(f"[Estimator.vec_score] y_labeled_probs[:10]: {y_labeled_probs[:10]}")

        # Compute AUC-PR using only normalized scores for labeled vectors
        precision, recall, _ = precision_recall_curve(y_labeled_true, y_labeled_probs)
        auc_pr = auc(recall, precision)
        return auc_pr
