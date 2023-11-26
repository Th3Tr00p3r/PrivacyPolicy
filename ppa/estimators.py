import logging
from collections import Counter
from typing import List, Tuple  # , Set

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator

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
from ppa.utils import timer

# from ppa.utils import timer


class D2VClassifier(BaseEstimator):
    """Doc."""

    def __init__(
        self,
        # General
        epochs: int,
        iterative_training=False,
        train_score: bool = False,
        random_state: int = None,
        threshold: float = 0.5,
        # Doc2Vec kwargs
        vector_size: int = 300,
        dm: int = 1,
        window: int = 5,
        min_count: int = 0,
        sample: float = 0.0,
        hs: int = 1,
        negative: float = 0.0,
        workers: int = 1,
    ):

        # Loop through the arguments and set attributes dynamically
        arguments = locals()
        del arguments["self"]  # Remove 'self' from the dictionary
        for key, value in arguments.items():
            setattr(self, key, value)

    #     @timer(1000)
    def fit(self, X, y, X_test=None, y_test=None):
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

        # iterative training
        if self.iterative_training:

            # get learning rate ranges for each epoch, linearly decaying
            alpha_edges = np.linspace(self.model.alpha, self.model.min_alpha, self.epochs + 1)
            alpha_ranges = np.array([alpha_edges[:-1], alpha_edges[1:]]).T

            for epoch in range(self.epochs):
                # train a Doc2Vec model on the entire, sparsely labeled dataset
                # set the alpha decay range according to the pre-defined ranges
                self.model.alpha, self.model.min_alpha = alpha_ranges[epoch]
                logging.info(f"[Estimator.fit] [epoch {epoch}] Training {self.get_params()}")
                logging.info(
                    f"[Estimator.fit] [epoch {epoch}] alpha: {self.model.alpha:.2e}, min_alpha: {self.model.min_alpha:.2e}"
                )
                self.model.train(X, total_examples=self.model.corpus_count, epochs=1)

                # calculate training score (optional)
                if self.train_score:
                    logging.info(
                        f"[Estimator.fit] [epoch {epoch}] Training score: {self.score(X, y, verbose=False, epochs=epoch+1, alpha=alpha_ranges[0][0], min_alpha=alpha_ranges[epoch][1])}"
                    )
                if X_test is not None:
                    logging.info(
                        f"[Estimator.fit] [epoch {epoch}] Validation score: {self.score(X_test, y_test, verbose=False, epochs=epoch+1, alpha=alpha_ranges[0][0], min_alpha=alpha_ranges[epoch][1])}"
                    )

            # return epochs, alpha and min_alpha to total range for inference
            self.model.alpha, self.model.min_alpha = alpha_ranges[0][0], alpha_ranges[-1][-1]
            self.model.epochs = self.epochs

        # single call to train with (possibly) multiple epochs
        else:
            # Train a Doc2Vec model on the entire, sparsely labeled dataset
            logging.info(f"[Estimator.fit] Training {self.get_params()}")
            self.model.train(X, total_examples=self.model.corpus_count, epochs=self.epochs)

            # calculate training score (optional)
            if self.train_score:
                logging.info(f"[Estimator.fit] Training score: {self.score(X, y, verbose=False)}")

        return self

    def transform(
        self,
        X: List[TaggedDocument],
        normalized=False,
        epochs=None,
        alpha=None,
        min_alpha=None,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        logging.info(
            f"[Estimator.transform] Inferring{' normalized ' if normalized else ' '}vector embeddings..."
        )
        if normalized:
            return np.array(
                [
                    (
                        vec := self.model.infer_vector(
                            td.words, epochs=epochs, alpha=alpha, min_alpha=min_alpha
                        )
                    )
                    / np.linalg.norm(vec)
                    for td in X
                ]
            )
        else:
            return np.array(
                [
                    self.model.infer_vector(
                        td.words, epochs=epochs, alpha=alpha, min_alpha=min_alpha
                    )
                    for td in X
                ]
            )

    def decision_function(self, X, **kwargs) -> np.ndarray:
        """Doc."""

        X_vec = self.transform(X, **kwargs)

        # Use similaities between mean good/bad train vectors and samples to compute scores
        good_sims = self.model.dv.cosine_similarities(self.model.dv["good"], X_vec)
        bad_sims = self.model.dv.cosine_similarities(self.model.dv["bad"], X_vec)
        # scale to [0, 1] range and return scores
        return ((good_sims - bad_sims) + 2) / 4

    #     @timer(1000)
    def predict(
        self, X, threshold: float = None, get_scores=False, **kwargs
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Doc."""

        threshold = threshold if threshold is not None else self.threshold
        scores = self.decision_function(X, **kwargs)
        y_pred = np.where(scores > threshold, -1, 1)
        if get_scores:
            return y_pred, scores
        else:
            return y_pred

    #     @timer(1000)
    def score(self, X: List[TaggedDocument], y: List[str], verbose=True, **kwargs):
        """Doc."""

        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
        y_true, labeled_idxs = self.valid_labels(y)

        if verbose:
            logging.info("[Estimator.score] Filtering labeled samples...")
        X_labeled = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]

        # predict and get scores
        y_pred, y_scores = self.predict(X_labeled, get_scores=True, **kwargs)

        # Compute balanced accuracy and the precision-recall curve
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # Calculate the AUC-PR
        auc_pr = auc(recall, precision)

        if verbose:
            # TESTESTEST
            RocCurveDisplay.from_predictions(y_true, y_scores, name="ROC-AUC")
            PrecisionRecallDisplay.from_predictions(y_true, y_scores, name="AUPRC")
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")
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

    def valid_labels(self, y) -> Tuple[np.ndarray, np.ndarray]:
        """Doc."""

        conv_dict = dict(unlabeled=np.nan, good=-1, bad=1)
        y_arr = np.array([conv_dict[label] for label in y])
        labeled_idxs = ~np.isnan(y_arr)
        return y_arr[labeled_idxs], labeled_idxs

    @timer(1000)
    def sanity_check(self, X_train, n_samples=1_000, plot=False):
        """Doc."""

        max_rank = 10
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
        sorted_rank_counts = dict(sorted(Counter(ranks).items()))
        sorted_rank_counts[f">{max_rank-1}"] = sorted_rank_counts.pop(max_rank + 1)

        if plot:
            with Plotter(
                suptitle="Sanity Check",
                xlabel="Similarity Rank",
                ylabel="Num. Documents",
            ) as ax:
                ax.bar([str(k) for k in sorted_rank_counts.keys()], sorted_rank_counts.values())

        return sorted_rank_counts


# class Doc2VecIsolationForestEstimator(BaseEstimator):
#    """Doc."""
#
#    implemented_metrics: Set[str] = {"auprc", "bal_acc"}
#
#    def __init__(
#        self,
#        # General
#        epochs: int,
#        onlne_learning: bool,
#        metric: str,
#        train_score: bool = False,
#        random_state: int = None,
#        # Doc2Vec kwargs
#        vector_size: int = 300,
#        dm: int = 1,
#        window: int = 5,
#        min_count: int = 0,
#        sample: float = 0.0,
#        hs: int = 1,
#        negative: float = 0.0,
#        workers: int = 1,
#        # IsolationForest kwargs
#        n_estimators: int = 300,
#        contamination: str | float = "auto",
#        max_samples: int | str = "auto",
#        max_features: int | float = 1.0,
#        bootstrap: bool = False,
#        verbose: int = 0,
#        warm_start: bool = True,
#        n_jobs: int = 1,
#    ):
#
#        if metric not in self.implemented_metrics:
#            raise ValueError(
#                f"Unknown metric '{self.metric}'. Must be in {self.implemented_metrics}"
#            )
#
#        # Get the local variables as a dictionary
#        arguments = locals()
#        del arguments["self"]  # Remove 'self' from the dictionary
#
#        # Loop through the arguments and set attributes dynamically
#        for key, value in arguments.items():
#            setattr(self, key, value)
#
#    #     @timer(1000)
#    def fit(self, X, y):
#        """Doc."""
#
#        # Initialize both models
#        self.vec_model = Doc2Vec(
#            vector_size=self.vector_size,
#            dm=self.dm,
#            window=self.window,
#            min_count=self.min_count,
#            sample=self.sample,
#            hs=self.hs,
#            negative=self.negative,
#            seed=self.random_state,
#            workers=self.workers,
#        )
#
#        self.clf_model = IsolationForest(
#            n_estimators=self.n_estimators,
#            contamination=self.contamination,
#            max_samples=self.max_samples,
#            max_features=self.max_features,
#            bootstrap=self.bootstrap,
#            random_state=self.random_state,
#            verbose=self.verbose,
#            warm_start=self.warm_start,
#            n_jobs=self.n_jobs,
#        )
#
#        # Count and display the amount of samples for each label
#        logging.info(f"[Estimator.fit] label_counter: {Counter(y)}")
#
#        # build vocabulary for Doc2Vec
#        self.vec_model.build_vocab(X)
#
#        # Training begins
#        if self.onlne_learning:
#            # get learning rate ranges for each epoch
#            alpha_edges = np.geomspace(
#                self.vec_model.alpha, self.vec_model.min_alpha, self.epochs + 1
#            )
#            self.alpha_ranges = np.array([alpha_edges[:-1], alpha_edges[1:]]).T
#
#            # get increasing number of IsolationForest estimators in logarithmic fashion fo each epoch, e.g. [ 500,  575,  643,  707,  766,  820,  871,  917,  960, 1000]
#            start_val = self.clf_model.n_estimators // 2
#            stop_val = self.clf_model.n_estimators
#            self.n_estimators_list = np.hstack(
#                (
#                    [start_val],
#                    np.cumsum(
#                        np.flip(
#                            np.diff(
#                                np.flip(np.geomspace(stop_val, start_val, self.epochs, dtype=int))
#                            )
#                        )
#                    )
#                    + start_val,
#                )
#            )
#
#            # iterative training
#            #             self.vec_loss = [] # NOTE Not implemented in Gensim
#            for epoch in range(self.epochs):
#                # train a Doc2Vec model on the entire, sparsely labeled dataset
#                # set the alpha decay range according to the pre-defined ranges
#                self.vec_model.alpha, self.vec_model.min_alpha = self.alpha_ranges[epoch]
#                logging.info(
#                    f"[Estimator.fit] [epoch {epoch}] Training {self.vec_model.lifecycle_events[0]['params']}"
#                )
#                #                 logging.info(f"[Estimator.fit] [epoch {epoch}] vec_model.alpha: {self.vec_model.alpha:.2e}, vec_model.alpha: {self.vec_model.min_alpha:.2e}")
#                self.vec_model.train(
#                    X, total_examples=self.vec_model.corpus_count, epochs=1, compute_loss=True
#                )
#                #                 self.vec_loss.append(self.vec_model.get_latest_training_loss()) # NOTE: Not implemented, returns 0.0
#                #                 logging.info(f"[Estimator.fit] [epoch {epoch}] vec_loss: {self.vec_loss[epoch]:.2e}") # NOTE: Not implemented in Gensim, returns 0.0
#
#                # transform the documents into vectors
#                logging.info(
#                    f"[Estimator.fit] [epoch {epoch}] Using model vectors as X_train_vec..."
#                )
#                X_train_vec = self.vec_model.dv.vectors
#
#                # Increase the number of trees in IsolationForest according to predetermined list of values
#                self.clf_model.set_params(n_estimators=self.n_estimators_list[epoch])
#                # Train the IsolationForest model using the all samples
#                logging.info(
#                    f"[Estimator.fit] Fitting IsolationForest model `{self.clf_model.get_params()}` to {X_train_vec.shape[0]:,} vector samples..."
#                )
#                self.clf_model.fit(X_train_vec)
#
#                # calculate training score (optional)
#                if self.train_score:
#                    logging.info(
#                        f"[Estimator.fit] [epoch {epoch}] Training score: {self.score(X, y, verbose=False)}"
#                    )
#
#        # Training vectorizer first, than classifier
#        else:
#            # train a Doc2Vec model on the entire, sparsely labeled dataset
#            logging.info(f"[Estimator.fit] Training {self.vec_model.lifecycle_events[0]['params']}")
#            self.vec_model.train(
#                X, total_examples=self.vec_model.corpus_count, epochs=self.epochs, compute_loss=True
#            )
#            #             self.vec_loss = [self.vec_model.get_latest_training_loss()]
#            #             logging.info(f"[Estimator.fit] vec_loss: {self.vec_loss[0]:.2e}")
#
#            # transform the documents into vectors
#            logging.info("[Estimator.fit] Using model vectors as X_train_vec...")
#            X_train_vec = self.vec_model.dv.vectors
#
#            # Train the IsolationForest model using the all samples
#            logging.info(
#                f"[Estimator.fit] Fitting IsolationForest model `{self.clf_model.get_params()}` to {X_train_vec.shape[0]:,} vector samples..."
#            )
#            self.clf_model.fit(X_train_vec)
#
#            # calculate training score (optional)
#            if self.train_score:
#                logging.info(f"[Estimator.fit] Training score: {self.score(X, y, verbose=False)}")
#
#        return self
#
#    #     @timer(1000)
#    def predict(self, X=None, X_vec=None, **kwargs):
#        """Doc."""
#
#        # transform X if X_vec not supplied
#        X_vec = X_vec if X_vec is not None else self.vec_transform(X, **kwargs)
#
#        # return classifier prediction
#        logging.info("[Estimator.predict] Predicting...")
#        return self.clf_model.predict(X_vec)
#
#    #     @timer(1000)
#    def score(self, X: List[TaggedDocument], y: List[str], verbose=True):
#        """Doc."""
#
#        # convet labeles to an array and keep only "good"/"bad" elements, and their indices
#        y_true, labeled_idxs = self.valid_labels(y)
#
#        if verbose:
#            logging.info("[Estimator.score] Filtering labeled samples...")
#        X_test = [X[idx] for idx in np.nonzero(labeled_idxs)[0]]
#
#        # transform X, y
#        X_test_vec = self.vec_transform(X_test)
#
#        # Calculate scores
#        if verbose:
#            logging.info("[Estimator.score] Calculating score...")
#        y_scores = self.clf_model.decision_function(X_test_vec)
#
#        # Compute the precision-recall curve
#        precision, recall, _ = precision_recall_curve(y_true, y_scores)
#
#        # Calculate the AUC-PR
#        auc_pr = auc(recall, precision)
#
#        # predict and calculate balanced accuracy
#        y_pred = self.predict(X_vec=X_test_vec)
#        bal_acc = balanced_accuracy_score(y_true, y_pred)
#
#        if verbose:
#            # TESTESTEST
#            RocCurveDisplay.from_predictions(y_true, y_scores, name="ROC-AUC")
#            PrecisionRecallDisplay.from_predictions(y_true, y_scores, name="AUPRC")
#            print(classification_report(y_true, y_pred))
#            # calculate individual accuracies
#            good_idxs = y_true == -1
#            bad_idxs = y_true == 1
#            good_accuracy = sum(y_pred[good_idxs] == y_true[good_idxs]) / y_true[good_idxs].size
#            bad_accuracy = sum(y_pred[bad_idxs] == y_true[bad_idxs]) / y_true[bad_idxs].size
#
#            logging.info(f"[Estimator.score] AUPRC: {auc_pr:.2f}")
#            logging.info(f"[Estimator.score] ACC: Good={good_accuracy:.2f}, Bad={bad_accuracy:.2f}")
#            logging.info(f"[Estimator.score] Balanced ACC: {bal_acc}")
#            # /TESTESTEST
#
#        if verbose:
#            logging.info(f"[Estimator.score] Returning '{self.metric}' score.")
#
#        if self.metric == "auprc":
#            return auc_pr
#        elif self.metric == "bal_acc":
#            return bal_acc
#
#    def valid_labels(self, y) -> Tuple[np.array, np.array]:
#        """Doc."""
#
#        conv_dict = dict(unlabeled=np.nan, good=-1, bad=1)
#        y_arr = np.array([conv_dict[label] for label in y])
#        labeled_idxs = ~np.isnan(y_arr)
#        return y_arr[labeled_idxs], labeled_idxs
#
#    def vec_transform(self, X: List[TaggedDocument], normalized=False):
#        """Doc."""
#
#        inference_params = {}
#        if self.onlne_learning:
#            inference_params = dict(
#                epochs=self.epochs,
#                alpha=self.alpha_ranges[0][0],  # initial learning rate
#                min_alpha=self.alpha_ranges[-1][-1],  # final learning rate
#            )
#        logging.info(f"[Estimator.vec_transform] inference_params: {inference_params}")
#
#        logging.info(
#            f"[Estimator.vec_transform] Inferring{' normalized ' if normalized else ' '}vector embeddings..."
#        )
#        if normalized:
#            return np.array(
#                [
#                    (vec := self.vec_model.infer_vector(td.words, **inference_params))
#                    / np.linalg.norm(vec)
#                    for td in X
#                ]
#            )
#        else:
#            return np.array([self.vec_model.infer_vector(td.words, **inference_params) for td in X])
