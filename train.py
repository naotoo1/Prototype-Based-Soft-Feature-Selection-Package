"""Implementation of Soft Prototype Feature Selection  Algorithm"""

import argparse
from collections import Counter
from typing import (
    Dict,
    Any,
    Union,
    List, 
    Tuple
)
import logging
import os
import random
from dataclasses import (
    dataclass,
    field
)
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import numpy.linalg as ln
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn_lvq import (
    MrslvqModel,
    LmrslvqModel
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt

from dataset1 import DATA
from mutate_labels import MutatedValidation
from mutated_validation import (
    MutatedValidationScore,
    TrainRun,
    EvaluationMetricsType,
)
from proto_initializer import (
    get_Kmeans_prototypes,
    Initializer,
)


class Verbose(int, Enum):
    YES = 0
    NO = 1


class SavedModelUpdate(str, Enum):
    TRUE = "update"
    FALSE = "keine-update"


class ValidationType(str, Enum):
    MUTATEDVALIDATION = "mv"
    HOLDOUT = "ho"


class LVQ(str, Enum):
    MRSLVQ = "mrslvq"
    LMRSLVQ = "lmrslvq"

@dataclass
class LocalRejectStrategy:
    significant: List[int]
    insignificant: List[int]
    significant_hit: List[int]
    insignificant_hit: List[int]
    tentative: Union[List[int], None]


@dataclass
class SelectedRelevances:
    significant: Union[List[int], List[str], np.ndarray]
    insignificant: Union[List[int], List[str], np.ndarray]


@dataclass
class HitsInfo:
    features: List[int]
    hits: List[int]


@dataclass
class SelectedRelevancesExtra:
    significant: HitsInfo
    insignificant: HitsInfo


@dataclass
class BestLearnedResults:
    omega_matrix: List[np.ndarray]
    evaluation_metric_score: List[float]
    num_prototypes: int


@dataclass
class GlobalRelevanceFactorsSummary:
    omega_matrix: np.ndarray
    lambda_matrix: np.ndarray
    lambda_diagonal: np.ndarray
    lambda_row_sum: np.ndarray
    feature_relevance_dict: Dict[str, Any]
    weight_significance: np.ndarray


@dataclass
class LocalRelevanceFactorSummary:
    omega_matrix: List[np.ndarray]
    lambda_matrix: List[np.ndarray]
    lambda_diagonal: List[np.ndarray]
    lambda_row_sum: np.ndarray
    feature_relevance_dict: Dict[str, Any]
    weight_significance: np.ndarray
    feature_significance: np.ndarray


@dataclass
class GlobalFeatureSelection:
    relevance: GlobalRelevanceFactorsSummary
    eval_score: List[float]
    num_prototypes: int


@dataclass
class LocalFeatureSelection:
    relevance: LocalRelevanceFactorSummary
    eval_score: List[float]
    num_prototypes: int


@dataclass
class TrainModelSummary:
    selected_model_evaluation_metrics_scores: List[float]
    final_omega_matrix: List[np.ndarray]
    final_prototypes: List[np.ndarray]


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


Path("./evaluation").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler(
        filename="evaluation/report.txt",
        encoding='utf-8', 
        mode="w"
        )],
    level=logging.INFO,
)


def evaluation_metric_logs(
        evaluation_metric_scores: List[float],
        model_name: str,
        validation: str,
        log: bool = True,
) -> None:
    report = [{validation: evaluation_metric_scores}]
    if log:
        return logging.info("%s:%s", model_name, report)


def train_hold_out(
        input_data: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        latent_dim: int,
        init_prototypes:Union[np.ndarray,None],
        num_prototypes: int = 1,
        regularization: float = 0.0001,
        sigma: int = 1,
        max_iter: int = 1000,
        gtol: float = 0.00001,
        display: bool = False,
        classwise: bool = False,
        random_state: Union[int, None] = None,
        evaluation_metric: str = EvaluationMetricsType.ACCURACY.value,
) -> TrainModelSummary:
    prototypes, omega_matrix = [], []
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, labels, test_size=0.3, random_state=4
    )

    model = matrix_rslvq(
        model=model_name,
        latent_dim=latent_dim,
        init_prototypes=init_prototypes,
        num_prototypes=num_prototypes,
        regularization=regularization,
        sigma=sigma,
        max_iter=max_iter,
        gtol=gtol,
        display=display,
        random_state=random_state,
        classwise=classwise,
    )

    X_train, y_train = np.array(X_train), np.array(y_train)
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)
    model.fit(X_train, y_train)
    prototypes.append(model.w_)
    if model_name == LVQ.MRSLVQ:
        omega_matrix.append(model.omega_)  # type: ignore
    else:
        omega_matrix.append(model.omegas_)  # type: ignore

    outputs = model.predict(X_test)

    return TrainModelSummary(
        selected_model_evaluation_metrics_scores=[
            accuracy_score(y_test, outputs)  # type: ignore
        ],
        final_omega_matrix=omega_matrix,
        final_prototypes=prototypes,
    )


def train_model_by_mv(
        input_data: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        latent_dim: int,
        regularization: float = 0.05,
        sigma: int = 1,
        max_iter: int = 1000,
        gtol: float = 0.00001,
        display: bool = False,
        classwise: bool = False,
        random_state: Union[int, None] = None,
        init_prototypes:Union[np.ndarray,None]=None,
        num_prototypes: int = 1,
        perturbation_distribution: str = "balanced",
        perturbation_ratio: float = 0.2,
        evaluation_metric: str = EvaluationMetricsType.ACCURACY.value,
) -> TrainModelSummary:
    model = matrix_rslvq(
        model=model_name,
        latent_dim=latent_dim,
        init_prototypes=init_prototypes,
        num_prototypes=num_prototypes,
        regularization=regularization,
        sigma=sigma,
        max_iter=max_iter,
        gtol=gtol,
        display=display,
        random_state=random_state,
        classwise=classwise,
    )

    model_mv = clone(model)

    mutated_validation = MutatedValidation(
        labels=labels.astype(np.int64),
        perturbation_ratio=perturbation_ratio,
        perturbation_distribution=perturbation_distribution
    )

    mutate_list = mutated_validation.get_mutated_label_list  # use this list instead
    results, mv_score, prototypes, omega_matrix = [], [], [], []
    for train_runs in range(2):
        if train_runs == TrainRun.ORIGINAL:
            model.fit(input_data, labels)
            prototypes.append(model.w_)
            if model_name == LVQ.MRSLVQ:
                omega_matrix.append(model.omega_)  # type: ignore
            else:
                omega_matrix.append(model.omegas_)  # type: ignore
            results.append(model.predict(input_data))

        if train_runs == TrainRun.MUTATED:
            model_mv.fit(input_data, mutate_list)
            if model_name == LVQ.MRSLVQ:
                omega_matrix.append(model_mv.omega_)  # type: ignore
            else:
                omega_matrix.append(model_mv.omegas_)  # type: ignore
            results.append(model_mv.predict(input_data))

            mv_scorer = MutatedValidationScore(
                mutated_labels=mutated_validation,
                mutate=mutate_list,
                original_training_predicted_labels=results[0],
                mutated_training_predicted_labels=results[1],
                evaluation_metric=evaluation_metric,
            )
            mv_score.append(mv_scorer.get_mv_score)

    return TrainModelSummary(
        selected_model_evaluation_metrics_scores=mv_score,
        final_omega_matrix=omega_matrix,
        final_prototypes=prototypes,
    )


def matrix_rslvq(
        model: str,
        latent_dim: int,
        init_prototypes:Union[np.ndarray,None],
        num_prototypes: int,
        regularization: float,
        sigma: int,
        max_iter: int,
        gtol: float,
        display:bool,
        random_state: Union[int, None],
        classwise: bool = False,
) -> Union[MrslvqModel, LmrslvqModel]:

    if model == LVQ.MRSLVQ:
        return MrslvqModel(
            initial_prototypes=init_prototypes,
            prototypes_per_class=num_prototypes,
            regularization=regularization,
            initialdim=latent_dim,
            sigma=sigma,
            max_iter=max_iter,
            gtol=gtol,
            display=display,
            random_state=random_state,
        )
    # else:
    if model == LVQ.LMRSLVQ:
        return LmrslvqModel(
            prototypes_per_class=num_prototypes,
            regularization=regularization,
            initialdim=None,
            sigma=sigma,
            max_iter=max_iter,
            gtol=gtol,
            display=display,
            random_state=random_state,
            classwise=classwise,
        )
    # raise RuntimeError(
    #     "specified_lvq: none of the models did match",
    # )


@dataclass
class TM:
    input_data: np.ndarray
    labels: np.ndarray
    model_name: str
    latent_dim: int
    num_classes: int
    init_prototypes:Union[np.ndarray,None]
    num_prototypes: int
    # optimal_search: str
    feature_list: Union[List[str], None] = None
    eval_type: Union[str, None] = None
    regularization: float = 0
    max_epochs: int = 10
    save_model: bool = False
    significance: bool = True
    perturbation_distribution: str = "balanced"
    perturbation_ratio: float = 0.2
    evaluation_metric: str = EvaluationMetricsType.ACCURACY.value
    epsilon: Union[float, None] = 0.0001
    norm_ord: str = "fro"
    termination: str = "metric"
    patience: int = 1
    verbose: int = 0
    summary_metric_list: list = field(default_factory=lambda: [])

    def train_ho(self, increment: int) -> TrainModelSummary:
        return train_hold_out(
            input_data=self.input_data,
            labels=self.labels,
            model_name=self.model_name,
            latent_dim=self.latent_dim,
            init_prototypes=self.init_prototypes,
            num_prototypes=self.num_prototypes + increment,
            # save_model=self.save_model,
            evaluation_metric=self.evaluation_metric,
            max_iter=self.max_epochs,
        )

    def train_mv(self, increment: int) -> TrainModelSummary:
        return train_model_by_mv(
            input_data=self.input_data,
            labels=self.labels,
            model_name=self.model_name,
            latent_dim=self.latent_dim,
            num_prototypes=self.num_prototypes + increment,
            # save_model=self.save_model,
            evaluation_metric=self.evaluation_metric,
            perturbation_distribution=self.perturbation_distribution,
            perturbation_ratio=self.perturbation_ratio,
            max_iter=self.max_epochs,
        )

    @property
    def final(self) -> BestLearnedResults:  # type: ignore
        (metric_list, matrix_list, should_continue, counter) = ([], [], True, -1)
        while should_continue:
            counter += 1
            train_eval_scheme = (
                self.train_mv(increment=counter)
                if self.eval_type == ValidationType.MUTATEDVALIDATION.value
                else self.train_ho(increment=counter)
            )
            validation_score = (
                train_eval_scheme.selected_model_evaluation_metrics_scores
            )
            omega_matrix = train_eval_scheme.final_omega_matrix
            num_prototypes = (
                    len((train_eval_scheme.final_prototypes[0])) // self.num_classes
            )
            metric_list.append(validation_score[0])
            matrix_list.append(omega_matrix[0])
            if counter < self.patience:
                continue
            condition = counter == self.max_epochs
            stability = get_stability(
                metric1=metric_list[-2],
                metric2=metric_list[-1],
                matrix1=matrix_list[-2],
                matrix2=matrix_list[-1],
                convergence=self.termination,
                epsilon=self.epsilon,
                learner=self.model_name,
                matrix_ord=self.norm_ord,
            )
            if (
                    condition is False
                    and self.termination == "metric"
                    and stability is True
            ):
                should_continue = False
                return BestLearnedResults(
                    omega_matrix=omega_matrix,
                    evaluation_metric_score=validation_score,
                    num_prototypes=num_prototypes,
                )
            if (
                    condition is False
                    and self.termination == "matrix"
                    and stability is True
            ):
                should_continue = False
                return BestLearnedResults(
                    omega_matrix=omega_matrix,
                    evaluation_metric_score=validation_score,
                    num_prototypes=num_prototypes,
                )
            if (
                    condition is True
                    and self.termination == "metric"
                    and stability is True
            ):
                should_continue = False
                return BestLearnedResults(
                    omega_matrix=omega_matrix,
                    evaluation_metric_score=validation_score,
                    num_prototypes=num_prototypes,
                )
            if (
                    condition is True
                    and self.termination == "matrix"
                    and stability is True
            ):
                should_continue = False
                return BestLearnedResults(
                    omega_matrix=omega_matrix,
                    evaluation_metric_score=validation_score,
                    num_prototypes=num_prototypes,
                )

    @property
    def feature_selection(self) -> Union[GlobalFeatureSelection, LocalFeatureSelection]:
        if (
                self.model_name == LVQ.MRSLVQ
                and self.eval_type == ValidationType.MUTATEDVALIDATION.value
        ):
            train_eval_scheme = self.final
            validation_score = train_eval_scheme.evaluation_metric_score
            omega_matrix = train_eval_scheme.omega_matrix
            relevance = get_lambda_matrix(
                omega_matrix=omega_matrix, feature_list=self.feature_list
            )
            return GlobalFeatureSelection(
                relevance=relevance,
                eval_score=validation_score,
                num_prototypes=train_eval_scheme.num_prototypes,
            )
        if (
                self.model_name == LVQ.MRSLVQ
                and self.eval_type == ValidationType.HOLDOUT.value
        ):
            train_eval_scheme = self.final
            validation_score = train_eval_scheme.evaluation_metric_score
            omega_matrix = train_eval_scheme.omega_matrix
            relevance = get_lambda_matrix(
                omega_matrix=omega_matrix, feature_list=self.feature_list
            )
            return GlobalFeatureSelection(
                relevance=relevance,
                eval_score=validation_score,
                num_prototypes=train_eval_scheme.num_prototypes,
            )

        if (
                self.model_name == LVQ.LMRSLVQ
                and self.eval_type == ValidationType.MUTATEDVALIDATION.value
        ):
            train_eval_scheme = self.final
            validation_score = train_eval_scheme.evaluation_metric_score
            omega_matrix = train_eval_scheme.omega_matrix
            num_prototypes = train_eval_scheme.num_prototypes
            relevance = get_local_lambda_matrix(
                omega_matrix=omega_matrix,
                feature_list=self.feature_list,
                num_prototypes=num_prototypes,
                num_classes=self.num_classes,
            )
            return LocalFeatureSelection(
                relevance=relevance,
                eval_score=validation_score,
                num_prototypes=num_prototypes,
            )

        if (
                self.model_name == LVQ.LMRSLVQ
                and self.eval_type == ValidationType.HOLDOUT.value
        ):
            train_eval_scheme = self.final
            validation_score = train_eval_scheme.evaluation_metric_score
            omega_matrix = train_eval_scheme.omega_matrix
            num_prototypes = train_eval_scheme.num_prototypes
            relevance = get_local_lambda_matrix(
                omega_matrix=omega_matrix,
                feature_list=self.feature_list,
                num_classes=self.num_classes,
                num_prototypes=num_prototypes,
            )
            return LocalFeatureSelection(
                relevance=relevance,
                eval_score=validation_score,
                num_prototypes=num_prototypes,
            )

    @property
    def summary_results(self):
        feature_selection = self.feature_selection
        if self.model_name == LVQ.LMRSLVQ and self.significance is True:
            summary = get_relevance_summary(
                feature_significance=feature_selection.relevance.feature_significance,  # type: ignore
                evaluation_metric_score=feature_selection.eval_score[0],
                verbose=self.verbose,
            )
            return SelectedRelevances(
                significant=summary.significant,
                insignificant=summary.insignificant,
            )
        if self.model_name == LVQ.LMRSLVQ and self.significance is False:
            summary = get_relevance_elimination_summary(
                weight_significance=feature_selection.relevance.lambda_row_sum,
                num_protypes_per_class=self.feature_selection.num_prototypes,
                lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                evaluation_metric_score=feature_selection.eval_score[0],
                verbose=self.verbose,
                input_dim=self.latent_dim,
                num_classes=self.num_classes,
            )

            visualize(
                features=summary.significant.features,
                hits=summary.significant.hits,
                significance=True,
                eval_score=feature_selection.eval_score[0],
            )
            visualize(
                features=summary.insignificant.features,
                hits=summary.insignificant.hits,
                significance=False,
                eval_score=feature_selection.eval_score[0],
            )
            return SelectedRelevancesExtra(
                significant=summary.significant,
                insignificant=summary.insignificant,
            )
        if self.model_name == LVQ.MRSLVQ and self.significance is True:
            summary = get_relevance_global_summary(
                lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                weight_significance=feature_selection.relevance.weight_significance,
                significance=self.significance,
                evaluation_metric_score=feature_selection.eval_score[0],
                verbose=self.verbose,
            )
            return SelectedRelevances(
                significant=np.array(summary.significant).flatten(),
                insignificant=np.array(summary.insignificant).flatten(),
            )
        if self.model_name == LVQ.MRSLVQ and self.significance is False:
            summary = get_relevance_global_summary(
                lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                weight_significance=feature_selection.relevance.weight_significance,
                significance=self.significance,
                evaluation_metric_score=feature_selection.eval_score[0],
                verbose=self.verbose,
            )
            return SelectedRelevances(
                significant=np.array(summary.significant).flatten(),
                insignificant=np.array(summary.insignificant).flatten(),
            )
        raise RuntimeError("summary_results: none of the above cases match")


def get_lambda_matrix(
        omega_matrix: List[np.ndarray], feature_list: Union[List[str], None] = None
) -> GlobalRelevanceFactorsSummary:
    omega = omega_matrix[0]
    lambda_matrix = omega.T @ omega
    list_of_features = np.arange(len(lambda_matrix.shape))
    lambda_diagonal = np.diagonal(lambda_matrix)
    row_sum_squared_omega_ij = np.sum(lambda_matrix, axis=1)
    attributes = feature_list if feature_list is not None else list_of_features

    feature_relevance_elimination_dict = dict(zip(attributes, row_sum_squared_omega_ij))
    weight_features_significance = np.argsort(lambda_diagonal)[::-1]

    return GlobalRelevanceFactorsSummary(
        omega_matrix=omega,
        lambda_matrix=lambda_matrix,
        lambda_diagonal=lambda_diagonal,
        lambda_row_sum=row_sum_squared_omega_ij,
        feature_relevance_dict=feature_relevance_elimination_dict,
        weight_significance=weight_features_significance,
    )


def get_local_lambda_matrix(
        omega_matrix: List[np.ndarray],
        num_classes: int,
        num_prototypes: int,
        feature_list: Union[List[str], None] = None,
) -> LocalRelevanceFactorSummary:
    omega = omega_matrix[0]
    (
        omega_matrix,
        lambda_matrix,
        lambda_diagonal,
        lambda_row_sum,
        feature_relevance_dict,
        weight_significance,
    ) = ([], [], [], [], [], [])
    for local_matrix in omega:
        relevance_factor_summary = get_lambda_matrix(
            omega_matrix=[local_matrix], feature_list=feature_list
        )
        omega_matrix.append(relevance_factor_summary.omega_matrix)
        lambda_matrix.append(relevance_factor_summary.lambda_matrix)
        lambda_diagonal.append(relevance_factor_summary.lambda_diagonal)
        lambda_row_sum.append(relevance_factor_summary.lambda_row_sum)
        feature_relevance_dict.append(relevance_factor_summary.feature_relevance_dict)
        weight_significance.append(relevance_factor_summary.weight_significance)
    feature_signficance = np.array(
        [f"f{weight[0]}" for _index, weight in enumerate(weight_significance)]
    ).reshape(num_classes, num_prototypes)
    class_labels = [
        f"class_label_{label}_relevances_{i}"
        for label in np.arange(len(omega))
        for i in range(num_classes)
    ]
    feature_relevance_dict = dict(zip(class_labels, feature_relevance_dict))
    return LocalRelevanceFactorSummary(
        omega_matrix=omega_matrix,
        lambda_matrix=lambda_matrix,
        lambda_diagonal=lambda_diagonal,
        lambda_row_sum=lambda_row_sum,  # type: ignore
        feature_relevance_dict=feature_relevance_dict,
        weight_significance=np.array(weight_significance),
        feature_significance=feature_signficance,
    )


def get_relevance_global_summary(
        lambda_row_sum: np.ndarray,
        weight_significance: np.ndarray,
        evaluation_metric_score: float,
        significance: bool,
        verbose: int = 0,
) -> SelectedRelevances:
    significant, insignificant = [], []
    select_vector = weight_significance if significance is True else lambda_row_sum
    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )
    if significance:
        for index, class_relevance in enumerate(select_vector):
            summary = get_global_logs(
                index=index,
                class_relevance=class_relevance,
                verbose=verbose,
                state="pass",
            )
            significant.append(summary.significant)
    if significance is False:
        for feature_index, feature_label in enumerate(
                np.argsort(lambda_row_sum)[::-1], start=1
        ):
            cond = float(lambda_row_sum[feature_label]) > 0
            if cond is True:
                summary = get_global_logs(
                    index=feature_index,
                    class_relevance=feature_label,
                    verbose=verbose,
                    state="pass",
                )
                significant.append(summary.significant)
            if cond is False:
                summary = get_global_logs(
                    index=feature_index,
                    class_relevance=feature_label,
                    verbose=verbose,
                    state="fail",
                )
                insignificant.append(summary.insignificant)
    return SelectedRelevances(
        significant=significant,  # type: ignore
        insignificant=insignificant,  # type: ignore
    )


def get_relevance_summary(
        feature_significance: np.ndarray,
        evaluation_metric_score: float,
        verbose: int = 0,
) -> SelectedRelevances:
    significant_features = []

    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )
    for index, class_relevance in enumerate(feature_significance):
        for feature_index, feature_label in enumerate(class_relevance):
            if verbose == Verbose.YES:
                print(
                    "Passes the test: ",
                    f"class {index}",
                    " - Ranking: ",
                    f"prototype {feature_index}.",
                    feature_label,
                    "✔️",
                )
                significant_features.append(feature_label)
            if verbose == Verbose.NO:
                logging.info(
                    "%s %s %s %s %s %s",
                    "Passes the test: ",
                    f"class {index}",
                    " - Ranking: ",
                    f"prototype {feature_index}.",
                    feature_label,
                    "✔️",
                )
                significant_features.append(feature_label)

    significant_features = [
        int(feature.replace("f", "")) for feature in significant_features
    ]
    return SelectedRelevances(
        significant=significant_features,
        insignificant=[],
    )


def get_eval_score(
        evaluation_metric_score: float,
        verbose: int = 0,
):
    if verbose == Verbose.YES:
        print(
            f"Evaluation Score:{evaluation_metric_score:.2f}",
        )
    if verbose == Verbose.NO:
        logging.info(
            "%s %s",
            "Evaluation Score: ",
            evaluation_metric_score,
        )

def get_global_logs(
        index: int,
        class_relevance: int,
        verbose: int = 0,
        state: str = "pass",
) -> SelectedRelevances:
    significant_features, insignificant = [], []
    verbosity = verbose == Verbose.YES
    status = state == "pass"
    if (
            verbosity is True
            and status is True
    ):
        print(
            "Passes the test: ",
            index,
            " - Ranking: ",
            f"f{class_relevance}",
            "✔️",
        ) 
        significant_features.append(class_relevance)
    if (
            verbosity is False
            and status is True
    ):
        logging.info(
            "%s %s %s %s %s",
            "Passes the test: ",
            index,
            " - Ranking: ",
            f"f{class_relevance}",
            "✔️",
        )
        significant_features.append(class_relevance)
    if (
            verbosity is True
            and status is False
    ):
        print(
            "Fails the test: ",
            index,
            " - Ranking: ",
            f"f{class_relevance}",
            "❌",
        )
        insignificant.append(class_relevance)
    if (
            verbosity is False
            and status is False
    ):
        logging.info(
            "%s %s %s %s %s",
            "Fails the test: ",
            index,
            " - Ranking: ",
            f"f{class_relevance}",
            "❌",
        )
        insignificant.append(class_relevance)
    return SelectedRelevances(
        significant=significant_features,
        insignificant=insignificant,
    )


def get_relevance_logs(
        label: int,
        index: int,
        feature_label: str,
        verbose: int = 0,
        state: str = "pass",
) -> SelectedRelevances:
    significant_features, insignificant = (
        [],
        [],
    )
    verbosity = verbose == Verbose.YES
    status = state == "pass"
    if (
            verbosity is True
            and status is True
    ):
        print(
            "Passes the test: ",
            f"class {label}",
            " - Ranking: ",
            f"prototype {index}.",
            f"f{feature_label}",
            "✔️",
        )
        significant_features.append(feature_label)
    if (
            verbosity is False
            and status is True
    ):
        logging.info(
            "%s %s %s %s %s %s",
            "Passes the test: ",
            f"class {label}",
            " - Ranking: ",
            f"prototype {index}.",
            f"f{feature_label}",
            "✔️",
        )
        significant_features.append(feature_label)
    if (
            verbosity is True
            and status is False
    ):
        print(
            "Fails the test: ",
            f"class {label}",
            " - Ranking: ",
            f"prototype {index}.",
            f"f{feature_label}",
            "❌",
        )
        insignificant.append(feature_label)
    if (
            verbosity is False
            and status is False
    ):
        logging.info(
            "%s %s %s %s %s %s",
            "Fails the test: ",
            f"class {label}",
            " - Ranking: ",
            f"prototype {index}.",
            f"f{feature_label}",
            "❌",
        )
        insignificant.append(feature_label)
    return SelectedRelevances(
        significant=significant_features,
        insignificant=insignificant,
    )


def get_relevance_elimination_summary(
        weight_significance: np.ndarray,
        num_protypes_per_class: int,
        num_classes: int,
        input_dim: int,
        lambda_row_sum: np.ndarray,
        evaluation_metric_score: float,
        verbose: int = 0,
) -> SelectedRelevancesExtra:
    significant, insignificant = [], []
    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )

    num_prototypes = len(weight_significance) // num_classes
    print(num_prototypes)

    weight_significance = np.reshape(
        weight_significance, (num_classes, num_prototypes, input_dim)
    )
    for class_label, weight_summary in enumerate(weight_significance):
        for index, class_relevance in enumerate(weight_summary):
            for _feature_index, feature_label in enumerate(
                    np.argsort(class_relevance)[::-1]
            ):
                classwise_relevance = list(lambda_row_sum[index])
                cond_1 = float(classwise_relevance[feature_label]) > 0
                if cond_1 is True:
                    report = get_relevance_logs(
                        label=class_label,
                        index=index,
                        feature_label=feature_label,
                        verbose=verbose,
                        state="pass",
                    )
                    significant.append(report.significant)
                if cond_1 is False:
                    report = get_relevance_logs(
                        label=class_label,
                        index=index,
                        feature_label=feature_label,
                        verbose=verbose,
                        state="fail",
                    )
                    insignificant.append(report.insignificant)

    return SelectedRelevancesExtra(
        significant=get_hits_significance(np.array(significant).flatten()),
        insignificant=get_hits_significance(np.array(insignificant).flatten()),
    )


def get_hits_significance(summary_list: np.ndarray) -> HitsInfo:
    summary_dict = Counter(summary_list)  # type: ignore
    sorted_feature_keys = sorted(
        summary_dict, key=lambda k: (summary_dict[k], k), reverse=True
    )

    hits = [summary_dict[key] for key in sorted_feature_keys]

    return HitsInfo(
        features=sorted_feature_keys,  # type: ignore
        hits=hits,
    )


def get_matrix_stability(
        matrix1, matrix2, epsilon: Union[float, None], matrix_ord: str
) -> bool:
    distance = (ln.norm((matrix2 - matrix1), ord=matrix_ord)).numpy()
    if epsilon is not None:
        return bool(0 < distance <= epsilon)
    return bool(distance == 0)


def get_metric_stability(
        metric1: float,
        metric2: float,
        epsilon: Union[float, None],
) -> bool:
    metric1, metric2 = np.round(metric1, 2), np.round(metric2, 2)
    difference = metric2 - metric1
    if epsilon is not None:
        return bool(0 <= difference <= epsilon)
    return bool(metric2 == metric1)


def get_stability(
        metric1: float,
        metric2: float,
        matrix1: np.ndarray,
        matrix2: np.ndarray,
        convergence: str,
        epsilon: Union[float, None],
        learner: str,
        matrix_ord: str,
) -> bool:
    if (
            convergence == "metric"
            and learner == LVQ.MRSLVQ
    ):
        return get_metric_stability(
            metric1=metric1,
            metric2=metric2,
            epsilon=epsilon,
        )
    if (
            convergence == "matrix"
            and learner == LVQ.MRSLVQ
    ):
        return get_matrix_stability(
            matrix1=matrix1,
            matrix2=matrix2,
            epsilon=epsilon,
            matrix_ord=matrix_ord,
        )
    if (
            convergence == "metric"
            and learner == LVQ.LMRSLVQ
    ):
        return get_metric_stability(
            metric1=metric1,
            metric2=metric2,
            epsilon=epsilon,
        )
    if (
            convergence == "matrix"
            and learner == LVQ.LMRSLVQ
    ):
        raise RuntimeError(
            "get_stability: computational cost may be very high: consider metric case"
        )
    # raise RuntimeError(
    #     "get_stability: none of the cases match",
    # )


def visualize(
        features: list,
        hits: list,
        significance: bool,
        eval_score: Union[float, None],
):
    relevance = "significant " if significance is True else "insignificant"
    _fig, ax = plt.subplots()
    ax.bar(features, hits)
    ax.set_xticks(features)
    ax.set_ylabel("number of hits per prototype")
    ax.set_xlabel(f"{relevance} features")
    # ax.set_title(f"Feature relevance summary ({np.round(eval_score,4)})")
    ax.set_title("Feature relevance summary")
    plt.savefig(f"evaluation/feature_{relevance}_rank_plot.png")


def reject_strategy(
        significant: list,
        insignificant: list,
        significant_hit: list,
        insignificant_hit: list,
) -> LocalRejectStrategy:
    intersection = list(set(significant) & set(insignificant))

    index_sig_conflicts = [
        index for index, value in enumerate(significant) if value in intersection
    ]

    value_sig_conflicts = [
        value for index, value in enumerate(significant) if value in intersection
    ]

    index_insig_conflicts = [
        index for index, value in enumerate(insignificant) if value in intersection
    ]

    value_insig_conflicts = [
        value for index, value in enumerate(insignificant) if value in intersection
    ]

    index_list, new_val_insig = [], []
    for val_sig in value_sig_conflicts:
        for index_value, value in enumerate(value_insig_conflicts):
            if val_sig == value:
                new_val_insig.append(value)
                index_list.append(index_insig_conflicts[index_value])

    significant_hits = [significant_hit[index] for index in index_sig_conflicts]

    insignificant_hits = [insignificant_hit[index] for index in index_list]

    rejection_strategy_significant = [
        value_sig_conflicts[hit_index]
        for hit_index, hit in enumerate(significant_hits)
        if hit <= insignificant_hits[hit_index]
    ]

    rejection_strategy_insignificant = [
        new_val_insig[hit_index]
        for hit_index, hit in enumerate(insignificant_hits)
        if hit < significant_hits[hit_index]
    ]

    tentative_strategy = [
        value_sig_conflicts[hit_index]
        for hit_index, hit in enumerate(significant_hits)
        if hit == insignificant_hits[hit_index]
    ]

    new_significant = [
        feature
        for feature in significant
        if feature not in rejection_strategy_significant
    ]

    new_insignificant = [
        feature
        for feature in insignificant
        if feature not in rejection_strategy_insignificant
    ]

    new_significant_hit = [
        significant_hit[index]
        for index, value in enumerate(significant)
        if value in new_significant
    ]

    new_insignificant_hit = [
        insignificant_hit[index]
        for index, value in enumerate(insignificant)
        if value in new_insignificant
    ]

    significant = significant if len(new_significant) == 0 else new_significant

    insignificant = insignificant if len(new_insignificant) == 0 else new_insignificant

    significant_hit = (
        significant_hit if len(new_significant) == 0 else new_significant_hit
    )

    insignificant_hit = (
        insignificant_hit if len(new_insignificant) == 0 else new_insignificant_hit
    )

    insignificant = [
        feature
        for _index, feature in enumerate(insignificant)
        if feature not in significant
    ]

    insignificant_hit = [
        insignificant_hit[index]
        for index, feature in enumerate(insignificant)
        if feature not in significant
    ]

    return LocalRejectStrategy(
        significant=significant,
        insignificant=insignificant,
        significant_hit=significant_hit,
        insignificant_hit=insignificant_hit,
        tentative=tentative_strategy,
    )


def reject(
        significant: list,
        insignificant: list,
        significant_hit: list,
        insignificant_hit: list,
        reject_options: bool,
) -> LocalRejectStrategy:
    if reject_options is True:
        strategy = reject_strategy(
            significant,
            insignificant,
            significant_hit,
            insignificant_hit,
        )
        return LocalRejectStrategy(
            significant=strategy.significant,
            insignificant=strategy.insignificant,
            significant_hit=strategy.significant_hit,
            insignificant_hit=strategy.insignificant_hit,
            tentative=strategy.tentative,
        )
    if reject_options is False:
        return LocalRejectStrategy(
            significant, insignificant, significant_hit, insignificant_hit, None
        )
    # raise RuntimeError(
    #     "reject:none of the above matches",
    # )


def get_rejection_summary(
        significant: list,
        insignificant: list,
        significant_hit: list,
        insignificant_hit: list,
        reject_options: bool,
        vis: bool,
) -> LocalRejectStrategy:
    rejected_summary = reject(
        significant,
        insignificant,
        significant_hit,
        insignificant_hit,
        reject_options,
    )
    if vis is True:
        visualize(
            features=rejected_summary.significant,
            hits=rejected_summary.significant_hit,
            significance=True,
            eval_score=None,
        )
        visualize(
            features=rejected_summary.insignificant,
            hits=rejected_summary.insignificant_hit,
            significance=False,
            eval_score=None,
        )
    if vis is False:
        pass

    return LocalRejectStrategy(
        significant=rejected_summary.significant,
        insignificant=rejected_summary.insignificant,
        significant_hit=rejected_summary.significant_hit,
        insignificant_hit=rejected_summary.insignificant_hit,
        tentative=rejected_summary.tentative,
    )


def get_ozone_data(file_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(file_dir)
    dataframe.drop(columns="Date")
    dataframe.dropna()
    dataframe = dataframe.to_numpy()[1:2535]
    dataframe = dataframe.tolist()
    dataset = []
    for case in dataframe:
        case.pop(0)
        dataset.append(case)

    features = np.array(
        [
            [eval(value) for _index, value in enumerate(instance[:-1])]
            for instance in dataset
            if "?" not in instance
        ]
    )
    labels = np.array([instance[-1] for instance in dataset if "?" not in instance])
    return features, labels


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    seed_everything(seed=4)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ppc", type=int, required=False, default=1)
    parser.add_argument("--dataset", type=str, required=False, default="wdbc")
    parser.add_argument("--model", type=str, required=False, default=LVQ.MRSLVQ)
    parser.add_argument("--regularization", type=float, required=False, default=1.0)
    parser.add_argument("--eval_type", type=str, required=False, default="mv")
    parser.add_argument("--max_iter", type=int, required=False, default=100)
    parser.add_argument("--verbose", type=int, required=False, default=1)
    parser.add_argument(
        "--significance", action='store_true', default=False
    )
    parser.add_argument("--norm_ord", type=str, required=False, default="fro")
    parser.add_argument(
        "--evaluation_metric", type=str, required=False, default="accuracy"
    )
    parser.add_argument("--perturbation_ratio", type=float, required=False, default=0.2)
    parser.add_argument("--termination", type=str, required=False, default="metric")
    parser.add_argument(
        "--perturbation_distribution", type=str, required=False, default="global"
    )
    # parser.add_argument("--optimal_search", type=str, required=False, default="gpu")
    parser.add_argument("--reject_option", action='store_true', default=False)
    parser.add_argument("--epsilon", type=float, required=False, default=0.05)
    parser.add_argument("--proto_init", type=str, required=False, default="SMCI")
    parser.add_argument("--omega_init", type=str, required=False, default="OLTI")

    model = parser.parse_args().model
    eval_type = parser.parse_args().eval_type
    ppc = parser.parse_args().ppc
    max_iter = parser.parse_args().max_iter
    dataset = parser.parse_args().dataset
    verbose = parser.parse_args().verbose
    significance = parser.parse_args().significance
    norm_ord = parser.parse_args().norm_ord
    # optimal_search = parser.parse_args().optimal_search
    evaluation_metric = parser.parse_args().evaluation_metric
    perturbation_ratio = parser.parse_args().perturbation_ratio
    termination = parser.parse_args().termination
    perturbation_distribution = parser.parse_args().perturbation_distribution
    reject_option = parser.parse_args().reject_option
    epsilon = parser.parse_args().epsilon
    regularization = parser.parse_args().regularization
    standard_scaler = StandardScaler()
    if dataset == "ozone":
        input_data, labels = get_ozone_data("./data/eighthr.csv")
        input_data = standard_scaler.fit_transform(
            input_data
            ) if eval_type == \
                ValidationType.MUTATEDVALIDATION.value else input_data
        num_classes = len(np.unique(labels))
        latent_dim = input_data.shape[1]
    elif dataset == "wdbc":
        train_data = DATA(random=4)
        input_data = standard_scaler.fit_transform(
            train_data.breast_cancer.input_data
            ) if eval_type == ValidationType.MUTATEDVALIDATION.value \
                else train_data.breast_cancer.input_data
        labels = train_data.breast_cancer.labels
        num_classes = len(np.unique(labels))
        latent_dim = input_data.shape[1]
        initializer = get_Kmeans_prototypes(
            input_data=input_data,num_cluster=num_classes
            ).Prototypes 

    else:
        raise NotImplementedError

    train = TM(
        input_data=input_data,
        labels=labels,
        model_name=model,
        # optimal_search=optimal_search,
        latent_dim=latent_dim,
        num_classes=num_classes,
        init_prototypes=None,
        num_prototypes=ppc,
        eval_type=eval_type,
        significance=significance,
        evaluation_metric=evaluation_metric,
        perturbation_ratio=perturbation_ratio,
        perturbation_distribution=perturbation_distribution,
        epsilon=epsilon,
        norm_ord=norm_ord,
        termination=termination,
        verbose=verbose,
        max_epochs=max_iter,
        regularization=regularization,
    )
    summary = train.summary_results

    summary_significant = (
        summary.significant
        if model == LVQ.MRSLVQ.value
        else summary.significant.features  # type: ignore
    )
    summary_insignificant = (
        summary.insignificant
        if model == LVQ.MRSLVQ.value
        else summary.insignificant.features  # type: ignore
    )
    SUMMARY_TITLE = (
        "Summary" if model == LVQ.MRSLVQ.value else "Without rejection strategy"
    )

    print(f"--------------------{SUMMARY_TITLE}-------------------------")
    significant_features = summary_significant
    insignificant_features = summary_insignificant
    print(
        "significant_features=",
        significant_features,
    )
    print(
        "insignificant_features=",
        insignificant_features,
    )
    print("significant_features_size=", len(significant_features))  # type: ignore
    print("insignificant_features_size=", len(insignificant_features))  # type: ignore

    if reject_option and model == LVQ.LMRSLVQ.value:
        rejected_strategy = get_rejection_summary(
            significant=summary.significant.features,
            insignificant=summary.insignificant.features,
            significant_hit=summary.significant.hits,
            insignificant_hit=summary.insignificant.hits,
            reject_options=True,
            vis=True,
        )

        print("----------------------With reject_strategy----------------------------")
        significant_features = rejected_strategy.significant
        insignificant_features = rejected_strategy.insignificant
        tentative_features = rejected_strategy.tentative
        print(
            "significant_features=",
            significant_features,
        )
        print(
            "insignificant_features=",
            insignificant_features,
        )

        print("tentative_features=", tentative_features)

        print("significant_features_size=", len(significant_features))
        print("insignificant_features_size=", len(insignificant_features))
        print("tentative_features_size=", len(tentative_features))  # type: ignore



## python prototype_based_feature_extratctor1.py --dataset wdbc --model mrslvq --eval_type ho --reject_option --perturbation_ratio 0.2
