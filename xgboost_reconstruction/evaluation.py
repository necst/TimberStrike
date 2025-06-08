"""
This module is taken from [tableak](https://github.com/eth-sri/tableak) and used to evaluate the reconstruction quality
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
        are returned separately.
    """
    cat_score = 0
    cont_score = 0
    num_cats = 0
    num_conts = 0

    for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            cont_score += 0 if (
                    float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
            num_conts += 1
        else:
            raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
                            'the string >cat< to mark the position of a categorical feature.')
    if detailed:
        if num_cats < 1:
            num_cats = 1
        if num_conts < 1:
            num_conts = 1
        return (cat_score + cont_score) / (num_cats + num_conts), cat_score / num_cats, cont_score / num_conts
    else:
        return (cat_score + cont_score) / (num_cats + num_conts)


def categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """
    Calculates an error score between the true mixed-type datapoint and a reconstructed mixed-type datapoint. For each
    categorical feature we count a 0-1 error by the rule of the category being reconstructed correctly. For each
    continuous feature we count a 0-1 error by the rule of the continuous variable being reconstructed within a
    symmetric tolerance interval around the true value. The tolerance parameters are set by 'tolerance_map'.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector or matrix if comprising more than
        datapoint.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector/matrix.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If a batch of data is given, then the average accuracy of the batch is returned. Additionally, if the flag
        'detailed' is set to True the reconstruction errors of the categorical and the continuous features are returned
        separately.
    """
    assert true_data.shape == reconstructed_data.shape
    score = 0
    cat_score = 0
    cont_score = 0
    if len(true_data.shape) > 1:
        for true_data_line, reconstructed_data_line in zip(true_data, reconstructed_data):
            assert len(true_data_line) == len(tolerance_map)
            if detailed:
                scores = _categorical_accuracy_continuous_tolerance_score(true_data_line, reconstructed_data_line,
                                                                          tolerance_map, True)
                score += 1 / true_data.shape[0] * scores[0]
                cat_score += 1 / true_data.shape[0] * scores[1]
                cont_score += 1 / true_data.shape[0] * scores[2]
            else:
                score += 1 / true_data.shape[0] * _categorical_accuracy_continuous_tolerance_score(true_data_line,
                                                                                                   reconstructed_data_line,
                                                                                                   tolerance_map)
    else:
        assert len(true_data) == len(tolerance_map)
        if detailed:
            scores = _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map,
                                                                      True)
            score += scores[0]
            cat_score += scores[1]
            cont_score += scores[2]
        else:
            score = _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map)

    if detailed:
        return score, cat_score, cont_score
    else:
        return score


def match_reconstruction_ground_truth(target_batch, reconstructed_batch, tolerance_map, return_indices=False,
                                      match_based_on='all'):
    """
    For a reconstructed batch of which we do not know the order of datapoints reconstructed, as the loss and hence the
    gradient of the loss is permutation invariant with respect to the input batch, this function calculates the optimal
    reordering i.e. matching to the ground truth batch to get the minimal reconstruction error. It uses the
    reconstruction score 'categorical_accuracy_continuous_tolerance_score'.

    :param target_batch: (np.ndarray) The target batch in mixed representation.
    :param reconstructed_batch: (np.ndarray) The reconstructed batch in mixed representation.
    :param tolerance_map: (list) The tolerance map required to calculate the reconstruction score.
    :param return_indices: (bool) Trigger to return the matching index map as well.
    :param match_based_on: (str) Select based on which feature type to match. Available are 'all', 'cat', 'cont'.
    :return: reordered_reconstructed_batch (np.ndarray), batch_cost_all (np.ndarray), batch_cost_cat (np.ndarray),
        batch_cost_cont (np.ndarray): The correctly reordered reconstructed batch, the minimal cost vectors of all
        feature costs, only categorical feature costs, only continuous feature costs.
    """
    assert match_based_on in ['all', 'cat', 'cont'], 'Please select a valid matching ground from all, cat, cont'

    if len(target_batch.shape) != 2 or len(reconstructed_batch.shape) != 2:
        target_batch = np.reshape(target_batch, (-1, len(tolerance_map)))
        reconstructed_batch = np.reshape(reconstructed_batch, (-1, len(tolerance_map)))

    batch_size = target_batch.shape[0]

    # create the cost matrix for matching the reconstruction with the true data to calculate the score
    cost_matrix_all, cost_matrix_cat, cost_matrix_cont = [np.zeros((batch_size, batch_size)) for _ in range(3)]
    for k, target_data_point in enumerate(target_batch):
        for l, recon_data_point in enumerate(reconstructed_batch):
            cost_all, cost_cat, cost_cont = categorical_accuracy_continuous_tolerance_score(
                target_data_point, recon_data_point, tolerance_map, True)
            cost_matrix_all[k, l], cost_matrix_cat[k, l], cost_matrix_cont[k, l] = cost_all, cost_cat, cost_cont

    # perform the Hungarian algorithm to match the reconstruction
    if match_based_on == 'all':
        row_ind, col_ind = linear_sum_assignment(cost_matrix_all)
    elif match_based_on == 'cat':
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cat)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cont)

    # create the "real" cost vectors and reorder the reconstructed batch according to the matching
    reordered_reconstructed_batch = reconstructed_batch[col_ind]
    batch_cost_all, batch_cost_cat, batch_cost_cont = cost_matrix_all[row_ind, col_ind], \
        cost_matrix_cat[row_ind, col_ind], \
        cost_matrix_cont[row_ind, col_ind]

    if return_indices:
        return reordered_reconstructed_batch, batch_cost_all, batch_cost_cat, batch_cost_cont, row_ind, col_ind
    else:
        return reordered_reconstructed_batch, batch_cost_all, batch_cost_cat, batch_cost_cont


def create_tolerance_map(x, categorical_idx, tol=0.319):
    """
    Given a tolerance value for multiplying the standard deviation, this method calculates a tolerance map that is
    required for the error calculation between a guessed/reconstructed batch and a true batch of data.

    :param x: (np.ndarray) The true/reference mixed-type feature matrix.
    :param categorical_idx: (list) The indices of the categorical features.
    :param tol: (float) Tolerance value. The tolerance interval for each continuous feature will be calculated as:
        [true - tol, true + tol].
    :return: (list) The tolerance map required for the error calculation.
    """
    x_std = np.std(x, axis=0)
    cont_indices = [i for i in range(x.shape[1]) if i not in categorical_idx]
    numeric_stds = x_std[cont_indices]
    tolerance_map = []
    pointer = 0

    for i in range(x.shape[1]):
        if i in cont_indices:
            tolerance_map.append(tol * numeric_stds[pointer])
            pointer += 1
        else:
            tolerance_map.append('cat')

    return tolerance_map


def match_reconstruction(X_rec, X_true, categorical_idx, tol=0.319):
    """
    Given a tolerance value for multiplying the standard deviation, this method calculates a tolerance map that is
    required for the error calculation between a guessed/reconstructed batch and a true batch of data.

    :param X_rec: (np.ndarray) The reconstructed mixed-type feature matrix.
    :param X_true: (np.ndarray) The true/reference mixed-type feature matrix.
    :param categorical_idx: (list) The indices of the categorical features.
    :param tol: (float) Tolerance value. The tolerance interval for each continuous feature will be calculated as:
        [true - tol, true + tol].
    :return: (list) The tolerance map required for the error calculation.
    """
    n_true = len(X_true)
    n_rec = len(X_rec)
    indices_true = indices_rec = None

    # Handle different sizes if needed (in this case the evaluation will be a lower bound of the true accuracy)
    if n_true != n_rec:
        min_size = min(n_true, n_rec)
        np.random.seed(42)
        indices_true = np.random.choice(n_true, size=min_size, replace=False)
        indices_rec = np.random.choice(n_rec, size=min_size, replace=False)
        X_true = X_true[indices_true]
        X_rec = X_rec[indices_rec]

        print(f"Original and reconstructed data have different sizes. Randomly selected {min_size} samples from each.")

    tolerance_map = create_tolerance_map(X_true, categorical_idx, tol)
    reordered_reconstructed_batch, batch_cost_all, _, _ = match_reconstruction_ground_truth(X_true, X_rec,
                                                                                            tolerance_map)
    return reordered_reconstructed_batch, batch_cost_all, tolerance_map


def reduce_dataset(X_true, reordered_reconstructed, batch_cost_all, threshold=0.95):
    """
    Given the reordered reconstructed batch and the true batch, this method reduces the dataset to a subset of the
    datapoints that have a reconstruction accuracy above a certain threshold.

    :param X_true: (np.ndarray) The true/reference mixed-type feature matrix.
    :param reordered_reconstructed: (np.ndarray) The reordered reconstructed mixed-type feature matrix.
    :param reconstruction_accuracy: (np.ndarray) The accuracy score
    :param threshold: (float) The target accuracy threshold.

    :return: (np.ndarray) The reduced dataset.
    """
    error_map = batch_cost_all

    acc = 1 - np.mean(error_map)

    n = error_map.size

    sorted_error_map = np.sort(error_map)
    sorted_indices = np.argsort(error_map)
    valid = sorted_indices[:n]

    while acc < threshold:
        n = n - 1
        valid = sorted_indices[:n]
        acc = (1 - sorted_error_map[:n].sum() / n) * 100

    return X_true[valid], reordered_reconstructed[valid], valid, acc, n


def get_k_least_important_features(model, k, feature_names, importance_type='weight'):
    """
    Get the k the least important feature names from a trained XGBoost classifier.

    Parameters:
    -----------
    model : xgboost.XGBClassifier
        The trained XGBoost classifier
    k : int
        Number of least important features to return
    feature_names : list, optional
        List of feature names
    importance_type : str, optional
        Type of feature importance: 'weight', 'gain', or 'cover'

    Returns:
    --------
    list: Names of the k least important features
    """
    importance_scores = model.get_booster().get_score(importance_type=importance_type)

    full_importance = {name: importance_scores.get(name, 0) for name in feature_names}
    least_important = sorted(full_importance.items(), key=lambda x: x[1])[:k]

    return [feature[0] for feature in least_important]
