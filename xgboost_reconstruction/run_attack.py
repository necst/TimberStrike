import contextlib
import json
import os
import pickle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import xgboost_reconstruction as xgb_rec
from xgboost_reconstruction.reconstruct import impute_missing_values


def tsne_plot(dataset, final_df, filename, to_drop=None):
    original = dataset.copy()
    reconstructed = final_df.copy()

    # eliminate to_drop columns
    if to_drop is not None:
        original = original.drop(columns=to_drop)
        reconstructed = reconstructed.drop(columns=to_drop)

    # reconstructed change "label" to "stroke"
    reconstructed = reconstructed.rename(columns={"label": "stroke"})

    tsne = TSNE(n_components=2, random_state=42)

    X = np.concatenate([original, reconstructed], axis=0)
    y = np.concatenate([np.zeros(original.shape[0]), np.ones(reconstructed.shape[0])])

    X_embedded = tsne.fit_transform(X)

    # save the plot
    plt.figure(figsize=(10, 10))

    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], label="Original", c="blue", alpha=0.5)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], label="Reconstructed", c="red", alpha=0.5)

    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.close()


def get_result(dataset, final_df):
    original = dataset.copy()
    reconstructed = final_df.copy()

    distances = pairwise_distances(reconstructed, original)

    closest, distance = np.argmin(distances, axis=1), np.min(distances, axis=1)

    sid = np.argmin(distance)

    print("=" * 50)
    print(f"Comparison for minimum distance sample {sid}")
    print(f"Reconstructed -> Original: {reconstructed.iloc[sid].values} -> {original.iloc[closest[sid]].values}")
    print(f"Distance: {distance[sid]}")

    print("=" * 50)
    print("Mean Distance: ", np.mean(distance))

    return distances


class Attack:
    def __init__(self, data: pd.DataFrame,
                 target: pd.DataFrame,
                 fn: list[str],
                 log_path: str,
                 thresholds: list[float],
                 nClients: int = None,
                 categorical_features: list[str] = None,
                 categorical_values: dict[str, list[str]] = None,
                 numerical_ranges: dict[str, list[float]] = None,
                 to_drop=None, tolerance=0.1,
                 force=False,
                 baseline=False):
        """ Utility class to run reconstruction attack on a given dataset

        :param data: The data to be reconstructed
        :param target: The target of the data
        :param fn: The feature names
        :param log_path: The path to store the logs
        :param nClients: The number of clients
        :param thresholds: The thresholds to be used for reconstruction (influence the number of samples to keep)
        :param categorical_features: The categorical features
        :param categorical_values: The categorical values
        :param to_drop: The columns to drop
        :param force: Force to run the reconstruction attack
        :param baseline: Whether to use the baseline or not
        """
        self.data = data
        self.target = target
        self.fn = fn
        self.log_path = log_path
        self.nClients = nClients
        self.thresholds = thresholds
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.numerical_ranges = numerical_ranges
        self.to_drop = to_drop
        self.tolerance = tolerance
        self.force = force
        self.baseline = baseline

    def run(self, xgb_params=None, model: xgb.XGBClassifier = None, name=None, compromised_df: pd.DataFrame = None,
            evaluation_step=5) -> tuple[
        dict[float, pd.DataFrame], xgb_rec.Database]:
        if name is None:
            if xgb_params is not None:
                name = f"rec_{self.nClients}_{xgb_params['n_estimators']}_{xgb_params['max_depth']}"
            else:
                name = f"rec_{self.nClients}"
        if not self.force and os.path.exists(os.path.join(self.log_path, f"{name}.pkl")):
            with open(os.path.join(self.log_path, f"{name}.pkl"), "rb") as f:
                db = pickle.load(f)
        else:
            if xgb_params is None:
                assert model is not None, "You must provide either xgb_params or model"
            else:
                model = xgb.XGBClassifier(**xgb_params)

                model.fit(self.data, self.target)

            n_trees = len(model.get_booster().get_dump())
            tree_dfs = [model.get_booster()[i].trees_to_dataframe() for i in range(n_trees)]
            xgb_info = xgb_rec.XGBoostInfo(model=model)
            db = self._attack(tree_dfs, xgb_info, name, compromised_df=compromised_df, evaluation_step=evaluation_step)

        dfs, db = self.eval_attack(db, compromised_df=compromised_df)

        return dfs, db

    def run_dfs(self, tree_dfs, xgb_info, name=None, compromised_df: pd.DataFrame = None, evaluation_step=5) -> tuple[
        dict[float, pd.DataFrame], xgb_rec.Database]:
        if name is None:
            name = f"rec_{self.nClients}"
        if not self.force and os.path.exists(os.path.join(self.log_path, f"{name}.pkl")):
            with open(os.path.join(self.log_path, f"{name}.pkl"), "rb") as f:
                db = pickle.load(f)
        else:
            db = self._attack(tree_dfs, xgb_info, name, compromised_df=compromised_df, evaluation_step=evaluation_step)

        dfs, db = self.eval_attack(db, compromised_df=compromised_df)

        return dfs, db

    def _attack(self, tree_dfs: list[pd.DataFrame], xgb_info: xgb_rec.XGBoostInfo, name: str, compromised_df=None,
                evaluation_step=5) -> xgb_rec.Database:
        reconstructor = xgb_rec.DatabaseReconstructor(tree_dfs, self.fn, xgb_info)

        print(f"Number of samples in reconstruction: {len(reconstructor.database.samples)}")

        log = os.path.join(self.log_path, "log")
        log_path = os.path.join(log, "iter_1.log")

        os.makedirs(log, exist_ok=True)
        # while reconstructor.step(log_path=log_path):
        while reconstructor.step(baseline=self.baseline):
            log_path = os.path.join(log, f"iter_{reconstructor.reconstruct_iteration}")
            if reconstructor.reconstruct_iteration >= evaluation_step and reconstructor.reconstruct_iteration % evaluation_step == 0:
                os.makedirs(log_path, exist_ok=True)
                with open(os.path.join(log_path, "reconstructed_db.pkl"), "wb") as f:
                    pickle.dump(reconstructor.database, f)

                    # perform evaluation
                    old_log = self.log_path
                    self.log_path = log_path
                    with open(os.path.join(log_path, "log.txt"), "w") as f:
                        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            self.eval_attack(reconstructor.database, compromised_df=compromised_df)
                    self.log_path = old_log

        db = reconstructor.database

        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(db, f)

        return db

    def eval_attack(self, db: xgb_rec.Database, compromised_df=None):
        # try different thresholds
        dfs = {}

        for thresh in self.thresholds:
            print(f"Threshold: {thresh}")
            final_df = xgb_rec.database_to_final_dataframe(db, categorical_features=self.categorical_features,
                                                           categorical_values=self.categorical_values,
                                                           numerical_ranges=self.numerical_ranges, threshold=thresh,
                                                           compromised_df=compromised_df)

            dfs[thresh] = final_df

            dataset = pd.concat([self.data, self.target], axis=1)

            print(f"Original: {dataset.shape}")
            print(f"Reconstructed: {final_df.shape}")

            X_reconstructed = np.hstack([final_df[self.fn].values, final_df['label'].values.reshape(-1, 1)])
            X_original = np.hstack([self.data.values, self.target.values.reshape(-1, 1)])

            if self.to_drop is not None and len(self.to_drop) > 0:
                drop_idx = [self.fn.index(name) for name in self.to_drop]
                X_reconstructed = np.delete(X_reconstructed, drop_idx, axis=1)
                X_original = np.delete(X_original, drop_idx, axis=1)

            print(f"X_reconstructed shape: {X_reconstructed.shape}")
            print(f"X_original shape: {X_original.shape}")

            categorical_idx = [self.fn.index(name) for name in self.categorical_features]

            reordered, error_map, tolerance_map = xgb_rec.match_reconstruction(X_reconstructed, X_original,
                                                                               categorical_idx, tol=self.tolerance)

            print(f"Tolerance map: {tolerance_map}")

            reconstruction_accuracy = (1 - error_map.sum() / error_map.size) * 100

            print(f"Reconstruction accuracy: {reconstruction_accuracy:.2f}%")

            fn = [f for f in self.fn if f not in self.to_drop]

            X_original = pd.DataFrame(X_original, columns=fn + ['label'])
            reordered = pd.DataFrame(reordered, columns=fn + ['label'])
            # print cols of X_original and reordered and compromised_df
            print(f"X_original columns: {X_original.columns}")
            print(f"Reordered columns: {reordered.columns}")
            if compromised_df is not None:
                print(f"Compromised columns: {compromised_df.columns}")
            compromised_df.drop(columns=self.to_drop, inplace=True, errors='ignore')

            # summary stats
            print(f"X_original summary: {X_original.describe()}")
            print(f"Reconstructed summary: {reordered.describe()}")

            reordered = impute_missing_values(reordered, self.categorical_features, compromised_df=compromised_df)

            get_result(reordered, X_original)

            tsne_plot(X_original, reordered, os.path.join(self.log_path, f"tsne_plot_{thresh}"))

        return dfs, db


def _get_leaves_per_sample(db, tree):
    """
    This function returns the leaves per sample (leaves in which the sample may fall in, for the current iteration)
    """
    leaves_per_sample = [[] for _ in range(len(db.get_samples()))]
    leaves = tree.leaves
    leaves = list(leaves.values())

    for sid in range(len(db.get_samples())):
        sample = db.get_samples()[sid]
        leaves_per_sample[sid] = _get_leaves(tree, sample, leaves)

    return leaves, leaves_per_sample


def _get_leaves(tree, sample, leaves_index):
    """
    This function returns the leaves in which the sample may fall in, for the current iteration
    """
    tree_leaves = tree.leaves  # this is a dict[int: TreeNode]
    leaves = []

    for leaf_id, leaf in tree_leaves.items():
        if leaf.may_contain(sample):
            # find the index of the leaf in the leaves_index
            index = leaves_index.index(leaf)
            leaves.append(index)

    return leaves


def aggregate(
        bst_prev_org: Optional[bytes],
        bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def run_attack_bagging_with_interleaving(reconstructor: xgb_rec.DatabaseReconstructor, models, victim_cid,
                                         feature_names,
                                         params, log, data, fn, cat_fn, cat_values, num_ranges, tol=0.1, to_drop=None,
                                         compromised_df=None,
                                         cold_start=True, evaluation_step=5, baseline=False):
    to_consider = []
    # log_path = f"{log}/iter_1.log"

    for i in range(len(models)):
        if models[i]["cid"] != victim_cid or cold_start:
            if models[i]["cid"] != victim_cid:
                to_consider.append(i)
            else:
                cold_start = False
        else:

            print(f"Victim client: {models[i]['cid']}")
            print(f"Clients to consider: {to_consider}")

            booster = None
            for j in to_consider:
                booster = aggregate(booster, models[j]["tree"])

            if booster is not None:
                booster = bytearray(booster)
                xgb_booster = xgb.Booster(params=params)
                xgb_booster.load_model(booster)
                boosters = [b for b in xgb_booster]
                cls_rec = xgb_rec.RecoverXGBClassifier(boosters).recover_xgb()
                cls_rec.get_booster().set_param(params=params)

                for j in range(len(boosters)):
                    tree = xgb_rec.build_tree(boosters[j].trees_to_dataframe(), feature_names, reconstructor.xgb_info)

                    leaves, leaves_per_sample = _get_leaves_per_sample(reconstructor.database, tree)

                    print(
                        f"Iteration {i}, Other client {j}, Number of leaves: {len(leaves)}, Number of samples: {len(leaves_per_sample)}")
                    print(
                        f"Mean leaf per sample: {np.mean([len(l) for l in leaves_per_sample])} (min: {np.min([len(l) for l in leaves_per_sample])}, max: {np.max([len(l) for l in leaves_per_sample])})")

                    for sid in range(len(leaves_per_sample)):
                        h_leaves = [leaves[leaf].H for leaf in leaves_per_sample[sid]]

                        pi = np.sum([h * leaves[leaf].leaf_value for h, leaf in
                                     zip(h_leaves, leaves_per_sample[sid])]) / np.sum(h_leaves)

                        reconstructor.database.get_samples()[sid].update_pi(pi)

            # if not reconstructor.step(log_path=log_path):
            if not reconstructor.step(baseline=baseline):
                break
            elif reconstructor.reconstruct_iteration >= evaluation_step and reconstructor.reconstruct_iteration % evaluation_step == 0:
                log_path = f"{log}/iter_{reconstructor.reconstruct_iteration}"
                os.makedirs(log_path, exist_ok=True)
                with open(os.path.join(log_path, "reconstructed_db.pkl"), "wb") as f:
                    pickle.dump(reconstructor.database, f)

                    # perform evaluation
                    old_log = log
                    log = log_path
                    with open(os.path.join(log_path, "log.txt"), "w") as f:
                        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            evaluate_attack(log, reconstructor.database, data, fn, cat_fn, cat_values, num_ranges, tol=tol,
                                            to_drop=to_drop, compromised_df=compromised_df)
                    log = old_log

            to_consider = []
            # log_path = f"{log}/iter_{reconstructor.reconstruct_iteration}.log"


def evaluate_attack(log_path: str, db: xgb_rec.Database, data: list, fn: list, cat_fn: list,
                    cat_values: dict, num_ranges: dict, tol: float = 0.1, to_drop: list = None, compromised_df=None):

    final_df = xgb_rec.database_to_final_dataframe(db, categorical_features=cat_fn, categorical_values=cat_values,
                                                   numerical_ranges=num_ranges, compromised_df=compromised_df)

    x = pd.DataFrame(data[0], columns=fn)
    y = pd.DataFrame(data[1], columns=['target'])

    dataset = pd.concat([x, y], axis=1)

    X_reconstructed = np.hstack([final_df[fn].values, final_df['label'].values.reshape(-1, 1)])
    X_original = np.hstack([x.values, y.values.reshape(-1, 1)])

    if to_drop is not None and len(to_drop) > 0:
        # make a boolean array out of to_drop
        drop_idx = [fn.index(name) for name in to_drop]
        X_reconstructed = np.delete(X_reconstructed, drop_idx, axis=1)
        X_original = np.delete(X_original, drop_idx, axis=1)

    print(f"X_reconstructed shape: {X_reconstructed.shape}")
    print(f"X_original shape: {X_original.shape}")

    categorical_idx = [fn.index(name) for name in cat_fn]

    reordered, error_map, tolerance_map = xgb_rec.match_reconstruction(X_reconstructed, X_original, categorical_idx,
                                                                       tol=tol)

    print(f"Tolerance map: {tolerance_map}")

    reconstruction_accuracy = (1 - error_map.sum() / error_map.size) * 100

    print(f"Reconstruction accuracy: {reconstruction_accuracy:.2f}%")

    fn = [f for f in fn if f not in to_drop]

    X_original = pd.DataFrame(X_original, columns=fn + ['label'])
    reordered = pd.DataFrame(reordered, columns=fn + ['label'])
    print(f"X_original columns: {X_original.columns}")
    print(f"Reordered columns: {reordered.columns}")
    if compromised_df is not None:
        print(f"Compromised columns: {compromised_df.columns}")
    compromised_df.drop(columns=to_drop, inplace=True, errors='ignore')

    print(f"X_original summary: {X_original.describe()}")
    print(f"Reconstructed summary: {reordered.describe()}")

    reordered = impute_missing_values(reordered, cat_fn, compromised_df=compromised_df)

    get_result(reordered, X_original)

    tsne_plot(X_original, reordered, os.path.join(log_path, 'tsne_plot'))


def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num
