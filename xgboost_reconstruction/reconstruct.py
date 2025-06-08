import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pulp
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

import xgboost_reconstruction as xgb_rec

np.random.seed(42)

class FeatureRange(dict[str, xgb_rec.Range]):
    def __init__(self, feature_range: dict[str, xgb_rec.Range] = None):
        super().__init__()
        self.feature_range = feature_range

    def __getitem__(self, key):
        return self.feature_range[key]

    def items(self):
        return self.feature_range.items()

    def __str__(self):
        s = ""
        for name, range in self.feature_range.items():
            rs = range.__str__()
            if rs != "":
                s += f"{name}: {rs}, "
        return s

    def __repr__(self):
        return self.__str__()


class Sample:
    def __init__(self, features_ranges: dict[str, xgb_rec.Range], label: int, pi: float):
        self.features = FeatureRange(features_ranges)
        self.label = label
        self.pi = pi  # the previous probability of the sample

    def update_feature(self, feature_name: str, min_val: float, max_val: float):
        self.features[feature_name].update_range(min_val, max_val)

    def update_pi(self, pi: float):
        self.pi += pi

    @property
    def sigmoid(self) -> float:
        return 1 / (1 + np.exp(-self.pi))

    @property
    def gi(self) -> float:
        """
        The gradient of the sample
        """
        return self.sigmoid - self.label

    @property
    def hi(self) -> float:
        """
        The hessian of the sample
        """
        return self.sigmoid * (1 - self.sigmoid)

    def update_range(self, feature_name: str, min_val: float, max_val: float):
        self.features[feature_name] = xgb_rec.Range(min_val, max_val)

    def update_label(self, label: int):
        self.label = label

    def __str__(self) -> str:
        return f"Features: {self.features}, Label: {self.label}"

    def __repr__(self) -> str:
        return f"Features: {self.features}, Label: {self.label}"


class Database:
    def __init__(self, xgb_info: xgb_rec.XGBoostInfo, tree_dfs: list[pd.DataFrame], first_tree: xgb_rec.Tree,
                 features_name: list[str]):
        self.xgb_info = xgb_info
        self.first_tree_df = tree_dfs[0]
        self.first_tree = first_tree
        self.features_name = features_name

        # initialize the samples
        self.samples: list[Sample] = self._init_samples()

    def add_sample(self, sample: Sample):
        self.samples.append(sample)

    def __str__(self):
        return str(self.samples)

    def __repr__(self):
        return str(self.samples)

    def get_samples(self) -> list[Sample]:
        return self.samples

    def _init_samples(self) -> list[Sample]:
        num_samples = self._get_num_samples(self.first_tree_df, self.xgb_info.base_score, self.xgb_info.lr,
                                            self.xgb_info.lambda_)

        samples = []

        for leaf_id, total_samples, N_0, N_1, leaf_value in num_samples:
            initial_pi = self.xgb_info.log_odds_base_score + leaf_value

            for positive_samples in range(N_1):
                ranges = self.first_tree.get_leaf(leaf_id).features_ranges
                ranges = deepcopy(ranges)
                sample = Sample(ranges, label=1, pi=initial_pi)
                samples.append(sample)
            for negative_samples in range(N_0):
                ranges = self.first_tree.get_leaf(leaf_id).features_ranges
                ranges = deepcopy(ranges)
                sample = Sample(ranges, label=0, pi=initial_pi)
                samples.append(sample)

        return samples

    def _get_num_samples(self, first_tree_df: pd.DataFrame, base_score: float, lr: float, lambda_: float) -> list[
        tuple[int, int, int, int, float]]:
        leaf_cover = first_tree_df[first_tree_df['Feature'] == 'Leaf'][['Node', 'Gain', 'Cover']].values

        # get the number of samples in each leaf
        num_samples = []

        for leaf in leaf_cover:
            leaf_id = int(leaf[0])
            leaf_value = float(leaf[1]) / lr
            cover = float(leaf[2])

            # get the number of samples in the leaf
            G = - leaf_value * (cover + lambda_)  # Gradient

            total_samples = round(cover / (base_score * (1 - base_score)))

            N_1 = round((cover / (1 - base_score)) - G)
            N_0 = total_samples - N_1

            num_samples.append((leaf_id, total_samples, N_0, N_1, leaf_value * lr))

        return num_samples

    def to_dataframe(self) -> pd.DataFrame:
        data = []

        def to_tuple(feature_range):
            if feature_range.min == -np.inf and feature_range.max == np.inf:
                return "-"
            return feature_range.min, feature_range.max

        for sample in self.samples:
            row = {feature: to_tuple(sample.features[feature]) for feature in self.features_name}
            row['pi'] = sample.pi
            row['sigmoid'] = sample.sigmoid
            row['gi'] = sample.gi
            row['hi'] = sample.hi
            row['label'] = sample.label
            data.append(row)
        return pd.DataFrame(data)


class DatabaseReconstructor:
    def __init__(self, tree_dfs: list[pd.DataFrame], features_name: list[str], xgb_info: xgb_rec.XGBoostInfo):
        self.xgb_info = xgb_info
        self.trees = xgb_rec.TreeFactory(tree_dfs, features_name, xgb_info).build_trees()
        self.database = Database(xgb_info, tree_dfs, self.trees[0], features_name)
        self.reconstruct_iteration = 1
        # load the environment variables
        load_dotenv()

    def step(self, log_path=None, time_limit=None, gap=None, topK=0, baseline=False):
        """
        This function reconstructs the database for the next iteration
        """

        print(f"Reconstructing iteration {self.reconstruct_iteration}")

        leaves, leaves_per_sample = self._get_leaves_per_sample()
        samples = self.database.get_samples()

        # print(leaves)
        # print(leaves_per_sample)
        print(
            f"Mean number of leaves per sample: {np.mean([len(l) for l in leaves_per_sample])} (min: {np.min([len(l) for l in leaves_per_sample])}, max: {np.max([len(l) for l in leaves_per_sample])})")

        I = len(samples)
        J = len(leaves)
        K = round(topK * I * J)  # topK assignments heuristics

        g = [sample.gi for sample in samples]
        h = [sample.hi for sample in samples]
        G = [leaf.G for leaf in leaves]
        H = [leaf.H for leaf in leaves]

        options = {}

        # get the stuff from the environment
        gurobi = False
        if os.getenv("GUROBI_ACCESSID") is not None and os.getenv("GUROBI_SECRET") is not None and os.getenv(
                "GUROBI_LICENSEID") is not None:
            gurobi = True
            options = {
                "WLSACCESSID": os.getenv("GUROBI_ACCESSID"),
                "WLSSECRET": os.getenv("GUROBI_SECRET"),
                "LICENSEID": int(os.getenv("GUROBI_LICENSEID")),
            }

        if baseline:
            # solve the problem using the baseline approach (random assignment to possible leaves)
            x = np.zeros((I, J))
            for i in range(I):
                possible_leaves = leaves_per_sample[i]
                if len(possible_leaves) == 0:
                    continue
                leaf_index = np.random.choice(possible_leaves)
                x[i, leaf_index] = 1
            for i in range(I):
                for j in range(J):
                    if x[i, j] == 1:
                        for feature_name, feature_range in samples[i].features.items():
                            feature_range.update_range(leaves[j].features_ranges[feature_name].min,
                                                       leaves[j].features_ranges[feature_name].max)
                        samples[i].update_pi(leaves[j].leaf_value)
                        break
        else:

            # Solve the problem
            problem, x = self._get_problem(I, J, g, h, G, H, leaves_per_sample, K)
            if gap is None:
                gap = 0 # if self.reconstruct_iteration == 1 else 0.00001  # 0% for the first iteration, 0.001% for the rest
            if time_limit is None:
                time_limit = 600 if self.reconstruct_iteration == 1 else 360  # 10 minutes for the first iteration, 6 minutes for the rest
            if gurobi:
                solver = pulp.GUROBI(msg=False, manageEnv=True, envOptions=options, timeLimit=time_limit, gapRel=gap,
                                     logPath=log_path)
            else:
                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap, logPath=log_path)
            problem.solve(solver)

            print(f"Problem status: {pulp.LpStatus[problem.status]}")
            print(f"Objective value: {pulp.value(problem.objective)}")
            print(f"Time: {solver.solveTime}")

            solver.close()
            # x = self.solve_problem_iteratively(I, J, g, h, G, H, leaves_per_sample, relax=False)

            # for i in range(I):
            #     for j in range(J):
            #         if x[i, j] == 1:
            #             print(f"Sample {i} is assigned to Leaf {j}")

            # # Calculate the total gradient and hessian for each leaf
            # for j in range(J):
            #     total_gradient = sum(g[i] * x[i, j].varValue for i in range(I))
            #     total_hessian = sum(h[i] * x[i, j].varValue for i in range(I))
            #     print(f'Leaf {j + 1}: Total Gradient = {total_gradient}, Total Hessian = {total_hessian}')

            # Update the samples
            for i in range(I):
                for j in range(J):
                    if x[i, j].varValue == 1:
                        # update ranges
                        # print(f"Sample {i} is assigned to Leaf {j}")

                        # update the ranges of the sample
                        for feature_name, feature_range in samples[i].features.items():
                            feature_range.update_range(leaves[j].features_ranges[feature_name].min,
                                                       leaves[j].features_ranges[feature_name].max)
                        # update the pi
                        samples[i].update_pi(leaves[j].leaf_value)

                        break  # the sample can only be assigned to one leaf

        self.reconstruct_iteration += 1

        if self.reconstruct_iteration == len(self.trees):
            print("Reconstruction finished")
            return False
        return True

    def _get_problem(self, I, J, g, h, G, H, leaves_per_sample, topK) -> (pulp.LpProblem, dict):

        if topK != 0:
            problem = pulp.LpProblem("Sample_Allocation_Problem", pulp.LpMinimize)

            # Define the decision variables
            x = pulp.LpVariable.dicts("x", ((i, j) for i in range(I) for j in range(J)), cat='Continuous', lowBound=0,
                                      upBound=1)
            grad = pulp.LpVariable.dicts("grad", (j for j in range(J)), lowBound=0, cat='Continuous')
            hess = pulp.LpVariable.dicts("hess", (j for j in range(J)), lowBound=0, cat='Continuous')

            # Constraints
            problem, x = self._add_constraints(problem, x, I, J, g, h, G, H, leaves_per_sample, grad, hess)

            # solve the relaxation problem
            solver = pulp.PULP_CBC_CMD(msg=False)
            problem.solve(solver)

            print(f"Relaxed problem status: {pulp.LpStatus[problem.status]}")

            # get the topK assignments (nearest to 0 or 1)
            sorted_x = sorted([(i, j, x[i, j].varValue) for i in range(I) for j in range(J)],
                              key=lambda x: min(x[2], 1 - x[2]))

            fixed_x = sorted_x[:topK]

            to_fix = []

            for i, j, val in fixed_x:
                if val > 0.5:
                    to_fix.append((i, j, 1))
                else:
                    to_fix.append((i, j, 0))

        # Solve the problem again, this time with the binary solution and the constraint to fix the topK assignments
        problem = pulp.LpProblem("Sample_Allocation_Problem", pulp.LpMinimize)

        # Define the decision variables
        x = pulp.LpVariable.dicts("x", ((i, j) for i in range(I) for j in range(J)), cat='Binary')
        grad = pulp.LpVariable.dicts("grad", (j for j in range(J)), lowBound=0, cat='Continuous')
        hess = pulp.LpVariable.dicts("hess", (j for j in range(J)), lowBound=0, cat='Continuous')

        # Constraints
        problem, x = self._add_constraints(problem, x, I, J, g, h, G, H, leaves_per_sample, grad, hess)

        # Fix the topK assignments
        if topK != 0:
            for i, j, val in to_fix:
                problem += x[i, j] == val

        problem.setObjective(pulp.lpSum([grad[j] + hess[j] for j in range(J)]))

        return problem, x

    def _add_constraints(self, problem, x, I, J, g, h, G, H, leaves_per_sample, grad, hess):
        # Constraints
        # Each sample must be assigned to exactly one leaf
        for i in range(I):
            problem += pulp.lpSum([x[i, j] for j in range(J)]) == 1

        # Absolute value constraints for gradients
        for j in range(J):
            problem += grad[j] >= pulp.lpSum([g[i] * x[i, j] for i in range(I)]) - G[j]
            problem += grad[j] >= G[j] - pulp.lpSum([g[i] * x[i, j] for i in range(I)])

        # Absolute value constraints for Hessians
        for j in range(J):
            problem += hess[j] >= pulp.lpSum([h[i] * x[i, j] for i in range(I)]) - H[j]
            problem += hess[j] >= H[j] - pulp.lpSum([h[i] * x[i, j] for i in range(I)])

        # Constraint: samples that cannot fall in a leaf should not be assigned to it
        for i in range(I):
            for j in range(J):
                if j not in leaves_per_sample[i]:
                    problem += x[i, j] == 0

        # Constraint: samples that can only fall in one leaf should be assigned to it
        for i in range(I):
            if len(leaves_per_sample[i]) == 1:
                problem += x[i, leaves_per_sample[i][0]] == 1

        return problem, x

    def _get_single_leaf_problem(self, I, J, g, h, G, H, leaf_index, assigned_samples, leaves_per_sample,
                                 relax=False) -> (
            pulp.LpProblem, dict):
        problem = pulp.LpProblem(f"Leaf_{leaf_index}_Problem", pulp.LpMinimize)

        # Define the decision variables for the current leaf
        x_leaf = pulp.LpVariable.dicts("x", (i for i in range(I)), cat=f"Continuous" if relax else "Binary")
        grad = pulp.LpVariable(f"grad_{leaf_index}", lowBound=0, cat='Continuous')
        hess = pulp.LpVariable(f"hess_{leaf_index}", lowBound=0, cat='Continuous')

        # Constraint: samples in assigned_samples should not be assigned to the current leaf because they are already assigned
        for i in assigned_samples:
            problem += x_leaf[i] == 0

        # Constraint: samples that cannot fall in the current leaf should not be assigned to it
        count = 0
        for i in range(I):
            if leaf_index not in leaves_per_sample[i]:
                problem += x_leaf[i] == 0
            else:
                count += 1

        # print(f"Leaf {leaf_index} has {count} samples possible to fall in it")

        # Constraint: The cumulative gradient and hessian should match the target as closely as possible
        problem += grad >= pulp.lpSum([g[i] * x_leaf[i] for i in x_leaf]) - G[leaf_index]
        problem += grad >= G[leaf_index] - pulp.lpSum([g[i] * x_leaf[i] for i in x_leaf])
        problem += hess >= pulp.lpSum([h[i] * x_leaf[i] for i in x_leaf]) - H[leaf_index]
        problem += hess >= H[leaf_index] - pulp.lpSum([h[i] * x_leaf[i] for i in x_leaf])

        # Objective function: Minimize the sum of absolute differences for the current leaf
        problem.setObjective(grad + hess)

        return problem, x_leaf

    def solve_problem_iteratively(self, I, J, g, h, G, H, leaves_per_sample, relax=False):
        assigned_samples = set()
        all_x = pulp.LpVariable.dicts("x", ((i, j) for i in range(I) for j in range(J)), cat='Binary')

        for j in range(J):
            problem, x_leaf = self._get_single_leaf_problem(I, J, g, h, G, H, j, assigned_samples, leaves_per_sample,
                                                            relax)
            solver = pulp.PULP_CBC_CMD(msg=False)
            problem.solve(solver)
            # print(f"Status of the problem for leaf {j}:", pulp.LpStatus[problem.status])

            # Collect the results for the current leaf
            for i in range(I):
                if not relax:
                    x = x_leaf[i].varValue
                else:
                    if x_leaf[i].varValue > 0.5:
                        x = 1
                    else:
                        x = 0
                if x == 1:
                    all_x[(i, j)] = 1
                    assigned_samples.add(i)
                    # update leaves_per_sample
                    leaves_per_sample[i] = [j]  # the sample can only fall in the current leaf
                else:
                    all_x[(i, j)] = 0
                    # update leaves_per_sample
                    if j in leaves_per_sample[i]:
                        leaves_per_sample[i].remove(j)

            # print(f"Already assigned samples: {len(assigned_samples)}")

        # Calculate the total gradient and hessian for each leaf using the binary solution
        for j in range(J):
            total_gradient = sum(g[i] * all_x[(i, j)] for i in range(I) if (i, j) in all_x)
            total_hessian = sum(h[i] * all_x[(i, j)] for i in range(I) if (i, j) in all_x)
            # print(f'Leaf {j + 1}: Total Gradient = {total_gradient}, Total Hessian = {total_hessian}')

        return all_x

    def _get_leaves_per_sample(self):
        """
        This function returns the leaves per sample (leaves in which the sample may fall in, for the current iteration)
        """
        leaves_per_sample = [[] for _ in range(len(self.database.get_samples()))]
        leaves = self.trees[self.reconstruct_iteration].leaves
        leaves = list(leaves.values())

        for sid in range(len(self.database.get_samples())):
            sample = self.database.get_samples()[sid]
            leaves_per_sample[sid] = self._get_leaves(sample, leaves)

        return leaves, leaves_per_sample

    def _get_leaves(self, sample, leaves_index):
        """
        This function returns the leaves in which the sample may fall in, for the current iteration
        """
        tree_leaves = self.trees[self.reconstruct_iteration].leaves  # this is a dict[int: TreeNode]
        leaves = []

        for leaf_id, leaf in tree_leaves.items():
            if leaf.may_contain(sample):
                # find the index of the leaf in the leaves_index
                index = leaves_index.index(leaf)
                leaves.append(index)

        return leaves


def database_to_final_dataframe(database: Database, categorical_features=None, categorical_values=None,
                                numerical_ranges=None,
                                threshold=None,
                                compromised_df: pd.DataFrame = None):
    """
    This function converts the database to a final dataframe that can be used for the visualization

    :param database: The database object
    :param categorical_features: The categorical features
    :param categorical_values: The possible values for the categorical features
    :param compromised_df: An optional dataframe to use for imputing the missing values
    """

    # at the moment the database contains ranges for each feature, take the mean of the ranges and return a dataframe
    if categorical_features is None:
        categorical_features = []
    if categorical_values is None:
        categorical_values = {}

    samples = database.get_samples()

    df = []

    count = 0

    for sample in samples:
        row = {}
        skip = False
        for feature_name, feature_range in sample.features.items():
            if feature_name in categorical_features:
                # find the value of the feature
                min = feature_range.min
                max = feature_range.max
                if feature_range.min == -np.inf and feature_range.max == np.inf:
                    row[feature_name] = np.nan
                    continue
                if feature_range.min == -np.inf:
                    min = 0
                if feature_range.max == np.inf:
                    max = np.max([value for value in categorical_values[feature_name]])
                possible_values = []
                for v in categorical_values[feature_name]:
                    if min <= v < max or min == v == max:
                        possible_values.append(v)
                row[feature_name] = possible_values[np.random.randint(0, len(possible_values))]
            else:
                min = feature_range.min
                max = feature_range.max
                if feature_range.min == -np.inf:
                    min = numerical_ranges[feature_name][0]
                if feature_range.max == np.inf:
                    max = numerical_ranges[feature_name][1]

                if threshold and (max - min >= threshold):
                    skip = True
                    break

                row[feature_name] = (min + max) / 2
        row['label'] = sample.label

        if not skip:
            count += 1
            df.append(row)

    df = pd.DataFrame(df)

    print(df)

    print(f"Number of samples: {count}")

    # print number of missing values for each column
    print("Missing values for each features")
    for feature in df.columns:
        print(f"{feature}: {df[feature].isnull().sum()}")

    # df = impute_missing_values(df, categorical_features, compromised_df)

    return df


class CustomImputer:
    def __init__(self, categorical_features):
        """
        Initialize the imputer with the list of categorical features to be imputed.

        :param categorical_features: The list of categorical features to be imputed.
        """
        self.categorical_features = categorical_features
        self.models = {}

    def fit(self, compromised: pd.DataFrame):
        """
        Fit the imputer on the compromised dataset.

        :param compromised: The compromised dataset.
        """
        self.models = {}

        for feature in self.categorical_features:
            if feature not in compromised.columns:
                continue
            train_data = compromised.dropna(subset=[feature])
            X_train = train_data.drop(columns=self.categorical_features, errors='ignore')
            y_train = train_data[feature]

            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)

            self.models[feature] = rf_classifier

    def transform(self, reconstructed):
        """
        Impute missing values in the reconstructed dataset.

        :param reconstructed: The reconstructed dataset.
        """
        imputed_data = reconstructed.copy()

        for feature in self.categorical_features:
            if feature not in imputed_data.columns:
                continue
            if imputed_data[feature].isnull().any():
                null_indices = imputed_data[imputed_data[feature].isnull()].index
                X_null = imputed_data.loc[null_indices].drop(columns=self.categorical_features, errors='ignore')

                imputed_data.loc[null_indices, feature] = self.models[feature].predict(X_null)

        return imputed_data

    def fit_transform(self, compromised, reconstructed):
        """
        Fit the imputer on the compromised dataset and then transform the reconstructed dataset.

        :param compromised: The compromised dataset.
        :param reconstructed: The reconstructed dataset.
        """
        self.fit(compromised)
        return self.transform(reconstructed)


def impute_missing_values(df, categorical_features, compromised_df: pd.DataFrame = None):
    """
    This function imputes the missing values in the dataframe

    :param df: The dataframe
    :param categorical_features: The categorical features
    :param compromised_df: An optional dataframe to use for imputing the missing values
    """
    if compromised_df is None:
        for feature in categorical_features:
            df[feature] = df[feature].fillna(df[feature].mode()[0])
            # try:
            #
            # except:
            #     pass
    else:
        imputer = CustomImputer(categorical_features)
        label_df = df['label']
        df = imputer.fit_transform(compromised_df, df.drop(columns=['label']))
        df = pd.concat([df, label_df], axis=1)

    return df
