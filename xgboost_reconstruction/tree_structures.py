from copy import deepcopy

import numpy as np
import pandas as pd
import pydotplus

import xgboost_reconstruction as xgb_rec


class Range:
    def __init__(self, min_val=None, max_val=None):
        self.min = min_val if min_val is not None else -np.inf
        self.max = max_val if max_val is not None else np.inf

    def update_range(self, min_val=None, max_val=None):  # update to a smaller range (more fine-grained)
        min_val = min_val if min_val is not None else -np.inf
        max_val = max_val if max_val is not None else np.inf
        self.min = min_val if self.min < min_val else self.min
        self.max = max_val if self.max > max_val else self.max

    def __str__(self):
        if self.min == -np.inf and self.max == np.inf:
            return ""
        return f"[{self.min}, {self.max}]"

    def __repr__(self):
        return self.__str__()


class TreeNode:
    def __init__(self, id=None, left=None, right=None, feature=None, threshold=None, value=None, gain=None,
                 H=None, G=None):
        self.id = id
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.leaf_value = value
        self.gain = gain
        self.H = H
        self.features_ranges = None
        self.G = G
        if threshold == 'nan' or feature == 'Leaf':
            self.is_leaf = True
        else:
            self.is_leaf = False

    def update_children(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"ID: {self.id}, Feature: {self.feature}, Threshold: {self.threshold}, Value: {self.leaf_value}, Gain: {self.gain}, G: {self.G}, H: {self.H}, is_leaf: {self.is_leaf}, {self.features_ranges if hasattr(self, 'features_ranges') else ''}"

    def may_contain(self, sample):
        assert self.is_leaf, "This function should be called on a leaf node"
        for feature_name, feature_range in sample.features.items():
            if feature_range.min >= self.features_ranges[feature_name].max or feature_range.max <= self.features_ranges[
                feature_name].min:
                return False
        return True


class Tree:
    def __init__(self, root=None, leaves: dict[int, TreeNode] = None):
        self.root = root
        self.leaves = leaves

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return str(self.root)

    def update_root(self, root):
        self.root = root

    def get_leaf(self, index) -> TreeNode:
        if index not in self.leaves:
            raise ValueError(f"Leaf with id {index} not found")
        return self.leaves[index]


# Tree factory from xgboost model
class TreeFactory:
    def __init__(self, tree_dfs: list[pd.DataFrame], feature_names: list[str], xgb_info: xgb_rec.XGBoostInfo):
        self.tree_dfs = tree_dfs
        self.features_names = feature_names
        self.xgb_info = xgb_info

    def build_trees(self) -> list[Tree]:
        return [build_tree(tree_df, self.features_names, self.xgb_info) for tree_df in self.tree_dfs]

def build_tree(tree: pd.DataFrame, features_names: list[str], xgb_info: xgb_rec.XGBoostInfo) -> Tree:
    root_df = tree.iloc[0]
    root = build_node(root_df, xgb_info)
    leaves = {}

    def traverse(node, node_df, tree, feature_ranges):
        if node.is_leaf:
            node.features_ranges = feature_ranges
            leaves[node.id] = node
            return

        try:
            left_df = tree[tree["ID"] == node_df["Yes"]].iloc[0]
            right_df = tree[tree["ID"] == node_df["No"]].iloc[0]

            range = feature_ranges[node.feature]
            left_range = deepcopy(feature_ranges)
            left_range[node.feature] = Range(min_val=range.min, max_val=node_df["Split"])
            right_range = deepcopy(feature_ranges)
            right_range[node.feature] = Range(min_val=node_df["Split"], max_val=range.max)

            left = build_node(left_df, xgb_info)
            right = build_node(right_df, xgb_info)

            node.update_children(left, right)

            traverse(left, left_df, tree, left_range)
            traverse(right, right_df, tree, right_range)
        except:
            node.is_leaf = True
            node.features_ranges = feature_ranges
            leaves[node.id] = node
            return

    feature_ranges = {name: Range() for name in features_names}
    traverse(root, root_df, tree, feature_ranges)

    return Tree(root, leaves)

def build_node(param: pd.Series, xgb_info: xgb_rec.XGBoostInfo) -> TreeNode:
    id = int(param["ID"].split('-')[-1])

    node = TreeNode(
        id=id,
        feature=param["Feature"],
        threshold=param["Split"],
        value=param["Gain"],
        gain=param["Gain"],
        H=param["Cover"]
    )

    if node.is_leaf:
        node.G = - (node.leaf_value / xgb_info.lr) * (node.H + xgb_info.lambda_)

    return node


class TreeVisualizer:

    def __init__(self, tree: Tree, feature_names: list):
        self.tree = tree
        self.feature_names = feature_names

    def plot_tree(self, name='tree.png'):
        """
        Plot the decision tree using graphviz and pydotplus

        :param name: the name of the image
        """

        def add_node(graph, node: TreeNode, depth=0):
            if node.is_leaf:
                graph.add_node(pydotplus.Node(node.id,
                                              label=f"Predict: {node.leaf_value}, ID: {node.id}",
                                              shape='box'))
            else:
                graph.add_node(pydotplus.Node(node.id,
                                              label=f"{node.feature} < {node.threshold}, ID: {node.id}",
                                              shape='ellipse'))
                add_node(graph, node.left, depth + 1)
                add_node(graph, node.right, depth + 1)
                if depth == 0:
                    graph.add_edge(pydotplus.Edge(node.id, node.left.id, label='True'))
                    graph.add_edge(pydotplus.Edge(node.id, node.right.id, label='False'))
                else:
                    graph.add_edge(pydotplus.Edge(node.id, node.left.id))
                    graph.add_edge(pydotplus.Edge(node.id, node.right.id))

        graph = pydotplus.Dot(graph_type='digraph', dpi=300)
        add_node(graph, self.tree.root)

        # use the graphviz to plot the tree
        graph.write_png('trees/' + name)
