import json
import tempfile

import numpy as np
import xgboost as xgb


class XGBoostInfo:
    def __init__(self, model: xgb.XGBClassifier = None, config: dict = None):
        if config is not None:
            self.config = config
        elif model is not None:
            self.config = json.loads(model.get_booster().save_config())
        else:
            raise ValueError("Either model or config must be provided")

    @property
    def num_boost_round(self):
        return int(self.config["learner"]["gradient_booster"]["gbtree_model_param"]["num_trees"])

    @property
    def base_score(self):
        return float(self.config["learner"]["learner_model_param"]["base_score"])

    @property
    def log_odds_base_score(self):
        return np.log(self.base_score / (1 - self.base_score))

    @property
    def lambda_(self):
        return float(self.config["learner"]["gradient_booster"]["tree_train_param"]["lambda"])

    @property
    def gamma(self):
        return float(self.config["learner"]["gradient_booster"]["tree_train_param"]["gamma"])

    @property
    def lr(self):
        return float(self.config["learner"]["gradient_booster"]["tree_train_param"]["learning_rate"])

    def __str__(self):
        return f"base_score: {self.base_score}, lambda: {self.lambda_}, gamma: {self.gamma}, learning_rate: {self.lr}"

    def __repr__(self):
        return str(self)


# Create a FedTreeXGBoostInfo class that inherits from XGBoostInfo and overrides all the properties and methods
class GenericXGBoostInfo(XGBoostInfo):
    def __init__(self, base_score: float = 0.5, lambda_: float = 0, gamma: float = 0, lr: float = 1):
        self.base_score = base_score
        self.lambda_ = lambda_
        self.gamma = gamma
        self.lr = lr

    @property
    def base_score(self):
        return float(self._base_score)

    @property
    def log_odds_base_score(self):
        return np.log(self.base_score / (1 - self.base_score))

    @property
    def lambda_(self):
        return float(self._lambda_)

    @property
    def gamma(self):
        return float(self._gamma)

    @property
    def lr(self):
        return float(self._lr)

    @base_score.setter
    def base_score(self, value):
        self._base_score = value

    @lambda_.setter
    def lambda_(self, value):
        self._lambda_ = value

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @lr.setter
    def lr(self, value):
        self._lr = value

    def __str__(self):
        return f"base_score: {self.base_score}, lambda: {self.lambda_}, gamma: {self.gamma}, learning_rate: {self.lr}"

    def __repr__(self):
        return str(self)


def load_xgb_model_from_boosters(path_prefix, n_estimators):
    """
    This function loads an xgboost model from a list of saved boosters model by fixing the number of estimators.
    """

    # print(f"Loading model from {path_prefix} with {n_estimators} estimators")

    # parse the first booster to get the parameters
    path = path_prefix + 'booster_0.json'
    with open(path, 'r') as f:
        model = json.load(f)

    # adjust iteration_indptr
    model['learner']['gradient_booster']['model']['iteration_indptr'] = [i for i in range(0, n_estimators + 1)]

    # adjust tree_info
    model['learner']['gradient_booster']['model']['tree_info'] = [0 for _ in range(n_estimators)]

    # adjust num_trees
    model['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'] = f"{n_estimators}"

    # adjust attributes (scikit-learn)
    model['learner']['attributes']['scikit_learn'] = "{\"_estimator_type\": \"classifier\"}"

    # model['learner']['gradient_booster']['model']['trees']
    for i in range(1, n_estimators):
        path = path_prefix + 'booster_{}.json'.format(i)
        with open(path, 'r') as f:
            booster = json.load(f)

        # adjust the id of the trees
        booster['learner']['gradient_booster']['model']['trees'][0]['id'] = i

        # append the trees
        model['learner']['gradient_booster']['model']['trees'] += \
            booster['learner']['gradient_booster']['model']['trees']

    return model


class RecoverXGBClassifier:
    def __init__(self, boosters: list[xgb.Booster]):
        self.boosters = boosters

    def recover_xgb(self) -> xgb.XGBClassifier:
        """
        This function recovers an xgboost model from a list of XGBoost boosters representing individual trees.
        """
        # temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            n_estimators = len(self.boosters)

            # save the boosters
            for idx, booster in enumerate(self.boosters):
                booster.save_model(f'{tmpdirname}/booster_{idx}.json')

            # load the model
            model_json = load_xgb_model_from_boosters(tmpdirname + '/', n_estimators)

            model = xgb.XGBClassifier()

            # save to a temporary file
            with open(f'{tmpdirname}/model_scratch.json', 'w') as f:
                json.dump(model_json, f)

            # load the model
            model.load_model(f'{tmpdirname}/model_scratch.json')

            return model
