import os
import pickle
from logging import INFO

import flwr as fl
import xgboost as xgb
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
from sklearn.metrics import f1_score

MALICIOUS_CLIENT_ID = 0

current_dir = os.path.dirname(__file__)


class XgbClient(fl.client.Client):
    def __init__(
            self,
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
            node_id,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.node_id = node_id

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round: bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )

            # save the model
            path = os.path.join(current_dir, "first_round_model")
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/model_{self.node_id}.pkl", "wb") as f:
                pickle.dump(bst, f)
        else:
            bst = xgb.Booster(params=self.params)

            for item in ins.parameters.tensors:  # len is always 1
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            if self.node_id == MALICIOUS_CLIENT_ID:
                # store the global model in a directory
                path = os.path.join(current_dir, "global_model")
                os.makedirs(path, exist_ok=True)
                with open(f"{path}/round_{global_round}.pkl", "wb") as f:
                    pickle.dump(global_model, f)

            # Local training
            bst = self._local_boost(bst)

        log(INFO, f"Local training completed at round {global_round}")
        log(INFO, f"Config: {ins.config}")

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={"cid": self.node_id},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        if self.node_id == MALICIOUS_CLIENT_ID:
            global_round = ins.config["global_round"]
            path = os.path.join(current_dir, "global_model")
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/round_{global_round}.pkl", "wb") as f:
                pickle.dump(para_b, f)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        f1 = f1_score(
            self.valid_dmatrix.get_label(),
            bst.predict(self.valid_dmatrix) > 0.5,
        )

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")
        log(INFO, f"F1 = {f1} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc, "F1": f1},
        )
