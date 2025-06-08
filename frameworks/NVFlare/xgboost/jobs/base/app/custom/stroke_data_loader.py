# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


class StrokeDataLoader(XGBDataLoader):
    def __init__(self, data_split_filename):
        """Reads HIGGS dataset and return XGB data matrix.

        Args:
            data_split_filename: file name to data splits
        """
        self.data_split_filename = data_split_filename

    def load_data(self, client_id: str):
        client_num = int(client_id.split("-")[1]) - 1
        with open(self.data_split_filename, "r") as file:
            data_split = json.load(file)

        data_path = data_split["data_path"]

        print(f"Bro i'm {client_num}: {data_path}")

        x_train = np.load(f"{data_path}/client_{client_num}/train/x_train.npy")
        y_train = np.load(f"{data_path}/client_{client_num}/train/y_train.npy")
        x_val = np.load(f"{data_path}/client_{client_num}/valid/x_valid.npy")
        y_val = np.load(f"{data_path}/client_{client_num}/valid/y_valid.npy")

        fn = ['age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
              'Residence_type', 'smoking_status', 'stroke']
        fn = [f for f in fn if f != 'stroke']

        df_train = pd.DataFrame(x_train, columns=fn)
        df_val = pd.DataFrame(x_val, columns=fn)
        df_train_labels = pd.DataFrame(y_train, columns=["stroke"])
        df_val_labels = pd.DataFrame(y_val, columns=["stroke"])

        # Create DMatrix
        train_dmatrix = xgb.DMatrix(df_train, label=df_train_labels)
        valid_dmatrix = xgb.DMatrix(df_val, label=df_val_labels)

        return train_dmatrix, valid_dmatrix
