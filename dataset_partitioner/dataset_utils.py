from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def kl_divergence(y_train_loc, y_train_glob):
    p = np.sum(y_train_loc, axis=0) / len(y_train_loc)
    q = np.sum(y_train_glob, axis=0) / len(y_train_glob)

    kl_divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))  # entropy(p, q)
    return kl_divergence


def w_distance(y_train_loc, y_test_glob):
    return wasserstein_distance(np.argmax(y_train_loc, axis=1), np.argmax(y_test_glob, axis=1))


def make_seed():
    # current_time = datetime.now()
    # seconds = (current_time - current_time.min).seconds
    # rounding = (seconds + 7.5) // 15 * 15
    # rounded = current_time + timedelta(seconds=rounding - seconds)
    # rounded = rounded.replace(microsecond=0)
    # seed = rounded.now().year + rounded.now().month + rounded.now().hour + rounded.now().minute
    # print('Seed:', seed)
    seed = 42
    return seed


def get_config(data, path):
    if data == 'stroke':
        return {
            'features': ['age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension', 'heart_disease', 'ever_married',
                         'work_type', 'Residence_type', 'smoking_status'],
            'numerical_features': ['age', 'avg_glucose_level', 'bmi'],
            'categorical_features': ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                                     'Residence_type', 'smoking_status'],
            'input_shape': (10,),
            'num_classes': 2,
            'num_features': 10,
            'data': 'stroke',
            'label': 'stroke'
        }
    elif data == 'diabetes':
        df = pd.read_csv(path)
        features = df.columns.tolist()
        features.remove('Outcome')
        return {
            'features': features,
            'numerical_features': features,
            'categorical_features': [],
            'input_shape': (8,),
            'num_classes': 2,
            'num_features': 8,
            'data': 'diabetes',
            'label': 'Outcome'
        }
    else:
        print("No other dataset implemented yet")
        exit()
