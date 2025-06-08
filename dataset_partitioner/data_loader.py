import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_stroke(path):
    test_path = f"{path.replace('.csv', '')}-test.csv"
    train_path = f"{path.replace('.csv', '')}-train.csv"

    def clean_stroke(df):
        cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                        'smoking_status']

        cat_df = df[cat_features]

        cat_dummy = pd.get_dummies(cat_df)

        num_features = ['age', 'avg_glucose_level', 'bmi']

        num_df = df[num_features]

        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        num_df = pd.DataFrame(imp_median.fit_transform(num_df), columns=num_df.columns)

        df_clean = pd.concat([num_df, cat_dummy, df['stroke']], axis=1).reset_index(drop=True)

        return df_clean

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train_clean = clean_stroke(df_train)
    df_test_clean = clean_stroke(df_test)

    X_train = df_train_clean.drop(columns=['stroke'])  # Features
    y_train = df_train_clean['stroke']  # Target variable

    X_test = df_test_clean.drop(columns=['stroke'])  # Features
    y_test = df_test_clean['stroke']  # Target variable

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    return X_train, y_train, X_test, y_test

def load_diabetes(path):
    df = pd.read_csv(path)
    # already preprocessed
    X = df.drop(columns=['Outcome'])  # Features
    y = df['Outcome']  # Target variable

    # Perform stratified split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    return x_train, y_train, x_test, y_test