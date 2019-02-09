import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC


def test_models(models_with_names, X, y, predict_fun, n_splits=5):
    for clf, name in models_with_names:
        print('{0: <20}'.format(name), end='')
        aucs = []
        skf = StratifiedKFold(n_splits=n_splits, random_state=42)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = predict_fun(clf, X_test)
            auc = metrics.roc_auc_score(y_test, y_pred)
            aucs.append(auc)
        auc = np.mean(aucs)
        print("AUC:", auc)


def test_all_sklearn(X, y):
    scaler = MinMaxScaler()
    print("\nCurrent scaler", scaler, end="\n\n")
    X_scaled = scaler.fit_transform(X)
    models_with_names = ((MultinomialNB(), "MultinomialNB"),
                         (BernoulliNB(), "BernoulliNB"))

    test_models(models_with_names, X_scaled, y, lambda clf, x: clf.predict_proba(x)[:, 1])

    for scaler in [MinMaxScaler(), StandardScaler()]:
        print("\nCurrent scaler", scaler, end="\n\n")

        X_scaled = scaler.fit_transform(X)
        models_with_names = ((RidgeClassifier(), "Ridge Classifier"),
                             (Perceptron(), "Perceptron"),
                             (PassiveAggressiveClassifier(), "Passive-Aggressive"),
                             (SGDClassifier(), "LinearSVC"),
                             (SGDClassifier(), "SGDClassifier"),
                             (LinearSVC(), "LinearSVC"))

        test_models(models_with_names, X_scaled, y, lambda clf, x: clf.decision_function(x))

        models_with_names = (
            (xgb.XGBClassifier(), "XGBoost")
            , (RandomForestClassifier(), "Random forest"))

        test_models(models_with_names, X_scaled, y, lambda clf, x: clf.predict_proba(x)[:, 1], n_splits=2)


def main():
    FEATURE_PATH = "../features/feaures_pyAA_all_10ms.csv"
    nrows = 30000
    feature_df = pd.read_csv(FEATURE_PATH, nrows=nrows)
    nfeatures = feature_df.shape[1] - 2
    only_features = feature_df.iloc[:, :nfeatures]
    X = only_features.as_matrix()
    y = feature_df.IS_LAUGHTER.as_matrix()
    test_all_sklearn(X, y)

if __name__ == '__main__':
    main()
