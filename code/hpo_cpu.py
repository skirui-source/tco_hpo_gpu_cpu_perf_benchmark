import argparse
import glob
import os
import time

import dask
import optuna
import xgboost as xgb
from dask.distributed import Client, LocalCluster, wait
from dask_ml.model_selection import train_test_split
from optuna.samplers import RandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_cv_folds = 5

label_column = "ArrDel15"
feature_columns = [
    "Year",
    "Quarter",
    "Month",
    "DayOfWeek",
    "Flight_Number_Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "OriginCityMarketID",
    "DestCityMarketID",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "ArrDel15",
    "AirTime",
    "Distance",
]


def ingest_data():
    dataset = dask.dataframe.read_parquet(
            glob.glob("./data/*.parquet"),
            columns=feature_columns,
        ).repartition(npartitions=1000)
    return dataset


def preprocess_data(dataset, *, client, i_fold):
    dataset = dataset.dropna()
    train, test = train_test_split(dataset, random_state=i_fold, shuffle=True)
    X_train, y_train = train.drop(label_column, axis=1), train[label_column]
    X_test, y_test = test.drop(label_column, axis=1), test[label_column]
    X_train, y_train = X_train.astype("float32"), y_train.astype("int32")
    X_test, y_test = X_test.astype("float32"), y_test.astype("int32")

    X_train = X_train.persist()
    y_train = y_train.persist()
    X_test = X_test.persist()
    y_test = y_test.persist()

    wait([X_train, y_train, X_test, y_test])

    return X_train, y_train, X_test, y_test


def train_xgboost(trial, *, dataset, client):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 10.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 100, log=True),
        "verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "hist",
    }
    num_boost_round = trial.suggest_int("num_boost_round", 100, 500, step=10)

    cv_fold_scores = []
    for i_fold in range(n_cv_folds):
        X_train, y_train, X_test, y_test = preprocess_data(
            dataset, client=client, i_fold=i_fold
        )

        dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        xgboost_output = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=num_boost_round,
        )
        trained_model = xgboost_output["booster"]

        dtest = xgb.dask.DaskDMatrix(client, X_test)
        pred = xgb.dask.predict(client, trained_model, dtest) > 0.5
        pred = pred.astype("int32").compute()
        y_test = y_test.compute()
        score = accuracy_score(y_test, pred)
        cv_fold_scores.append(score)
    final_score = sum(cv_fold_scores) / len(cv_fold_scores)
    return final_score


def train_randomforest(trial, *, dataset, client):
    params = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=10),
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 1000, log=True),
        "n_jobs": -1,
    }

    cv_fold_scores = []
    for i_fold in range(n_cv_folds):
        X_train, y_train, X_test, y_test = preprocess_data(
            dataset, client=client, i_fold=i_fold
        )
        trained_model = RandomForestClassifier(**params)
        trained_model.fit(X_train, y_train)

        pred = trained_model.predict(X_test)
        y_test = y_test.compute()
        score = accuracy_score(y_test, pred)
        cv_fold_scores.append(score)
    final_score = sum(cv_fold_scores) / len(cv_fold_scores)
    return final_score


def main(args):
    tstart = time.perf_counter()

    study = optuna.create_study(
        sampler=RandomSampler(seed=args.seed), direction="maximize"
    )

    with LocalCluster(n_workers=os.cpu_count()) as cluster:
        with Client(cluster) as client:
            dataset = ingest_data()
            client.persist(dataset)
            if args.model_type == "XGBoost":
                study.optimize(
                    lambda trial: train_xgboost(trial, dataset=dataset, client=client),
                    n_trials=100,
                    n_jobs=1,
                )
            else:
                study.optimize(
                    lambda trial: train_randomforest(
                        trial, dataset=dataset, client=client
                    ),
                    n_trials=100,
                    n_jobs=1,
                )

    tend = time.perf_counter()
    print(f"Time elapsed: {tend - tstart} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["XGBoost", "RandomForest"]
    )
    parser.add_argument("--seed", required=False, type=int, default=1)
    args = parser.parse_args()
    main(args)
