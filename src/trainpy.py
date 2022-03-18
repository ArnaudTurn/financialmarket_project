#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# train methods                                                               #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-21                                                            #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################

import random
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import IsolationForest
from utilspy import save_model, score_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from reports import generate_features_imp_report
from loaderpy import load_dataset
from utilspy import fill_missing_columns_on_df, detect_outliers_x, generate_folder


def build_train_test(
    df: pd.DataFrame, var_in: str, var_target: str, list_of_var_to_save: list
) -> set:
    """
    Given a Dataset the goal is to build a train dataset and a test dataset
    """
    rand_num = random.randint(0, 1000)
    df_copy = df.copy()
    x_train, x_test, y_train, y_test, comp_train, comp_test = train_test_split(
        pd.get_dummies(df[var_in]),
        df[var_target],
        df[list_of_var_to_save],
        random_state=rand_num,
    )

    # Filling missing variables
    x_test = fill_missing_columns_on_df(from_df=x_train, on_df=x_test)

    return x_train, x_test, y_train, y_test, comp_train, comp_test


def build_train_test_based(
    df_for_train: pd.DataFrame,
    df_for_test: pd.DataFrame,
    var_in: str,
    var_target,
    list_of_var_to_save: list,
) -> set:
    """
    Given a dataset for train and a dataset for test, this function will format correctly the table to use
    """
    df_train, df_test = df_for_train.copy(), df_for_test.copy()
    dim_x_test = df_test.shape[0]

    if var_target not in df_test.columns.tolist():
        y_test = -999999 * np.ones((dim_x_test,), dtype=np.int16)
    else:
        y_test = df_test[var_target]

    x_train, x_test, y_train, comp_train, comp_test = (
        pd.get_dummies(df_train[var_in]),
        pd.get_dummies(df_test[var_in]),
        df_train[var_target],
        df_train[list_of_var_to_save],
        df_test[list_of_var_to_save],
    )

    # Filling missing variables
    x_test = fill_missing_columns_on_df(from_df=x_train, on_df=x_test)

    return x_train, x_test, y_train, y_test, comp_train, comp_test


def build_test_model_bin(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    comp_train: pd.DataFrame,
    comp_test: pd.DataFrame,
    rm_outliers: bool = False,
) -> dict:
    """
    Given the output of the function build_train_test* this function will assess the quality of a binary model.
    It has been implemented in a course of small test.
    """
    x_train, x_test = x_train.copy(), x_test.copy()

    if rm_outliers:
        maskoutliers = detect_outliers_x(x_train)
        x_train, y_train, comp_train = (
            x_train.loc[maskoutliers, :].copy(),
            y_train[maskoutliers],
            comp_train.loc[maskoutliers, :].copy(),
        )

    var_comp = comp_train.columns.tolist()
    # Create model and predict
    model_ = model(n_estimators=300)
    model_.fit(x_train, y_train)
    prediction_arr = model_.predict_proba(x_test)[:, 1]

    # Add necessary variables for IR scoring
    x_test[var_comp] = comp_test
    x_train[var_comp] = comp_train

    # Create validation dataset with prediction
    x_test = x_test.assign(prediction=prediction_arr)
    # Create statistical indicator
    accuracyscore, precisionscore, rocscore, confusionmatrix = (
        accuracy_score(y_test, prediction_arr > 0.5),
        precision_score(y_test, prediction_arr > 0.5),
        roc_auc_score(y_test, prediction_arr),
        confusion_matrix(y_test, prediction_arr > 0.5),
    )

    # Generate output
    output_names = [
        "model",
        "prediction",
        "accuracyscore",
        "precisionscore",
        "rocscore",
        "confusion_matrix",
    ]
    output_obj = [
        model_,
        prediction_arr,
        accuracyscore,
        precisionscore,
        rocscore,
        confusionmatrix,
    ]
    return dict(zip(output_names, output_obj))


def build_test_model_reg(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    comp_train: pd.DataFrame,
    comp_test: pd.DataFrame,
    rm_outliers: bool = False,
) -> dict:
    """
    Given the output of the function build_train_test* this function will assess the quality of a regression model.
    """

    x_train, x_test = x_train.copy(), x_test.copy()

    if rm_outliers:
        maskoutliers = detect_outliers_x(x_train)
        x_train, y_train, comp_train = (
            x_train.loc[maskoutliers, :].copy(),
            y_train[maskoutliers],
            comp_train.loc[maskoutliers, :].copy(),
        )

    var_comp = comp_train.columns.tolist()
    # Create model and predict
    try:
        model_ = model(n_estimators=300, reg_lambda=0.5)
    except:
        model_ = model()

    model_.fit(x_train, y_train)
    prediction_arr = model_.predict(x_test)

    # Add necessary variables for IR scoring
    x_test[var_comp] = comp_test
    x_train[var_comp] = comp_train

    # Create validation dataset with prediction
    x_test = x_test.assign(prediction=prediction_arr)

    # Create statistical indicator
    r2score, mae, mse, corr_coef = (
        r2_score(y_test, prediction_arr),
        mean_absolute_error(y_test, prediction_arr),
        mean_squared_error(y_test, prediction_arr),
        np.corrcoef(y_test, prediction_arr)[0, 1],
    )

    # Create financial indicator
    ir_score_train, ir_test_score, ir_valid_score = (
        score_model(df=x_train),
        score_model(df=x_test),
        score_model(df=x_test, var_target="prediction"),
    )

    # Generate output
    output_names = [
        "model",
        "prediction",
        "r2score",
        "mae",
        "mse",
        "corrcoef",
        "IR_TRAIN",
        "IR_TEST",
        "IR_VALID",
    ]
    output_obj = [
        model_,
        prediction_arr,
        r2score,
        mae,
        mse,
        corr_coef,
        ir_score_train,
        ir_test_score,
        ir_valid_score,
    ]
    return dict(zip(output_names, output_obj))


def build_evaluate_model_reg(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    comp_train: pd.DataFrame,
    comp_test: pd.DataFrame,
    rm_outliers: bool = False,
    ir_score: bool = True,
) -> dict:
    """
    Given the output of the function build_train_test* this function will return the newly created prediction
    """
    var_comp = comp_train.columns.tolist()
    var_train = x_train.columns.tolist()

    x_train, x_test = x_train.copy(), x_test.copy()

    if rm_outliers:
        maskoutliers = detect_outliers_x(x_train)
        x_train, y_train, comp_train = (
            x_train.loc[maskoutliers, :].copy(),
            y_train[maskoutliers],
            comp_train.loc[maskoutliers, :].copy(),
        )

    # Filling missing variables
    x_test = fill_missing_columns_on_df(from_df=x_train, on_df=x_test)

    # Create model and predict
    model_ = model(n_estimators=300)
    model_.fit(x_train, y_train)
    prediction_arr = model_.predict(x_test[var_train])

    # Add necessary variables for IR scoring
    x_test[var_comp] = comp_test
    x_train[var_comp] = comp_train

    # Create validation dataset with prediction
    x_test = x_test.assign(prediction=prediction_arr)

    # Create statistical indicator
    # r2score, mae, mse = r2_score(y_test, prediction_arr), mean_absolute_error(y_test, prediction_arr), mean_squared_error(y_test, prediction_arr)

    # Create financial indicator
    if ir_score:
        ir_score_train, ir_valid_score = score_model(df=x_train), score_model(
            df=x_test, var_target="prediction"
        )
    else:
        ir_score_train, ir_valid_score = None, None

    # Generate output
    output_names = ["model", "prediction", "IR_TRAIN", "IR_VALID", "xtest"]
    output_obj = [model_, prediction_arr, ir_score_train, ir_valid_score, x_test]
    return dict(zip(output_names, output_obj))


def train_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    comp_train: pd.DataFrame,
    var_target: str = "T1_RETURN",
    rm_outliers: bool = False,
    ir_score: bool = True,
):
    """
    Given the train dataset and its target, build a model
    """
    var_comp = comp_train.columns.tolist()
    var_train = x_train.columns.tolist()

    x_train = x_train.copy()

    if rm_outliers:
        maskoutliers = detect_outliers_x(x_train)
        x_train, y_train, comp_train = (
            x_train.loc[maskoutliers, :].copy(),
            y_train[maskoutliers],
            comp_train.loc[maskoutliers, :].copy(),
        )

    # Create model and predict
    model_ = model(n_estimators=300)
    model_.fit(x_train, y_train)

    # Add necessary variables for IR scoring
    x_train[var_comp] = comp_train

    # Create financial indicator
    if ir_score:
        ir_score_train = score_model(df=x_train, var_target=var_target)
    else:
        ir_score_train = None

    # Generate output
    output_names = ["model", "IR_TRAIN"]
    output_obj = [model_, ir_score_train]
    return dict(zip(output_names, output_obj))


def build_evaluate_model_reg_from_files(
    input_df_path: str,
    target_var: str,
    list_of_var_to_save: list,
    var_in_train: list,
    rule_for_train: str = None,
    rule_for_test: str = None,
    output_directory: str = "output",
    eval_test:str="test"
) -> None:
    """
    Given a dataset path create a model and evaluate the performance of the model
    """
    path_output = generate_folder(
        output_path=output_directory, folder_name="model_test"
    )
    features_importances_path = f"{path_output}\\features_importances.csv"
    model_path = f"{path_output}\model_pkg.pkl"

    df_train = load_dataset(input_df_path)
    if rule_for_train:
        df_train_select = df_train.query(rule_for_train, engine="python")

    if rule_for_test:
        df_test_select = df_train.query(rule_for_test, engine="python")

    x_train, x_test, y_train, y_test, comp_train, comp_test = build_train_test_based(
        df_for_train=df_train_select,
        df_for_test=df_test_select,
        var_in=var_in_train,
        var_target=target_var,
        list_of_var_to_save=list_of_var_to_save,
    )
    
    if eval_test == "eval":
        output_result = build_evaluate_model_reg(LGBMRegressor,x_train=x_train, x_test=x_test, y_train=y_train, comp_train = comp_train, comp_test=comp_test)
    
    elif eval_test == "test":
        output_result = build_test_model_reg(LGBMRegressor,x_train=x_train, x_test=x_test, y_train=y_train, y_test = y_test,comp_train = comp_train, comp_test=comp_test)
    
    features_imp_df = generate_features_imp_report(model=output_result["model"], df = x_train)

    features_imp_df.to_csv(features_importances_path)
    save_model(output_result,filepath=model_path)
