#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# utils methods                                                               #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-21                                                            #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
from datetime import datetime
import joblib
import yaml

# This function is used to score the model. in its current state the model produce an Information Ratio equals to 1.69
def score_model(
    df: pd.DataFrame,
    var_id: str = "RP_ENTITY_ID",
    var_target: str = "T1_RETURN",
    var_SG90: str = "GROUP_E_ALL_SG90",
    var_date: str = "DATE",
) -> float:
    """
    Given the original dataset build the score
    """
    df_copy = (
        df[[var_date, var_id, var_SG90, var_target]]
        .dropna(axis=0)
        .drop_duplicates()
        .copy()
    )
    res_df = (
        df_copy.groupby(var_date)
        .apply(
            lambda x: pd.Series(
                {"AVGRET": np.mean(np.sign(x[var_SG90]) * x[var_target])}
            )
        )
        .reset_index()
    )
    # AnnualizedReturn
    AnnualizedReturn = np.mean(np.log(res_df.AVGRET + 1)) * 252
    # AnnualizedVolatility
    AnnualizedVolatility = np.sqrt(np.var(np.log(res_df.AVGRET + 1))) * np.sqrt(252)
    # Information Ratio
    InformationRatio = AnnualizedReturn / AnnualizedVolatility
    return InformationRatio


def keep_minus_one(arr: np.array):
    """
    Given an array add all values -1
    """
    new_arr = arr - 1
    final_arr = np.unique(np.hstack((new_arr, arr)))
    return final_arr


def fill_missing_columns_on_df(
    from_df: pd.DataFrame, on_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Give 2 datasets, fill missing variables with 0 form on to another
    """
    list_from_col = from_df.columns.tolist()
    list_on_col = on_df.columns.tolist()
    list_col_to_add = list(set(list_from_col) - set(list_on_col))
    if list_col_to_add:
        on_df[list_col_to_add] = 0
    return on_df


def compute_RSI(df: pd.DataFrame, var: str, window: int, min_periods: int = 0):
    """
    Given a DataFrame this function compute the financial indicator RSI
    """
    delta = df[var].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Exponential mooving average
    _gain = up.ewm(com=(window - 1), min_periods=min_periods).mean()
    _loss = down.abs().ewm(com=(window - 1), min_periods=min_periods).mean()

    # RS computation
    RS = _gain / _loss
    # add RSI feature
    df[f"RSI_{var}_{window}"] = 100 - (100 / (1 + RS))
    return df


def compute_indicators(df: pd.DataFrame, var: list, windows: int, min_periods: int = 0):
    """
    Given a DataFrame this function compute rolling indicators such as moving average, drawdown and other simple financial indicators
    """

    df_feat = (
        df[var]
        .rolling(windows, min_periods=min_periods)
        .agg(
            {
                f"mean_{var}_{windows}": np.mean,
                f"std_{var}_{windows}": np.std,
                f"UPPER_BB_{var}_{windows}": lambda x: np.mean(x) + np.std(x) * 2,
                f"LOWER_BB_{var}_{windows}": lambda x: np.mean(x) - np.std(x) * 2,
                f"Drawdown_min_{var}_{windows}": lambda x: (max(x) - min(x)) / min(x),
                f"Drawdown_max_{var}_{windows}": lambda x: (max(x) - min(x)) / max(x),
                f"Max_decrease{var}_{windows}": lambda x: (max(x) - np.array(x)[-1])
                / max(x),
                f"Min_decrease{var}_{windows}": lambda x: (min(x) - np.array(x)[-1])
                / min(x),
            }
        )
    )
    return df_feat


def detect_outliers_x(df: pd.DataFrame) -> np.array:
    """
    Given a dataframe, the function search for outliers and return an array used to select the non outliers
    """
    isof = IsolationForest(contamination=0.1)
    yhat = isof.fit_predict(df.fillna(0))
    return yhat != -1


def check_exist(folder_path: str):
    if os.path.isdir(folder_path):
        None
    else:
        os.mkdir(folder_path)


def get_unique_date():
    now = datetime.now()
    now = str(now).replace("-", "").replace(":", "").replace(" ", "_").split(".")[0]
    return now


def save_model(object, filepath: str):
    joblib.dump(
        object,
        filepath,
    )


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            load_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return load_yaml


def generate_folder(output_path: str, folder_name: str) -> str:
    unique_date = get_unique_date()
    unique_folder_path = f"{output_path}\{folder_name}_{unique_date}"
    check_exist(unique_folder_path)
    return unique_folder_path
