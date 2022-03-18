#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Prepare methods                                                             #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 22021-12-21                                                           #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################

import pandas as pd
import numpy as np
import argparse
from loaderpy import load_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilspy import compute_indicators, compute_RSI, keep_minus_one
import re


def preprocess_dataset(
    df: pd.DataFrame,
    var_date: str = "DATE",
    var_id: str = "RP_ENTITY_ID",
    var_target: str = "T1_RETURN",
) -> pd.DataFrame:
    """
    Given the original dataset remove all duplicates while minimizing the number of missing values
    """
    df_copy = df.copy()
    ## Filling missing values
    # First time we go backward
    new_data_test = (
        df_copy.groupby([var_date, var_id])
        .bfill()
        .assign(DATE=df_copy[var_date], RP_ENTITY_ID=df_copy[var_id])
        .copy()
    )

    # Second time we go forward
    new_data_test2 = (
        new_data_test.groupby([var_date, var_id])
        .ffill()
        .assign(DATE=df_copy[var_date], RP_ENTITY_ID=df_copy[var_id])
        .copy()
    )

    ## Removing duplicates
    # Final dataset by removing the duplicates rows
    final_data_set = new_data_test2.drop_duplicates(
        subset=[var_date, var_id, var_target], keep="first"
    ).copy()

    return final_data_set


def fe_dataset(
    df: pd.DataFrame, var_date: str = "DATE", var_id: str = "RP_ENTITY_ID"
) -> pd.DataFrame:
    """
    Given a transformed dataset build all the necessary features including : past returns, financial indicators
    """
    df_copy = df.copy()
    # Variables related to the date
    final_data_set = df_copy.assign(
        real_date=lambda x: pd.to_datetime(x[var_date], format="%Y-%m-%d")
    )
    final_data_set = final_data_set.assign(
        delta_date=lambda x: x.groupby(var_id).real_date.diff().apply(lambda y: y.days),
        year=lambda x: x[var_date].apply(lambda y: y[:4]),
        day=lambda x: x["real_date"].apply(lambda y: y.strftime("%A")),
        month=lambda x: x[var_date].apply(lambda y: y[5:7]),
    )

    # Variables related to the returns : Pt/Pt-2, Pt/Pt-3, Pt/Pt-4
    final_data_set = final_data_set.assign(
        T0_RATIO=lambda x: x["T0_RETURN"].apply(lambda y: (y + 1)),
        LOG_T0_RETURN=lambda x: x["T0_RATIO"].apply(lambda y: np.log(y)),
        T0_RETURN_1=lambda x: x.groupby(var_id)["T0_RETURN"].shift(1),
        T0_RETURN_2=lambda x: x.groupby(var_id)["T0_RETURN"].shift(2),
        T0_RETURN_3=lambda x: x.groupby(var_id)["T0_RETURN"].shift(3),
        T0_RETURN_4=lambda x: x.groupby(var_id)["T0_RETURN"].shift(4),
        T0_RETURN_5=lambda x: x.groupby(var_id)["T0_RETURN"].shift(5),
        RATIO_0_1=lambda x: x["T0_RETURN"] * x["T0_RETURN_1"],
        RATIO_0_2=lambda x: x["T0_RETURN"] * x["T0_RETURN_1"] * x["T0_RETURN_2"],
        RATIO_0_3=lambda x: x["T0_RETURN"]
        * x["T0_RETURN_1"]
        * x["T0_RETURN_2"]
        * x["T0_RETURN_3"],
        RATIO_0_4=lambda x: x["T0_RETURN"]
        * x["T0_RETURN_1"]
        * x["T0_RETURN_2"]
        * x["T0_RETURN_3"]
        * x["T0_RETURN_4"],
        RATIO_0_5=lambda x: x["T0_RETURN"]
        * x["T0_RETURN_1"]
        * x["T0_RETURN_2"]
        * x["T0_RETURN_3"]
        * x["T0_RETURN_4"]
        * x["T0_RETURN_5"],
        PRICE_EST=lambda x: 100 * np.exp(x.groupby(var_id)["LOG_T0_RETURN"].cumsum()),
    )

    # Adding all the other return related variables
    for i in range(5, 30):
        final_data_set[f"T0_RETURN_{i}"] = final_data_set.groupby(var_id)[
            "T0_RETURN"
        ].shift(i)

    # Financial indicators starter pack
    # (this one is slow - to be reworked)
    feat_price_indicators = final_data_set.groupby(var_id).apply(
        lambda x: compute_indicators(df=x, var="PRICE_EST", windows=14)
    )
    final_data_set[feat_price_indicators.columns.tolist()] = feat_price_indicators

    final_data_set["RSI_PRICE_EST_14"] = final_data_set.groupby(var_id).apply(
        lambda x: compute_RSI(df=x, var="PRICE_EST", window=14)
    )["RSI_PRICE_EST_14"]
    final_data_set["RSI_PRICE_EST_28"] = final_data_set.groupby(var_id).apply(
        lambda x: compute_RSI(df=x, var="PRICE_EST", window=28)
    )["RSI_PRICE_EST_28"]

    # Few indicators regarding the bearish/bullish of the last days
    final_data_set = final_data_set.assign(
        slow_invest_pace=lambda x: 1
        * (
            (x["T0_RETURN"] < x["T0_RETURN_1"])
            & (x["T0_RETURN_1"] < x["T0_RETURN_2"])
            & (x["T0_RETURN_2"] < x["T0_RETURN_3"])
            & (x["T0_RETURN_3"] < x["T0_RETURN_4"])
            & (x["T0_RETURN_4"] < x["T0_RETURN_5"])
        ),
        downtrend_3d=lambda x: 1
        * ((x["T0_RETURN"] < 0) & (x["T0_RETURN_1"] < 0) & (x["T0_RETURN_2"] < 0)),
        uptrend_3d=lambda x: 1
        * ((x["T0_RETURN"] > 0) & (x["T0_RETURN_1"] > 0) & (x["T0_RETURN_2"] > 0)),
    )

    # Complementary indicators
    final_data_set = final_data_set.assign(
        min_T0_RETURN_5d=lambda x: x[
            [
                "T0_RETURN",
                "T0_RETURN_1",
                "T0_RETURN_2",
                "T0_RETURN_3",
                "T0_RETURN_4",
                "T0_RETURN_5",
            ]
        ].min(axis=1),
        max_T0_RETURN_5d=lambda x: x[
            [
                "T0_RETURN",
                "T0_RETURN_1",
                "T0_RETURN_2",
                "T0_RETURN_3",
                "T0_RETURN_4",
                "T0_RETURN_5",
            ]
        ].max(axis=1),
        mean_T0_RETURN_5d=lambda x: x[
            [
                "T0_RETURN",
                "T0_RETURN_1",
                "T0_RETURN_2",
                "T0_RETURN_3",
                "T0_RETURN_4",
                "T0_RETURN_5",
            ]
        ].mean(axis=1),
        var_T0_RETURN_5d=lambda x: x[
            [
                "T0_RETURN",
                "T0_RETURN_1",
                "T0_RETURN_2",
                "T0_RETURN_3",
                "T0_RETURN_4",
                "T0_RETURN_5",
            ]
        ].var(axis=1),
    )

    return final_data_set


def preprocess_from_files(
    input_df_path: str,
    output_df_path: str,
    var_target: str = "T1_RETURN",
    var_id: str = "RP_ENTITY_ID",
    var_date: str = "DATE",
) -> None:
    """
    Given a path of data transform the data before building the models
    """
    final_df = load_dataset(path_df=input_df_path)
    final_df_preprocessed = preprocess_dataset(
        df=final_df, var_date=var_date, var_id=var_id, var_target=var_target
    )
    final_df_preprocessed = fe_dataset(
        final_df_preprocessed, var_date=var_date, var_id=var_id
    )
    final_df_preprocessed.to_csv(output_df_path, sep=",", index=False)
