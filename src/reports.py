#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Reports methods                                                             #
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


def generate_features_imp_report(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a model and training dataset, returns the features importance
    """
    return pd.DataFrame(
        {"features": df.columns.tolist(), "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)



def generate_IR_performance(model, df: pd.DataFrame) -> None:
    None