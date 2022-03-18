#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# loader method                                                               #
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
import os
import datetime


def load_dataset(path_df: str) -> pd.DataFrame:
    """
    Given dataset path load the data
    """
    dataset = pd.read_csv(
        filepath_or_buffer=path_df,
        sep=",",
        low_memory=False,
        error_bad_lines=False,
    )
    return dataset
