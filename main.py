#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Prepare class                                                               #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-21                                                           #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################


import pandas as pd
from trainpy import build_evaluate_model_reg_from_files
from preprocesspy import preprocess_from_files
from utilspy import load_yaml
from reports import generate_features_imp_report


if __name__ == "__main__":
    ## Loading configuration file
    configeval = load_yaml("configeval.yaml")

    ##Orchestrator variables
    buildpreprocess, buildeval = (
        configeval["configlobal"]["buildpreprocess"],
        configeval["configlobal"]["buildeval"],
    )

    ## IDS variables
    var_target, var_id, var_date = (
        configeval["configlobal"]["var_target"],
        configeval["configlobal"]["var_id"],
        configeval["configlobal"]["var_date"],
    )

    ## Rules on training and tetsing datasets
    list_date_train, list_date_test, list_var_to_save = (
        configeval["configlobal"]["datetrain"],
        configeval["configlobal"]["datetest"],
        configeval["configlobal"]["list_of_var_to_save"],
    )
    query_train = f"(year in {list_date_train})&({var_target}.notnull())"
    query_test = f"(year in {list_date_test})&({var_target}.notnull())"

    ## Variables to consider in the training
    list_of_tries = configeval["configlobal"]["ntry"]

    ## Check & Execute preprocess
    if buildpreprocess:
        input_pre_process, output_preprocess = (
            configeval["configpreprocess"]["input_df_path"],
            configeval["configpreprocess"]["output_df_path"],
        )
        preprocess_from_files(
            input_df_path=input_pre_process, output_df_path=output_preprocess
        )

    ##Check & Execute evaluation
    if buildeval:
        eval_input_df_path = configeval["configeval"]["input_df_path"]
        var_in_train = configeval["configeval"]["var_in_train"]
        rule_for_train = configeval["configeval"]["rule_for_train"]
        rule_for_test = configeval["configeval"]["rule_for_test"]
        output_directory = configeval["configeval"]["output_directory"]
        eval_test = configeval["configeval"]["eval_or_test"]
        if not rule_for_train:
            rule_for_train = query_train

        if not rule_for_test:
            rule_for_test = query_test

        if not var_in_train:
            for i, j in list_of_tries.items():
                build_evaluate_model_reg_from_files(
                    input_df_path=eval_input_df_path,
                    target_var=var_target,
                    list_of_var_to_save=list_var_to_save,
                    var_in_train=j,
                    rule_for_train=rule_for_train,
                    rule_for_test=rule_for_test,
                    output_directory=output_directory,
                    eval_test=eval_test
                )
