import time
from configparser import ConfigParser

import pandas as pd
from memory_profiler import memory_usage

from Causality_analysis_modules import utils
from Workflow.inference import Inference
from Workflow.preprocessing import Preprocessing


def execute(keyword, config, dataset_name, output_name, percentage_in_CI, percentage_of_CI):
    reader = ConfigParser()
    reader.read(config)
    activities_col_name = eval(reader['DATASET']['activities_col'])[0]
    caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
    timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
    transition_names = eval(reader['INFERENCE']['transitions'])
    component_set = eval(reader['STRUCTURE_SETTING']['components_set'])
    dynamic = eval(reader['STRUCTURE_SETTING']['dynamics_set'])
    inference_tool = Inference(activities_col_name, caseIDs_col_name, timestamps_col_name, transition_names,
                               component_set, dynamic)
    if (keyword == 'preprocessing'):
        causality_df = Preprocessing(activities_col_name, caseIDs_col_name, timestamps_col_name).preprocessing(dataset_name, reader)
        inference_tool.inference(causality_df, output_name)
    else:
        df = pd.read_csv(dataset_name, sep=',')
        #results = pd.read_csv('performances.csv', sep=',')
        causality_df = utils.addNoise(df, activities_col_name, caseIDs_col_name, timestamps_col_name, percentage_in_CI,percentage_of_CI)
        start_time = time.time()
        start_memory = memory_usage()[0]
        row = dict()
        inference_tool.inference(causality_df, output_name)
        end_time = time.time()
        end_memory = memory_usage()[0]
        execution_time = end_time - start_time
        mem_usage = end_memory - start_memory
        row['TEST'] = output_name
        row['TIME'] = execution_time
        row['MEMORY'] = mem_usage
        #results.loc[len(results)] = row
        #results.to_csv('performances.csv', index=False)
    return row
