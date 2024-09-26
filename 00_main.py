import sys
from configparser import  ConfigParser
from Workflow.preprocessing import Preprocessing
from Workflow.inference import Inference
from Causality_analysis_modules import utils
import pandas as pd
import time
from memory_profiler import memory_usage

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
        results = pd.read_csv('performances.csv', sep=',')
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
        results.loc[len(results)] = row
        #results.to_csv('performances.csv', index=False)
    return results

if __name__ == "__main__":
    if len(sys.argv) == 7:
        t = tuple(sys.argv[1:])
        execute(*t)

