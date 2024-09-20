import pandas as pd
import sys
from configparser import ConfigParser
import random
def addNoise(df,percentage_in_caseID, percentage_of_caseID,number_test):
    activities=df[activities_col_name].unique().tolist()
    cases=df[caseIDs_col_name].unique().tolist()
    times= int((len(cases)*percentage_of_caseID)/100)
    for i in range(times):
        case = random.choice(cases)
        timestamps=df[df[caseIDs_col_name]==case][timestamps_col_name]
        timestamp_min = min(timestamps)
        timestamp_max = max(timestamps)
        number_rows_tot=len(timestamps)
        number_new_rows=int((number_rows_tot*percentage_in_caseID)/100)
        for j in range(number_new_rows):
            act = random.choice(activities)
            timestamp=random.uniform(timestamp_min,timestamp_max)
            new_line = {caseIDs_col_name:case,timestamps_col_name:timestamp,activities_col_name:act}
            df = df.append(new_line, ignore_index=True)
    df=df.sort_values(by=timestamps_col_name)
    df.to_csv('Threat_to_validity_TEST'+str(number_test)+'.csv',index=False)




if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        df = pd.read_csv(dataset_name)
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        percentage_in_CI = eval(reader['NOISE']['percentage_in_CaseID'])
        percentage_of_CI = eval(reader['NOISE']['percentage_of_CaseID'])
        number_test=0
        for i in percentage_of_CI:
            for j in percentage_in_CI:

                addNoise(df, i, j,number_test)
                number_test+=1

