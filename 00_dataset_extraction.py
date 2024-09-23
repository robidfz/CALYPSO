from configparser import ConfigParser
import pandas as pd
import sys
import numpy as np
import ast
from Methodology import utils

def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return value

def parsing_strings(df, column_name,expression):

    if expression.startswith("<"):
        threshold = float(expression[1:])
        result = df[df[column_name] < threshold]
    elif expression.startswith(">"):
        threshold = float(expression[1:])
        result = df[df[column_name] > threshold]
    elif expression.startswith("<="):
        threshold = float(expression[2:])
        result = df[df[column_name] <= threshold]
    elif expression.startswith(">="):
        threshold = float(expression[2:])
        result = df[df[column_name] >= threshold]
    elif expression.startswith("=="):
        threshold = str(expression[2:])
        result = df[df[column_name] == threshold]
    elif expression.startswith("!="):
        threshold = str(expression[2:])
        result = df[df[column_name] != threshold]
    else:
        raise ValueError("Operator not defined")
    return result




def dataset_creation(dataset,reader):
    with open(dataset, 'r') as file:
        rows = file.readlines()
    data = []
    lst = eval(reader['dataset_definition']['features'])
    for i,row in enumerate(rows):
        elem = row.strip().split(';')
        data_row=dict()
        for j,l in enumerate(lst):
            if(j==0):
                data_row[l]='e_'+str(i)
            else:
                data_row[l]=elem[j-1]
        data.append(data_row)

    df = pd.DataFrame(data)
    df['Value'] = df['Value'].replace('', np.nan)
    df['Value'] = df['Value'].apply(convert_to_numeric)
    df['time:timestamp']=pd.to_numeric(df['time:timestamp'])
    #df=df.head(100)

    return df

def dataset_checking(df,report_file,reader):
    report=open(report_file,'w')
    report.write('EFFECT:\n')
    options=reader.options('effect_checking')
    df_hyp = df
    for hyp in options:
        lst = eval(reader['effect_checking'][hyp])
        for t in lst:
            report.write(str(t[0]) + ' ' + str(t[1]) + '\n')
            df_hyp = df_hyp[df_hyp[t[0]] == t[1]]
        report.write('Number of effect occurence: ')
        report.write(str(df_hyp.shape[0])+'\n\n\n')
    report.write('POSSIBLE CAUSES:\n')
    options = reader.options('cause_checking')
    for o in options:
        df_hyp = df
        lst=eval(reader['cause_checking'][o])
        for t in lst:
            report.write(str(t[0]) + ' '+str(t[1])+'\n')
            if(t[0]=='Value'):
                df_hyp = parsing_strings(df_hyp,t[0],t[1])
            else:
                df_hyp=df_hyp[df_hyp[t[0]]==t[1]]
        report.write('Number of possible cause occurence: ')
        report.write(str(df_hyp.shape[0]) + '\n')

        report.write('\n')


    report.close()





def preprocessing_step1(reader,df,template):
    property=reader[template]['property'].split(',')
    features=eval(reader[template]['feature'])
    value=eval(reader[template]['value'])
    output=eval(reader[template]['output'])
    filename = eval(reader[template]['filename'])[0]


    for i,p in enumerate(property):
        actual_features=features[i]
        actual_values=value[i]
        actual_output=output[i]

        if (p == 'filtering'):
            #filter excluding debug records from simulator dataset that are stored in Warning level
            df= parsing_strings(df,actual_features[0],actual_values[0])

        #elif(p=='add_caseid'):
         #   colname=actual_features[0]
          #  val=actual_values[0]
           # name=actual_output[0]
            #df=caseID_definition(df,colname,val,name)

        elif(p=='add_column1'):
            #t=ast.literal_eval(actual_values)
            column_dict = dict(actual_values)
            df=utils.addColumn1(df,column_dict,actual_features[0],actual_output[0])
        elif(p=='add_column2'):
            df = utils.addColumn2(df, actual_features[0], actual_values, actual_output[0])
        elif(p=='renaming'):
            new_columns=dict()
            for i,v in enumerate(actual_features):
                new_columns[v]=actual_values[i]
            df = df.rename(columns=new_columns)




    df.to_csv(filename , index=False)

    return df



def preprocessing_step2(reader,df, template):
    property = reader[template]['property'].split(',')
    features = eval(reader[template]['feature'])
    value = eval(reader[template]['value'])
    output = eval(reader[template]['output'])
    filename = eval(reader[template]['filename'])[0]

    for i, p in enumerate(property):
        actual_features = features[i]
        actual_values = value[i]
        actual_output = output[i]

        if (p == 'filtering1'):
            # filter excluding debug records from simulator dataset that are stored in Warning level
            inference_df= utils.parsing_strings(df, actual_features[0], actual_values[0])
        if(p=='filtering2'):
            groups=df.groupby(caseIDs_col_name)
            col_name = actual_features[0]
            col_name2 = actual_features[1]
            for index, group in groups:
                for i, values in enumerate(actual_values):
                    col_value = values[0]
                    col_value2 = values[1]
                    df_1=utils.parsing_strings(group, col_name, col_value)
                    df_2=utils.parsing_strings(df_1, col_name2, col_value2)
                    df_2.sort_values([timestamps_col_name],inplace=True, ascending=True)
                    if(df_2.shape[0]>0):
                        adding_row=df_2.iloc[0]
                        col_to_rename=actual_output[i][0]
                        new_name=actual_output[i][1]
                        adding_row[col_to_rename] = new_name
                        inference_df.loc[len(inference_df)]=adding_row
                        #inference_df=pd.concat([inference_df,adding_row], axis=0)




    inference_df.sort_values([timestamps_col_name],inplace=True)
    inference_df=utils.addingUpState(inference_df,activities_col_name,caseIDs_col_name,timestamps_col_name)
    inference_df.to_csv(filename, index=False)

    return inference_df

if __name__ == "__main__":
    if len(sys.argv) == 4:
        config = sys.argv[1]
        dataset=sys.argv[2]
        filename=sys.argv[3]
        reader = ConfigParser()
        reader.read(config)
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        structure_effects_names = eval(reader['INFERENCE']['structure_effects'])
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        df=dataset_creation(dataset,reader)
        dataset_checking(df,filename,reader)
        print(df.shape)
        template='preprocessing_step1_setting'
        df=preprocessing_step1(reader,df,template)
        template = 'dataset_timeseries_setting'
        df_timeseries = preprocessing_step1(reader, df, template)
        template = 'preprocessing_step2_setting'
        df = preprocessing_step2(reader, df, template)
        #template = 'dataset_inference_setting'
        #df_inference = preprocessing(reader, df, template)








