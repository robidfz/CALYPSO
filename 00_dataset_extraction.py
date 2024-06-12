from configparser import ConfigParser
import pandas as pd
import sys
import numpy as np
import ast


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



def caseID_definition(df,column,value,name):
    counter=0
    caseID=list()
    i=0
    while(i<len(df[column])):

        while(i<len(df[column]) and df[column][i].endswith(value) ):
            caseID.append(counter)
            i=i+1
        counter = counter + 1
        while(i<len(df[column]) and not df[column][i].endswith(value) ):
            caseID.append(counter)
            i = i + 1
        counter = counter + 1

    df[name]=caseID
    return df
def addColumn1(dataset,column_dict,features,output):
    new_col = list()
    for i, row in dataset.iterrows():
        for key,values in column_dict.items():
            if(row[features].startswith(key)):
                new_val = row[features]+"_"+str(row[values])
                new_val=new_val.replace(' ','_')
                new_col.append(new_val)
    dataset[output] = new_col
    print('fatto')
    return dataset
def addColumn2(dataset,feature,value,output):
    new_col=list()
    for i, row in dataset.iterrows():
        val = row[feature].replace(" ", "_")
        flag=0
        for v in value:
            if(val.endswith(v[1])):
                elem= v[0]
                new_col.append(elem)
                flag=1
        if(flag==0):
            elem=value[2][0]
            new_col.append(elem)

    dataset[output] = new_col
    return dataset
def preprocessing(reader,df,template):
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
            df=addColumn1(df,column_dict,actual_features[0],actual_output[0])
        elif(p=='add_column2'):
            df = addColumn2(df, actual_features[0], actual_values, actual_output[0])
        elif(p=='renaming'):
            new_columns=dict()
            for i,v in enumerate(actual_features):
                new_columns[v]=actual_values[i]
            df = df.rename(columns=new_columns)




    df.to_csv(filename , index=False)

    return df





if __name__ == "__main__":
    if len(sys.argv) == 4:
        config = sys.argv[1]
        dataset=sys.argv[2]
        filename=sys.argv[3]
        reader = ConfigParser()
        reader.read(config)
        df=dataset_creation(dataset,reader)
        dataset_checking(df,filename,reader)
        print(df.shape)
        template='preprocessing_setting'
        df=preprocessing(reader,df,template)
        #template = 'dataset_inference_setting'
        #df_inference = preprocessing(reader, df, template)
        template = 'dataset_timeseries_setting'
        df_timeseries = preprocessing(reader, df, template)







