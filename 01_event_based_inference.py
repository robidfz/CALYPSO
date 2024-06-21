import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt
from configparser import ConfigParser
import sys
from sklearn.tree import _tree
from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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
def preprocessing(reader,df, template):
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
            inference_df= parsing_strings(df, actual_features[0], actual_values[0])
        if(p=='filtering2'):

            col_feature=actual_features[0][0]
            col_value=actual_features[0][1]
            for i,values in enumerate(actual_values):
                name_feature=values[0]
                instance_value=values[1]
                df_1=parsing_strings(df, col_feature, name_feature)
                df_2=parsing_strings(df_1, col_value, instance_value)
                #result=pd.concat([df_1,df_2],axis=0)
                inference_df=pd.concat([inference_df,df_2], axis=0)

        if(p=='renaming'):
            for index, row in inference_df.iterrows():
                col_name=actual_features[0][0]
                col_value=actual_features[0][1]
                if(row[col_name].startswith(col_value)):
                    inference_df.at[index,actual_output[0]]=actual_values[0]


    inference_df.sort_values([timestamps_col_name],inplace=True)



    inference_df.to_csv(filename, index=False)

    return inference_df

def ends_with_any(value, suffixes):
    if isinstance(value, str):
        return any(value.endswith(suffix) for suffix in suffixes)
    return False





def windowMatrix(df,rows_col_name):
    col=df[rows_col_name].unique()
    cases = df.groupby(caseIDs_col_name)
    new_rows = list()
    for case, activities in cases:
        new_row = dict()
        new_row['CaseID']=case
        for idx, row in activities.iterrows():
            new_row[row[rows_col_name]] = 1
        new_rows.append(new_row)
    window_matrix = pd.DataFrame(new_rows)
    window_matrix.fillna(0, inplace=True)
    window_matrix.to_csv('windows_matrix.csv', index=False)
    window_matrix=window_matrix[col]
    return  window_matrix

def updatingWindowMatrix(df,logic_operator,operand1,operand2):
    new_cause=operand1+'_'+logic_operator+'_'+operand2
    if(new_cause not in df.columns):
        values=list()
        if(logic_operator=='AND'):
            for i,row in df.iterrows():
                if(row[operand1]==1 and row[operand2]==1):
                    values.append(1)
                else:
                    values.append(0)
        if (logic_operator == 'OR'):
            for i, row in df.iterrows():
                if (row[operand1] == 0 and row[operand2] == 0):
                    values.append(0)
                else:
                    values.append(1)
        df[new_cause]=values
    return df


def computingProbability(df,cause,effect=None):
    if(effect!=None):
        c_and_x=df[(df[effect] == 1) & (df[cause] ==1)].shape[0]
        c=df[df[cause] == 1].shape[0]
        probability=c_and_x/c
    else:
        c = df[df[cause]==1].shape[0]
        all=df.shape[0]
        probability=c/all
    return probability

def primafacie_condition1(effect,cause,df):
    cases = df[caseIDs_col_name].unique()
    condition1 = True
    flag = False
    i = 0
    while (condition1 and i < len(cases)):
        group = df[df[caseIDs_col_name] == cases[i]]
        if (cause in group[activities_col_name].values):
            cause_timestamp = group[group[activities_col_name] == cause][timestamps_col_name].values[0]
            if (effect in group[activities_col_name].values):
                flag = True
                effect_timestamp = group[group[activities_col_name] == effect][timestamps_col_name].values[0]
                condition1 = (cause_timestamp < effect_timestamp)
                if((effect=='X_C1s_is_down' or effect=='X_C3s_is_down')  and cause_timestamp==effect_timestamp):
                    print(cause+'-'+effect+' time:'+str(cause_timestamp)+' caseid: '+str(cases[i])+'\n')
        i = i + 1
    condition1=(condition1 and flag)
    return condition1

def primafacie_condition2and3(effect,cause,window_matrix):

    condition3 = False
    cause_probability = computingProbability(window_matrix, cause)
    condition2 = (cause_probability > 0)
    if (condition2):
        conditional_probability = computingProbability(window_matrix, cause, effect)
        effect_probability = computingProbability(window_matrix, effect)
        condition3 = (conditional_probability > effect_probability)
    pf = ( condition2 and condition3 )
    return pf

def primafacie(effect,cause,window_matrix,condition1=False,df=None):
    if(condition1):
        cond1=primafacie_condition1(effect,cause,df)
        if(cond1):
            cond2_3=primafacie_condition2and3(effect,cause,window_matrix)
        else:
            cond2_3=False
        pf=(cond1 and cond2_3)
    else:
        pf=primafacie_condition2and3(effect,cause,window_matrix)

    return pf
def epsilon_averages(prima_facie_causes,df,effect):

    dim=len(prima_facie_causes)
    Pcandx = np.zeros((dim, dim))
    Pcandnotx = np.zeros((dim, dim))
    concurrent_causes=np.zeros(dim)
    alpha=1
    beta=2
    eps_dict=dict()
    for j,causes in enumerate(prima_facie_causes):
        number_causes = 0

        for k in range(len(prima_facie_causes)):
            if(j!=k):
                Ecandx = df[(df[effect] == 1) & (df[causes] == 1) & (df[prima_facie_causes[k]] == 1)].shape[0]
                candx = df[(df[causes] == 1) & (df[prima_facie_causes[k]] == 1)].shape[0]
                candnotx = df[(df[causes] == 0) & (df[prima_facie_causes[k]] == 1)].shape[0]
                if(candx!=0):
                    Pcandx[j, k] = (alpha+Ecandx)/(beta+candx)
                    Ecandnotx = df[(df[effect] == 1) & (df[causes] == 0) & (df[prima_facie_causes[k]] == 1)].shape[0]
                    Pcandnotx[j, k] = (alpha + Ecandnotx) / (beta + candnotx)
                    number_causes += 1

                    if (Ecandnotx==0 and (causes in prima_facie_causes[k])):
                        Pcandx[j, k] = 0
                        Pcandnotx[j, k] = 0
                        number_causes += 1

                else:
                    Pcandx[j, k] = 0
                    Pcandnotx[j, k] = 0


        if(number_causes==0):
            Eandc = df[(df[causes] == 1) & (df[effect] == 1)].shape[0]
            c = df[(df[causes] == 1)].shape[0]
            Eandnotc=df[(df[causes] == 0) & (df[effect] == 1)].shape[0]
            notc=df[(df[causes] == 0)].shape[0]
            Pcandx[j, j] = ((alpha+Eandc) / (beta+c)) - ((alpha+Eandnotc)/(beta+notc))
            number_causes=1
        concurrent_causes[j] = number_causes


    for i in range(Pcandx.shape[0]):
        card = concurrent_causes[i]
        row=Pcandx[i]-Pcandnotx[i]
        eps=row.sum()
        eps=eps/card
        eps_dict[prima_facie_causes[i]]=eps

    #eps_avg=np.sum(sum, axis=1)/(len(prima_facie_causes)-1)
    f.write('Effect;Cause;Eps\n')
    for j, (key,value) in enumerate(eps_dict.items()):
        f.write(str(effect)+';'+str(key)+';'+str(value)+'\n')


    return eps_dict






def structureDefinition(dataset,effect_names,component_col_name):
    window_matrix = windowMatrix(dataset, component_col_name)
    A = dataset[component_col_name].unique().tolist()
    A=[a for a in A if a.endswith('is_down')]
    f.write('\n\n\n\n\n\nPREDICATES EVALUATION\n')
    cause_effect_dict=dict()
    for e in effect_names:
        f.write('Evaluating EFFECT '+ str(e)+'\n')
        f.write('Effect;Cause;Eps\n')
        prima_facie=list()
        for c in A:
            if(e!=c):
                if (primafacie(e, c, window_matrix, True, dataset)):
                    prima_facie.append(c)
        eps_avg=epsilon_averages(prima_facie,window_matrix,e)
        mean = np.mean(list(eps_avg.values()))
        std_dev = np.std(list(eps_avg.values()))

        if (std_dev > 0.3):
            z_scores_dict = dict()
            for key, value in eps_avg.items():
                z_score = (value - mean) / std_dev
                z_scores_dict[key] = z_score
            f.write('\n\n\nZ-SCORE TEST \n')
            f.write('Effect;Cause;Zeta-score\n')
            for key, z_value in z_scores_dict.items():
                f.write(str(e) + ';' + str(key) + ';' + str(z_value) + '\n')
            f.write('\n\n\n\n')
            threshold = 0.3
        else:
            z_scores_dict = eps_avg.copy()
            threshold = 0.70
        new_causes = [(key, eps_avg[key]) for key, value in z_scores_dict.items() if value >= threshold]
        cause_effect_dict[e] = new_causes
    structureVisualisation(cause_effect_dict)
    f.write('\n\n\n')
    return window_matrix, cause_effect_dict
def structureVisualisation(cause_effect_dict):
    G = nx.Graph()

    for effect, causes in cause_effect_dict.items():
        ant = effect
        for c in causes:
            cons=c[0]
            G.add_node(ant)
            G.add_node(cons)
            G.add_edge(ant, cons, weight=round(c[1],4))
    graphVisualization(G,'PdFT_structure.pdf')







def graphVisualization(G,name_fig):
    fig, ax = plt.subplots(figsize=(12, 10))
    node_style = 'circle, draw, fill, inner sep=2pt'
    for node in G.nodes():
        G.nodes[node]['tex'] = '\\node[{}] ({})'.format(node_style, node)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=True, node_size=4000, font_size=16, arrowsize=30, node_color='#D3FBFB',ax=ax, font_color="black", edge_color="black")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(name_fig)
    plt.close()







def discoveringPredicates(windows_matrix,effects_names,structure_causes):


    f.write('\n\n\n\n\n\nPREDICATES EVALUATION\n')
    f.write('Effect;Cause;Eps\n')
    possible_activities=windows_matrix.columns
    #selecting signals predicates
    predicates_causes=[a for a in possible_activities if a.startswith('sig')]
    for e in effects_names:
        f.write('Evaluating EFFECT ' + str(e) + '\n')
        #selecting the significant causes for e computed in the previous step
        significant_causes = structure_causes[e]
        significant_causes = [c[0] for c in significant_causes]
        A=predicates_causes+significant_causes
        composed_causes=list()
        for i,p in enumerate(significant_causes):
            for j in range(i+1,len(significant_causes)):
                #building the composed causes and verify the prima facie conditions
                windows_matrix = updatingWindowMatrix(windows_matrix, 'AND', p, significant_causes[j])
                composed_cause=p + '_AND_' + significant_causes[j]
                if (primafacie(e, composed_cause, windows_matrix)):
                    composed_causes.append(composed_cause)
        #selecting the composed causes that are prima facie for e
        A=A+composed_causes
        eps_avg = epsilon_averages(A, windows_matrix, e)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        E = pd.read_csv(dataset_name)
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        structure_effects_names = eval(reader['INFERENCE']['structure_effects'])
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        #structure_causes= eval(reader['INFERENCE']['structure_causes'])
        #predicates_causes=eval(reader['INFERENCE']['predicates_causes'])
        predicates_effects_names = eval(reader['INFERENCE']['predicates_effects'])
        template='PREPROCESSING_SETTING'
        E=preprocessing(reader,E,template)
        E_structure = E[(~E[activities_col_name].str.endswith('failed_by_itself'))]
        f = open('results.txt', 'w')
        window_matrix,structure_dict=structureDefinition(E_structure,structure_effects_names,activities_col_name)
        discoveringPredicates(window_matrix,predicates_effects_names,structure_dict)




        f.close()







