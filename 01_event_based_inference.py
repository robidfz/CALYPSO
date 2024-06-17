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

def structureDefinition_old(effect_names,A,E,rules,windows_matrix,structure_causes):
    f.write('STRUCTURE EVALUATION\n')
    cause_effect_dict=dict()
    prima_facie_dict=dict()
    for e in effect_names:
        f.write('\n\n\nEFFECT CONSIDERED:'+str(e)+'\n')
        prima_facie = list()
        for a in A:
            if (e != a):
                if (primafacie(e, a, windows_matrix, E, True)):
                    prima_facie.append(a)
        prima_facie_dict[e]=prima_facie
        selected_causes=prima_facie.copy()
        for p in prima_facie:
            if(p not in structure_causes):
                selected_causes.remove(p)

        eps_avg = epsilon_averages(selected_causes, windows_matrix, e)
        #max_eps = max(eps_avg.values())
        mean = np.mean(list(eps_avg.values()))
        std_dev = np.std(list(eps_avg.values()))

        if(std_dev>0.3):
            z_scores_dict=dict()
            for key, value in eps_avg.items():
                z_score= (value - mean) / std_dev
                z_scores_dict[key]=z_score
            f.write('\n\n\nZ-SCORE TEST \n')
            f.write('Effect;Cause;Zeta-score\n')
            for key, z_value in z_scores_dict.items():
                f.write(str(e) + ';' + str(key) + ';' + str(z_value) + '\n')
            f.write('\n\n\n\n')
            threshold=0.3
        else:
            z_scores_dict=eps_avg.copy()
            threshold = 0.80
        new_causes = [(key,eps_avg[key]) for key, value in z_scores_dict.items() if value >= threshold]
        cause_effect_dict[e] = new_causes
    structureVisualisation(cause_effect_dict)
    f.write('\n\n\n')

    return prima_facie_dict,cause_effect_dict


def filtering_rules(effect, ar_df):
    filtered_df = ar_df[ar_df['consequents'].isin(effect)]
    grouped = filtered_df.groupby('consequents').apply(lambda x: list(zip(x['antecedents'], x['support'])))
    structure_dict = grouped.to_dict()

    return structure_dict
def structureDefinition(dataset,effect_names,component_col_name,structure_causes):
    dataset=dataset[dataset[component_col_name].isin(structure_causes)]
    items_matrix = windowMatrix(dataset, component_col_name)
    rules = apriori_ar(items_matrix)
    rulesVisualization(rules)
    structure_dict= filtering_rules(effect_names, rules)
    structureVisualisation(structure_dict)
    return structure_dict
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





def apriori_ar(df):
    df = fpgrowth(df, min_support=0.02, use_colnames=True, max_len=2, verbose=1)
    df_ar = association_rules(df, metric="confidence", min_threshold=0.8)
    df_ar['antecedents'] = df_ar['antecedents'].apply(lambda x: eval(str(x).replace("frozenset({", "").replace("})", "")))
    df_ar['consequents'] = df_ar['consequents'].apply(lambda x: eval(str(x).replace("frozenset({", "").replace("})", "")))

    df_ar.to_csv('rules.csv', index=False)
    return df_ar

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
def rulesVisualization(rules):
    G = nx.Graph()
    for idx, rule in rules.iterrows():
        ant = str(rule['antecedents'])
        cons = str(rule['consequents'])
        G.add_node(ant)
        G.add_node(cons)
        G.add_edge(ant, cons, weight=round(rule['confidence'], 2))
    graphVisualization(G,'association_rules.pdf')


def filtering_rules_old(effect,causes,ar_df,window_matrix):
    effect_list=list()
    effect_list.append(effect)
    composed_causes=list()
    composed_causes_lists=list()
    filtered_df=ar_df[ ar_df['consequents'].apply(lambda x: x==effect_list) ]
    for c in filtered_df['antecedents'].values:
        if(all(elem in causes for elem in c)):
            composed_causes_lists.append(c)
    filtered_df=filtered_df[ ar_df['antecedents'].apply(lambda x: x in composed_causes_lists) ]['antecedents']
    for row in filtered_df:
        cause1=row[0]
        for i in range(1,len(row)):
            window_matrix = updatingWindowMatrix(window_matrix, 'AND', cause1, row[i])
            cause1 = cause1 + '_AND_' + row[i]
            composed_causes.append(cause1)
    return window_matrix,composed_causes




def discoveringPredicates(dataset,effects_names,activities_col_name,predicate_causes):
    window_matrix = windowMatrix(dataset, activities_col_name)

    f.write('\n\n\n\n\n\nPREDICATES EVALUATION\n')
    f.write('Effect;Cause;Eps\n')

    for e in effects_names:
        prima_facie = list()
        #k= [key for key in structure_dict.keys() if key in e][0]
        #structure_comp = structure_dict[k]
        #structure_comp = [c[0] for c in structure_comp]
        #new_causes=[x for x in activities if any(x.startswith(prefix) for prefix in structure_comp)]
        for c in predicate_causes:
            if (primafacie(e,c,window_matrix,True,dataset)):
                prima_facie.append(c)
        composed_causes=list()
        for i,p in enumerate(prima_facie):
            for j in range(i+1,len(prima_facie)):
                window_matrix = updatingWindowMatrix(window_matrix, 'AND', p, prima_facie[j])
                composed_cause=p + '_AND_' + prima_facie[j]
                if (primafacie(e, composed_cause, window_matrix)):
                    composed_causes.append(composed_cause)
        for x in composed_causes:
            prima_facie.append(x)
        eps_avg = epsilon_averages(prima_facie, window_matrix, e)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_events_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        E = pd.read_csv(dataset_events_name)
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        structure_effects_names = eval(reader['INFERENCE']['structure_effects'])
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        component_col_name=eval(reader['DATASET']['items_col'])[0]
        structure_causes= eval(reader['INFERENCE']['structure_causes'])
        predicates_causes=eval(reader['INFERENCE']['predicates_causes'])
        predicates_effects_names = eval(reader['INFERENCE']['predicates_effects'])
        template='PREPROCESSING_SETTING'
        E=preprocessing(reader,E,template)
        activity_of_interest=['is_down','sigA>50']
        a=E[activities_col_name].unique()
        #E = E[mask]
        #E = E[(~E[activities_col_name].str.endswith('failed_by_itself')) ]
        f = open('results.txt', 'w')
        A=E[activities_col_name].unique()

        structure_dict=structureDefinition(E,structure_effects_names,component_col_name,structure_causes)
        discoveringPredicates(E,predicates_effects_names,activities_col_name,predicates_causes)




        f.close()







