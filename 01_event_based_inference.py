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
from PdFT_syntax_Elements.component import Component
from PdFT_syntax_Elements.structure import PdFT
from PdFT_syntax_Elements.event import Event
from PdFT_syntax_Elements.transition import Transition
file = open('results.txt', 'w')
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
            groups=df.groupby(caseIDs_col_name)
            col_name = actual_features[0]
            col_name2 = actual_features[1]
            for index, group in groups:
                for i, values in enumerate(actual_values):
                    col_value = values[0]
                    col_value2 = values[1]
                    df_1=parsing_strings(group, col_name, col_value)
                    df_2=parsing_strings(df_1, col_name2, col_value2)
                    df_2.sort_values([timestamps_col_name],inplace=True, ascending=True)
                    if(df_2.shape[0]>0):
                        adding_row=df_2.iloc[0]
                        col_to_rename=actual_output[i][0]
                        new_name=actual_output[i][1]
                        adding_row[col_to_rename] = new_name
                        inference_df.loc[len(inference_df)]=adding_row
                        #inference_df=pd.concat([inference_df,adding_row], axis=0)




    inference_df.sort_values([timestamps_col_name],inplace=True)
    inference_df=addingUpState(inference_df,activities_col_name)
    inference_df.to_csv(filename, index=False)

    return inference_df
def preprocessing_old(reader,df, template):
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
                for i, features in enumerate(actual_features):
                    col_name=features[0]
                    col_value=features[1]
                    col_name2=actual_values[i][0]
                    col_value2=actual_values[i][1]
                    if((row[col_name].startswith(col_value)) & (row[col_name2]>col_value2)):
                        inference_df.at[index,actual_output[i][0]]=actual_values[i][0]


    inference_df.sort_values([timestamps_col_name],inplace=True)


    inference_df=addingUpState(inference_df,activities_col_name)
    inference_df.to_csv(filename, index=False)

    return inference_df

def addingUpState(df,rows_col_name):
    act = df[rows_col_name].unique()
    components = extractComponents(act)
    cases = df.groupby(caseIDs_col_name)
    col=df.columns
    for case, activities in cases:

        timestamp=min(activities[timestamps_col_name].values)
        caseID=activities[caseIDs_col_name].unique()[0]
        for c in components:
            if (not any(c in element for element in activities[rows_col_name])):
                new_rows = dict()
                for column in col:
                    new_rows[column]=None

                new_rows[rows_col_name] = c+"_is_up"
                new_rows[timestamps_col_name]=timestamp
                new_rows[caseIDs_col_name]=caseID
                df.loc[len(df)]=new_rows
    df = df.sort_values(by=timestamps_col_name)
    return df


def ends_with_any(value, suffixes):
    if isinstance(value, str):
        return any(value.endswith(suffix) for suffix in suffixes)
    return False



def extractComponents(activities):
    activities = pd.Series(activities)
    result = activities.apply(splitActivity)
    components = result.apply(lambda x: x[0]).drop_duplicates()
    mask = ~components.str.startswith('sig')
    components = components[mask]
    return components



def windowMatrixold(df,rows_col_name):
    col=df[rows_col_name].unique()
    cases = df.groupby(caseIDs_col_name)
    components=extractComponents(col)
    new_rows = list()
    for case, activities in cases:
        new_row = dict()
        new_row['CaseID']=activities[caseIDs_col_name].unique()
        for c in components:
            if (not any(c in element for element in activities[rows_col_name])):
                new_row[c+"_is_up"] = 1
        for idx, row in activities.iterrows():
            new_row[row[rows_col_name]] = 1
        new_rows.append(new_row)
    window_matrix = pd.DataFrame(new_rows)
    window_matrix.fillna(0, inplace=True)
    window_matrix=window_matrix[col]
    return  window_matrix


def windowMatrix(df,rows_col_name):
    col=df[rows_col_name].unique()
    cases = df.groupby(caseIDs_col_name)
    new_rows = list()
    for case, activities in cases:
        new_row = dict()
        new_row['CaseID']=activities[caseIDs_col_name].unique()
        for idx, row in activities.iterrows():
            new_row[row[rows_col_name]] = 1
        new_rows.append(new_row)
    window_matrix = pd.DataFrame(new_rows)
    window_matrix.fillna(0, inplace=True)
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
    window_matrix.to_csv('window_matrix.csv', index=False)
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
        i = i + 1
    condition1=(condition1 and flag)
    return condition1
def findComponentState(group, timestamp, components):
    state = dict()

    for component in components:
        filtered_group = group[group[activities_col_name].str.startswith(component)]

        if(len(filtered_group)>0):
            #filtered_group=filtered_group[filtered_group[timestamps_col_name]<timestamp]
            closest_timestamp = filtered_group.iloc[(filtered_group[timestamps_col_name] - timestamp).abs().argmin()]
            state[closest_timestamp[activities_col_name]]=closest_timestamp[timestamps_col_name]
        else:
            closest_timestamp=group.iloc[group[timestamps_col_name].abs().argmin()]
            state[component+"_is_up"]=closest_timestamp[timestamps_col_name]

    return state




def primafacie_condition1_predicates(effect,cause,df):
    operator=findOperator(cause)

    preds=splitPredicate(cause)
    effect_initial, effect_final=splitTransitionActivities(effect)
    components=list()
    causes_initials=list()
    causes_finals=list()
    for e in preds:
        cause_initial, cause_final = splitTransitionActivities(e)
        causes_initials.append(cause_initial)
        causes_finals.append(cause_final)
        component, act = splitActivity(e)
        components.append(component)


    cases = df[caseIDs_col_name].unique()
    condition1 = True
    flag = False
    i = 0
    if(operator=='AND'):
        while (condition1 and i < len(cases)):
            group = df[df[caseIDs_col_name] == cases[i]]
            times=list()
            if ((effect_initial in group[activities_col_name].values) and (effect_final in group[activities_col_name].values)):
                #selecting effect transition timestamp
                effect_timestamp = max(group[group[activities_col_name] == effect_final][timestamps_col_name].values)

                if (all([a in group[activities_col_name].values for a in causes_initials]) and all([a in group[activities_col_name].values for a in causes_finals]) ):
                    flag = True
                    for a in causes_finals:
                        # selecting cause transition timestamp
                        cause_timestamp = group[group[activities_col_name] == a][timestamps_col_name].values
                        times.append(cause_timestamp)
                    cause_timestamp=max(times)
                    condition1 = (cause_timestamp < effect_timestamp)
            i = i + 1
    else:
        while (condition1 and i < len(cases)):
            group = df[df[caseIDs_col_name] == cases[i]]
            times = list()
            if ((effect_initial in group[activities_col_name].values) and (effect_final in group[activities_col_name].values)):
                # selecting effect transition timestamp
                effect_timestamp = max(group[group[activities_col_name] == effect_final][timestamps_col_name].values)
                if (any([a in group[activities_col_name].values for a in causes_initials]) and any([a in group[activities_col_name].values for a in causes_finals]) ):
                    flag = True
                    for e in causes_finals:
                        if(e in group[activities_col_name].values):
                            cause_timestamp = group[group[activities_col_name] == e][timestamps_col_name].values
                            times.append(cause_timestamp)
                    cause_timestamp = min(times)
                    condition1 = (cause_timestamp < effect_timestamp)
            i = i + 1
    condition1=(condition1 and flag)
    return condition1


def primafacie_condition1_predicates_old(effect,cause,df):
    operator=findOperator(cause)

    preds=splitPredicate(cause)
    effect_initial, effect_final=splitTransitionActivities(effect)
    components=list()
    for e in preds:
        cause_initial, cause_final = splitTransitionActivities(preds)
        component, act = splitActivity(e)
        components.append(component)
    if(effect_final=='X_top_is_up' and operator=='AND' and len(preds)==3):
        print('hello')

    cases = df[caseIDs_col_name].unique()
    condition1 = True
    flag = False
    i = 0
    if(operator=='AND'):
        while (condition1 and i < len(cases)):
            group = df[df[caseIDs_col_name] == cases[i]]
            times=list()
            if ((effect_initial in group[activities_col_name].values) and (effect_final in group[activities_col_name].values)):
                #selecting transition timestamp
                effect_timestamp = max(group[group[activities_col_name] == effect_final][timestamps_col_name].values)
                #extract last state occured for each components
                components_states=findComponentState(group, effect_timestamp,components)
                if (all([a in components_states.keys() for a in preds])):
                    flag = True
                    for a in preds:
                        #cause_timestamp = group[group[activities_col_name] == a][timestamps_col_name].values[0]
                        cause_timestamp = components_states[a]
                        times.append(cause_timestamp)
                    cause_timestamp=max(times)
                    condition1 = (cause_timestamp < effect_timestamp)
            i = i + 1
    else:
        while (condition1 and i < len(cases)):
            group = df[df[caseIDs_col_name] == cases[i]]
            times = list()
            if ((effect_initial in group[activities_col_name].values) and (effect_final in group[activities_col_name].values)):
                effect_timestamp = max(group[group[activities_col_name] == effect_final][timestamps_col_name].values)
                components_states = findComponentState(group, effect_timestamp, components)
                if (any([e in components_states.keys() for e in preds])):
                    flag = True
                    for e in preds:
                        if(e in components_states.keys()):
                            cause_timestamp = components_states[e]
                            times.append(cause_timestamp)
                    cause_timestamp = min(times)
                    condition1 = (cause_timestamp < effect_timestamp)
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
        condition3 = (conditional_probability >= effect_probability)
    pf = ( condition2 and condition3 )
    return pf

def primafacie(effect,cause,window_matrix,condition1=False,df=None):
    if(condition1):
        cond1=primafacie_condition1(effect,cause,df)
    else:
        cond1=primafacie_condition1_predicates(effect,cause,df)
    if(cond1):
        cond2_3=primafacie_condition2and3(effect,cause,window_matrix)
    else:
        cond2_3=False
    pf=(cond1 and cond2_3)

    return pf
def epsilon_averages(prima_facie_causes,df,effect):

    dim=len(prima_facie_causes)
    Pcandx = np.zeros((dim, dim))
    Pcandnotx = np.zeros((dim, dim))
    concurrent_causes=np.zeros(dim)
    alpha=1
    beta=2
    eps_list=list()
    if(effect=='X_top_is_down'):
        print('ciao')
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
        eps_list.append([prima_facie_causes[i],eps])

    #eps_avg=np.sum(sum, axis=1)/(len(prima_facie_causes)-1)



    return eps_list



def filteringSignificantCauses(eps_avg_dict,maximum=False):
    cause_effect_dict=dict()
    for effect, values in eps_avg_dict.items():
        causes=[c[0] for c in values]
        eps=[c[1] for c in values]
        if(maximum):
            try:
                max_value=max(eps)
            except:
                max_value=0
            new_causes = [(value[0], value[1]) for value in values if value[1] ==max_value]
        else:
            mean = np.mean(eps)
            std_dev = np.std(eps)
            filtered_causes = list()
            threshold=0
            if (std_dev > 0.1):
                threshold=mean
            new_causes = [(value[0],value[1]) for value in values if value[1] >= threshold]
        cause_effect_dict[effect] = new_causes
    for key, value in cause_effect_dict.items():
        for c in value:
            file.write(str(key) + ';' + str(c[0]) + ';' + str(c[1]) + '\n')
    file.write('\n\n\n\n')
    return cause_effect_dict


def structureDefinition(dataset,effect_names,component_col_name):
    window_matrix = windowMatrix(dataset, component_col_name)
    possible_activities= dataset[component_col_name].unique().tolist()
    cause_effect_dict=dict()
    for e in effect_names:
        A=[a for a in possible_activities if a.endswith(e)]
        file.write('Evaluating EFFECT '+ str(e)+'\n')

        for f in A:
            prima_facie = list()
            for a in A:
                if(f!=a):
                    if (primafacie(f, a, window_matrix, True, dataset)):
                        prima_facie.append(a)
            eps_avg=epsilon_averages(prima_facie,window_matrix,f)
            cause_effect_dict[f] = eps_avg
    cause_effect_dict=filteringSignificantCauses(cause_effect_dict)
    structure_dict,structure_list=structureVisualisation(cause_effect_dict)
    file.write('\n\n\n')
    return window_matrix, structure_dict,structure_list


def findComponent(components, name):
    for component in components:
        if component.name == name:
            return component
    return None

def structureVisualisation(cause_effect_dict):
    G = nx.DiGraph()
    structure=PdFT()
    col=['Effect','Effect Component','Cause','Cause Component','Effect Input Port','Effect Output Port','Cause Output Port','Event','Weight value']
    structure_df=pd.DataFrame(columns=col)
    significant_causes_dict=dict()
    structure_list=list()


    i=0
    counter=0
    for effect, values in cause_effect_dict.items():
        significant_causes=list()
        effect_comp, occ_name = splitActivity(effect)
        if(findComponent(structure_list, effect_comp)==None ):

            effect_obj= Component(effect_comp,'c_'+str(i))
            structure_list.append(effect_obj)
            structure.addComponent(effect_obj)
            effect_pedix = effect_obj.extractPedix()
            i=i+1
        else:
            #effect_obj = findComponent(structure_list, effect_comp)
            effect_obj=structure.findComponent(effect_comp)
            effect_pedix = effect_obj.extractPedix()

        causes = [c[0] for c in values]
        eps = [c[1] for c in values]
        cons = effect
        j = i
        for k,cause in enumerate(causes):
            cause_comp, occ_name = splitActivity(cause)
            if (findComponent(structure_list, cause_comp) == None):
                cause_obj = Component(cause_comp,'c_' + str(j))
                structure.addComponent(cause_obj)
                structure_list.append(cause_obj)
                cause_pedix = cause_obj.extractPedix()
                j = j + 1
            else:
                #cause_obj = findComponent(structure_list, cause_comp)
                cause_obj=structure.findComponent(cause_comp)
                cause_pedix = cause_obj.extractPedix()



            input_port='p_{' + str(effect_pedix)+','+str(cause_pedix)+'}'
            effect_obj.addInputPort(input_port)
            event=Event('e_' + str(counter),cause_obj.output_port,input_port,eps[k])
            structure.addEvent(event)
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]] = cause
            row[col[3]] = cause_obj.code
            row[col[4]] = event.structure[1]
            row[col[5]] = effect_obj.output_port
            row[col[6]] = event.structure[0]
            row[col[7]] = event.name
            row[col[8]] = event.weight
            structure_df.loc[len(structure_df)] = row
            ant=cause
            G.add_node(ant,name=cause_obj.code)
            G.add_node(cons,name=effect_obj.code)
            G.add_edge(ant, cons, name='e_' + str(counter) ,weight=round(eps[k],4))
            significant_causes.append([cause_comp,eps[k]])
            counter=counter+1
        i=j

        significant_causes_dict[effect_comp]=significant_causes
    structure_df.to_csv('PdFT_structure.csv',index=False)
    graphVisualization(G,structure,'PdFT_structure.pdf')
    return significant_causes_dict,structure






def graphVisualization(G,structure,name_fig):

    fig, ax = plt.subplots(figsize=(40, 40))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos,node_shape='s',node_color='lightblue',node_size=30000)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    node_labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=50, font_color='black')
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12,font_color='black')

    for (u, v, d) in G.edges(data=True):
        effect_name,act=splitActivity(v)
        cause_name,act_cause=splitActivity(u)
        effect_comp = structure.findComponent(effect_name)
        cause_comp=structure.findComponent(cause_name)
        p_name_in = effect_comp.getInputPort(cause_comp)
        p_name_out = cause_comp.output_port

        x_start, y_start = pos[u]
        x_end, y_end = pos[v]

        # Calcolare la direzione dell'arco
        dx = x_end - x_start
        dy = y_end - y_start
        angle = np.arctan2(dy, dx)

        # Offset per posizionare il quadrato sul bordo del nodo
        offset_x = 0.600167 * np.cos(angle)
        offset_y = 0.600167 * np.sin(angle)

        # Posizionare il quadrato sul bordo del nodo di partenza
        box_start_x = x_start + offset_x
        box_start_y = y_start + offset_y
        plt.plot([box_start_x], [box_start_y], marker='s', markersize=60, color='grey')
        plt.text(box_start_x, box_start_y, p_name_out, fontsize=20, ha='center', va='center', color='black')

        # Posizionare il quadrato sul bordo del nodo di arrivo
        box_end_x = x_end - offset_x
        box_end_y = y_end - offset_y
        plt.plot([box_end_x], [box_end_y], marker='s', markersize=60, color='grey')
        plt.text(box_end_x, box_end_y, p_name_in, fontsize=20, ha='center', va='center', color='black')

        # Disegnare l'arco dai quadrati
        #plt.arrow(box_start_x, box_start_y, box_end_x - box_start_x, box_end_y - box_start_y,length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
    #labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(name_fig)
    plt.close()





def matchingStrings(s1,s2,operator):
    elements=s1.split("_"+operator+"_")
    check=any(e in s2 for e in elements)
    return check

def matchingElements(s1,s2,operator):
    elements1=s1.split("_"+operator+"_")
    elements2 = s2.split("_" + operator + "_")
    check=(set(elements1)==set(elements2))
    return check

def findOperator(s):
    operator='AND'
    index=s.find(operator)
    if(index==-1):
        operator='OR'
    return operator
def splitPredicate(s):
    operator=findOperator(s)
    elements = s.split("_"+operator+"_")
    return elements
def splitActivity(activity):
    index0=activity.find('_')
    elem=activity[index0+1:]
    index=elem.find('_')+index0+1
    elem=activity[:index]
    occurence_name=activity[index+1:]
    return elem,occurence_name
def defineActivity(elem,occurrence):
    return elem+'_'+occurrence

def defineTransitionActivities(component,transition):
    initial_state = transition[0]
    final_state = transition[1]
    t=component+"_"+initial_state+"\\"+final_state
    return t

def splitTransitionActivities(transition_activity):
    transition_activity_index =transition_activity.rfind('\\')
    c,act=splitActivity(transition_activity)
    final_state=c+"_"+transition_activity[transition_activity_index+1:]
    index=transition_activity[:transition_activity_index].rfind("_")
    index_1=transition_activity[:index].rfind("_")
    initial_state=c+"_"+transition_activity[index_1+1:transition_activity_index]
    return initial_state,final_state
def buildingTransitions(window_matrix,transition):
    initial_state=transition[0]
    final_state=transition[1]
    activities=window_matrix.columns
    components= extractComponents(activities)
    for i,row in window_matrix.iterrows():
        for c in components:
            initial_activity=c+"_"+initial_state
            final_activity=c+"_"+final_state
            transition_activity=defineTransitionActivities(c,transition)
            if((row[initial_activity]==1 ) and (row[final_activity]==1)):
                window_matrix.at[i, transition_activity] = 1
            else:
                window_matrix.at[i, transition_activity] = 0

    window_matrix.to_csv("window_matrix.csv",index=False)
    return window_matrix

def discoveringPredicates(df,windows_matrix,effects_transition,structure_causes,structure_list):

    eps_avg_dict=dict()
    file.write('\n\n\n\n\n\nPREDICATES EVALUATION\n')
    file.write('Effect;Cause;Eps\n')
    possible_activities=windows_matrix.columns
    #selecting signals predicates
    effects_names=structure_causes.keys()
    predicates_causes=[a for a in possible_activities if a.startswith('sig')]
    operands=['AND','OR']
    for f in effects_names:

        for t in effects_transition:
            if (t[1] == 'is_up'and f=='X_top'):
                print('ciao')
            windows_matrix=buildingTransitions(windows_matrix,t)
            effect=defineTransitionActivities(f,t)
            #selecting the significant causes for f computed in the previous step
            connected_components = structure_causes[f]
            connected_components = [c[0] for c in connected_components]

            '''
            if (t == 'is_down'):
                print('sto prendendo i predicati')
                A=predicates_causes+significant_causes
            else:
                A=significant_causes
            '''
            #significant_causes = [c + "_" + t[1] for c in connected_components] +
            significant_causes=[defineTransitionActivities(c,t) for c in connected_components]


            #significant_causes=[c for c in windows_matrix.columns if any(c.startswith(prefix) for prefix in significant_causes)]
            A=list()
            composed_causes=list()
            for operand in operands:
                possible_composed_causes = significant_causes.copy()
                for i,p in enumerate(possible_composed_causes):
                    p_initial,p_final=splitTransitionActivities(p)
                    if(primafacie(effect, p_final, windows_matrix,True,df)):
                        A.append(p)

                    for j in range(i+1,len(possible_composed_causes)):
                        # building the composed causes and verify the prima facie conditions

                        if(matchingStrings(p,possible_composed_causes[j],operand)==False):
                            composed_cause = p + '_'+operand+'_' + possible_composed_causes[j]
                            check=any([matchingElements(composed_cause,x,operand) for x in possible_composed_causes])
                            if(check==False):
                                possible_composed_causes.append(composed_cause)
                                windows_matrix = updatingWindowMatrix(windows_matrix, operand, p, possible_composed_causes[j])

                                if (primafacie(effect, composed_cause, windows_matrix,False,df)):
                                    composed_causes.append(composed_cause)


            #selecting the composed causes that are prima facie for e
            A=A+composed_causes
            eps_avg = epsilon_averages(A, windows_matrix, effect)
            eps_avg_dict[effect]=eps_avg
    cause_effect_dict=filteringSignificantCauses(eps_avg_dict,False)
    structure=buildingSemantics(cause_effect_dict,structure_list)
    #buildingSemantics(eps_avg_dict,structure_list)
    return structure


def discoveringThresholds(df,windows_matrix,structure):

    eps_avg_dict=dict()
    file.write('\n\n\n\n\n\nPREDICATES EVALUATION\n')
    file.write('Effect;Cause;Eps\n')
    possible_activities=windows_matrix.columns
    #selecting signals predicates
    effects_names=['X_C11','X_C10','X_C31','X_C30','X_C21','X_C20']
    t=['is_up','is_down']
    thresholds_predicates=[a for a in possible_activities if a.startswith('sig')]
    for f in effects_names:
        effect = defineTransitionActivities(f, t)
        f=f+'_is_down'
        A=list()
        for a in thresholds_predicates:
            if(primafacie(f, a, windows_matrix,True,df)):
                A.append(a)
        eps_avg = epsilon_averages(A, windows_matrix, f)

        eps_avg_dict[effect]=eps_avg
    cause_effect_dict=filteringSignificantCauses(eps_avg_dict,False)
    buildingDynamics(cause_effect_dict,structure)
    #buildingSemantics(eps_avg_dict,structure_list)



def buildingSemantics(cause_effect_dict,structure):
    col=['Effect','Component Effect','Cause','Trigger','Alpha','Rho']
    predicate_df=pd.DataFrame(columns=col)
    for effect, values in cause_effect_dict.items():
        effect_comp, effect_state = splitActivity(effect)
        initial_state,final_state=splitTransitionActivities(effect)
        e_comp,initial_state=splitActivity(initial_state)
        e_comp,final_state=splitActivity(final_state)
        effect_obj = structure.findComponent(effect_comp)
        causes = [c[0] for c in values]
        eps=[c[1] for c in values]
        for k, cause in enumerate(causes):
            operator=findOperator(cause)
            elems=splitPredicate(cause)
            predicate=''
            for e in elems:
                cause_comp, occ_name = splitActivity(e)
                cause_obj = structure.findComponent(cause_comp)
                input_port=effect_obj.getInputPort(cause_obj)
                predicate=predicate+input_port+"_"+operator+"_"
            predicate=predicate[:-1]
            index=predicate.rfind("_")
            predicate=predicate[:index]
            t=effect_obj.findTransition(initial_state,final_state)
            t.setPredicate(predicate)
            t.setPort(effect_obj.output_port)
            t.setProbability(eps[k])
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]]=cause
            row[col[3]]=t.triggerFunction()
            row[col[4]]=t.alphaFunction()
            row[col[5]]=t.rhoFunction()
            predicate_df.loc[len(predicate_df)] = row
    predicate_df.to_csv('predicate_PdFT.csv',index=False)
    return structure

def splitDynamicCauses(cause):
    index = cause.find('>')
    threshold = cause[index + 1:]
    elem = cause[:index]

    return elem, threshold
def buildingDynamics(cause_effect_dict,structure):
    col=['Effect','Component Effect','Cause','Trigger','Alpha','Rho']
    predicate_df=pd.DataFrame(columns=col)
    for effect, values in cause_effect_dict.items():
        effect_comp, effect_state = splitActivity(effect)
        initial_state, final_state = splitTransitionActivities(effect)
        name,initial_state=splitActivity(initial_state)
        name,final_state=splitActivity(final_state)
        effect_obj = structure.findComponent(effect_comp)
        causes = [c[0] for c in values]
        eps=[c[1] for c in values]
        for k, cause in enumerate(causes):
            dynamic_name,threshold=splitDynamicCauses(cause)
            effect_obj.setDynamic(dynamic_name,threshold)
            predicate=cause
            t=effect_obj.findTransition(initial_state,final_state)
            t.setPredicate(predicate)
            t.setPort(effect_obj.output_port)
            t.setProbability(eps[k])
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]]=cause
            row[col[3]]=t.triggerFunction()
            row[col[4]]=t.alphaFunction()
            row[col[5]]=t.rhoFunction()
            predicate_df.loc[len(predicate_df)] = row
    predicate_df.to_csv('thresholds_PdFT.csv',index=False)

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
        E = E[(~E[activities_col_name].str.endswith('under_repair'))]
        template='PREPROCESSING_SETTING'
        E=preprocessing(reader,E,template)
        window_matrix,structure_dict,structure=structureDefinition(E,structure_effects_names,activities_col_name)
        structure=discoveringPredicates(E,window_matrix,predicates_effects_names,structure_dict,structure)
        discoveringThresholds(E,window_matrix,structure)




        file.close()







