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



def tree_to_code(tree, feature_names,f):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"for i in tree_.feature]
    f.write("def tree({}):\n".format(", ".join(feature_names)))
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write("{}if {} <= {}:\n".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            f.write("{}else:\n  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            f.write("{}return {}\n".format(indent, tree_.value[node]))
    recurse(0, 1)



def windowMatrix(df):
    #df = df[df[activities_col_name].str.endswith('is_down')]
    col=df[activities_col_name].unique()
    window_matrix = pd.DataFrame(columns=col)
    cases = df.groupby(caseIDs_col_name)
    new_rows = list()
    for case, activities in cases:
        new_row = dict()
        for idx, row in activities.iterrows():
            new_row[row[activities_col_name]] = 1
        new_rows.append(new_row)
    new_rows = pd.DataFrame(new_rows)
    window_matrix = pd.concat([window_matrix, new_rows], ignore_index=True)
    window_matrix.fillna(0, inplace=True)
    window_matrix.to_csv('causalities_dataset.csv', index=False)
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

def primafacie(effect,cause,window_matrix,df=None,condtion1=False):
    if(condtion1!=False):
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
    eps_dict=dict()
    for j,causes in enumerate(prima_facie_causes):
        number_causes = 0
        for k in range(len(prima_facie_causes)):
            Ecandx=df[(df[effect]==1) & (df[causes]==1) & (df[prima_facie_causes[k]]==1)].shape[0]
            candx = df[(df[causes] == 1) & (df[prima_facie_causes[k]] == 1)].shape[0]
            candnotx = df[(df[causes] == 0) & (df[prima_facie_causes[k]] == 1)].shape[0]
            if(j==k):
                Pcandx[j, k] = 0
                Pcandnotx[j, k] = 0
            else:
                if(candx!=0 and candnotx!=0):
                    Pcandx[j, k] = Ecandx/candx
                    Ecandnotx = df[(df[effect]==1) & (df[causes]==0) & (df[prima_facie_causes[k]]==1)].shape[0]
                    Pcandnotx[j, k] = Ecandnotx/candnotx
                    number_causes+=1
                elif(candnotx==0):
                    Pcandx[j, j] = 0
                    Pcandnotx[j, k] = 0
                    number_causes += 1
                else:
                    Pcandx[j, k] = 0
                    Pcandnotx[j, k] = 0


        if(number_causes==0):
            Eandc = df[(df[causes] == 1) & (df[effect] == 1)].shape[0]
            Pc = df[(df[causes] == 1)].shape[0]
            Pcandx[j, j] = Eandc / Pc
            number_causes=1
        concurrent_causes[j] = number_causes


    for i in range(Pcandx.shape[0]):
        card = concurrent_causes[i]
        row=Pcandx[i]-Pcandnotx[i]
        eps=row.sum()
        eps=eps/card
        eps_dict[prima_facie_causes[i]]=eps

    #eps_avg=np.sum(sum, axis=1)/(len(prima_facie_causes)-1)
    for j, (key,value) in enumerate(eps_dict.items()):
        f.write(str(effect)+';'+str(key)+';'+str(value)+'\n')


    return eps_dict


def decisionTree(df,effect,features,f):
    X=df[features]
    y=df[effect]
    classes=list()
    for i in y.unique():
        classes.append(str(i))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    text= export_text(clf,feature_names=features)
    f.write('\n\n\n Decision Tree accuracy for discovering rules for '+effect+'\n')
    #f.write(text)
    tree_to_code(clf,features,f)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=features, class_names=classes, rounded=True)
    f.write('\n\n\n\n')
    plt.savefig('Decision_tree_'+effect+'.pdf')
    #y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #f.write('\n\n\n Decision Tree accuracy for discovering rules for '+effect+'\n')
    #f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n\n")


def apriori_ar(df):
    df = fpgrowth(df, min_support=0.02, use_colnames=True, max_len=4, verbose=1)
    df_ar = association_rules(df, metric="confidence", min_threshold=0.5)
    df_ar['antecedents'] = df_ar['antecedents'].apply(lambda x: eval(str(x).replace("frozenset({", "[").replace("})", "]")))
    df_ar['consequents'] = df_ar['consequents'].apply(lambda x: eval(str(x).replace("frozenset({", "[").replace("})", "]")))

    df_ar.to_csv('rules.csv', index=False)
    return df_ar


def visualization(rules):
    G = nx.Graph()
    node_style = 'circle, draw, fill, inner sep=2pt'
    col_ant = 'blue'
    col_cons = 'grey'
    for idx, rule in rules.iterrows():
        ant = str(rule['antecedents'])
        cons = str(rule['consequents'])
        G.add_node(ant, color=col_ant, size=5000)
        G.add_node(cons, color=col_cons, size=5000)
        G.add_edge(ant, cons, weight=round(rule['confidence'], 2))
    for node in G.nodes():
        G.nodes[node]['tex'] = '\\node[{}] ({})'.format(node_style, node)
    edge_width = {e: "line width=1.5" for e in G.edges}
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20, 10))
    nx.draw(G, pos, with_labels=True, font_size=8, font_color='white', font_weight='bold', linewidths=5)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig('association_rules.pdf')
    plt.close()


def filtering_rules(effect,ar_df,window_matrix):
    effect_list=list()
    effect_list.append(effect)
    composed_causes=list()
    filtered_df=ar_df[ ar_df['consequents'].apply(lambda x: x==effect_list) ]['antecedents']
    for row in filtered_df:
        cause1=row[0]
        for i in range(1,len(row)):
            window_matrix = updatingWindowMatrix(window_matrix, 'AND', cause1, row[i])
            cause1 = cause1 + '_AND_' + row[i]
            composed_causes.append(cause1)
    return window_matrix,composed_causes



if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        E = pd.read_csv(dataset_name)
        activities_col_name=eval(reader['DATASET']['activities_col'])[0]
        effects_names=eval(reader['EFFECT']['effects_list'])
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        f = open('results.txt', 'w')

        #data = dataset(inference.source)
        #data=combinations_dataset(data)
        A=E[activities_col_name].unique()
        window_matrix=windowMatrix(E)
        #ar_items=association_rules_dataset(E)
        rules = apriori_ar(window_matrix)
        visualization(rules)

        for e in effects_names:
            prima_facie = list()
            for a in A:
                if(e!=a):
                    if(primafacie(e,a,window_matrix,E,True)):
                        prima_facie.append(a)
            f.write('Effect;Cause;Eps\n')
            eps_avg = epsilon_averages(prima_facie, window_matrix, e)
            #decisionTree(window_matrix, e, prima_facie, f)
            max_eps=max(eps_avg.values())
            new_causes=[key for key,value in eps_avg.items() if value==max_eps]
            window_matrix,composed_causes=filtering_rules(e,rules,window_matrix)
            '''
            for i, a in enumerate(new_causes):
                for j in range(i+1,len(new_causes)):
                    window_matrix = updatingWindowMatrix(window_matrix, 'AND', a, new_causes[j])
                    composed_causes.append(a+'_AND_'+new_causes[j])
                    window_matrix = updatingWindowMatrix(window_matrix, 'OR', a, new_causes[j])
                    composed_causes.append(a + '_OR_' + new_causes[j])
            '''
            for c in composed_causes:
                if(primafacie(e,c,window_matrix)):
                    new_causes.append(c)
            f.write('Composed Causes evaluation\n')
            f.write('Effect;Cause;Eps\n')
            eps_avg = epsilon_averages(new_causes, window_matrix, e)



        f.close()






'''

            if (e == 'X_top_is_down'):
                window_matrix = updatingWindowMatrix(window_matrix, 'OR', 'X_C1s_is_down', 'X_C2s_is_down')
                prima_facie.append('X_C1s_is_down_OR_X_C2s_is_down')
            if (e == 'X_C1s_is_down'):
                window_matrix = updatingWindowMatrix(window_matrix, 'AND', 'X_C10_is_down', 'X_C11_is_down')
                prima_facie.append('X_C10_is_down_AND_X_C11_is_down')
            if (e == 'X_C2s_is_down'):
                window_matrix = updatingWindowMatrix(window_matrix, 'AND', 'X_C20_is_down', 'X_C21_is_down')
                prima_facie.append('X_C20_is_down_AND_X_C21_is_down')
'''

