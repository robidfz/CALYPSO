import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt
from configparser import ConfigParser
import sys
from Inference import Inference
from sklearn.tree import _tree


def tree_to_code(tree, feature_names,f):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"for i in tree_.feature]
    print("def tree({}):".format(", ".join(feature_names)))
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            f.write("{}return {}".format(indent, tree_.value[node]))
    recurse(0, 1)



def dataset(df):
    #obs=prima_facie.copy()
    #obs.append(effect)
    df = df[df['observation'].str.endswith('is_down')]
    col=df['observation'].unique()
    bn_df = pd.DataFrame(columns=col)
    cases = df.groupby('case:concept:name')
    new_rows = list()
    for case, activities in cases:
        new_row = dict()
        for idx, row in activities.iterrows():
            new_row[row['observation']] = 1
        new_rows.append(new_row)
    new_rows = pd.DataFrame(new_rows)
    bn_df = pd.concat([bn_df, new_rows], ignore_index=True)
    bn_df.fillna(0, inplace=True)
    bn_df.to_csv('causalities_dataset.csv', index=False)
    return  bn_df

def combinations_dataset(df):
    new_col_1=list()
    new_col_2 = list()
    for i,row in df.iterrows():
        if(row['X_C10_is_down']==1 and row['X_C11_is_down']==1):
            new_col_1.append(1)
        else:
            new_col_1.append(0)
        if (row['X_C20_is_down'] == 1 and row['X_C21_is_down'] == 1):
            new_col_2.append(1)
        else:
            new_col_2.append(0)
    df['X_C10_AND_X_C11_down']=new_col_1
    df['X_C20_AND_X_C21_down'] = new_col_2
    df.to_csv('causalities_dataset_with_combinations.csv', index=False)
    return df


def computing_probability(df,feature,target=None):
    probability=0
    if(target!=None):
        c_and_x=df[(df[target] == 1) & (df[feature] == 1)].shape[0]
        c=df[df[feature] == 1].shape[0]
        probability=c_and_x/c
    else:
        c = df[df[feature] == 1].shape[0]
        all=df.shape[0]
        probability=c/all
    return probability

def analysis(features_initials,df,target):

    dim=len(features_initials)
    Pcandx = np.zeros((dim, dim))
    Pcandnotx = np.zeros((dim, dim))
    concurrent_causes=np.zeros(dim)
    for j,causes in enumerate(features_initials):
        number_causes = 0
        for k in range(len(features_initials)):
            Ecandx=df[(df[target]==1) & (df[causes]==1) & (df[features_initials[k]]==1)].shape[0]
            candx = df[(df[causes] == 1) & (df[features_initials[k]] == 1)].shape[0]
            candnotx = df[(df[causes] == 0) & (df[features_initials[k]] == 1)].shape[0]
            if(j==k):
                Pcandx[j, k] = 0
                Pcandnotx[j, k] = 0
            else:
                if(candx!=0 and candnotx!=0):
                    Pcandx[j, k] = Ecandx/candx
                    #Pcandx[k, j] = Pcandx[j, k]
                    Ecandnotx = df[(df[target]==1) & (df[causes]==0) & (df[features_initials[k]]==1)].shape[0]
                    #candnotx = df[(df[causes] == 0) & (df[features_initials[k]] == 1)].shape[0]
                    Pcandnotx[j, k] = Ecandnotx/candnotx
                    number_causes+=1
                    #Pcandnotx[k, j] = Pcandnotx[j, k]

                elif(candnotx==0):
                    Eandc = df[(df[causes] == 1) & (df[target] == 1)].shape[0]
                    Pc = df[(df[causes] == 1)].shape[0]
                    Pcandx[j, j] = 0
                    Pcandnotx[j, k] = 0
                    number_causes += 1
                else:
                    Pcandx[j, k] = 0
                    #Pcandx[k, j] = 0
                    Pcandnotx[j, k] = 0
                    #Pcandnotx[k, j] = 0
        concurrent_causes[j]=number_causes
        #n_causes=np.count_nonzero(Pcandx[j])
        if(number_causes==0):
            Eandc = df[(df[causes] == 1) & (df[target] == 1)].shape[0]
            Pc = df[(df[causes] == 1)].shape[0]
            Pcandx[j, j] = Eandc / Pc
    #sum=Pcandx-Pcandnotx
    eps_avg=list()
    for i in range(Pcandx.shape[0]):
        card = concurrent_causes[i]
        row=Pcandx[i]-Pcandnotx[i]
        eps=row.sum()
        eps=eps/card
        eps_avg.append(eps)
    #eps_avg=np.sum(sum, axis=1)/(len(features_initials)-1)
    for j, causes in enumerate(features_initials):
        f.write(str(target)+';'+str(causes)+';'+str(eps_avg[j])+'\n')


    return eps_avg


def decisionTree(df,target,features,f):
    X=df[features]
    y=df[target]
    classes=list()
    for i in y.unique():
        classes.append(str(i))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    text= export_text(clf,feature_names=features)
    f.write('\n\n\n Decision Tree accuracy for discovering rules for '+target+'\n')
    #f.write(text)
    tree_to_code(clf,features,f)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=features, class_names=classes, rounded=True)
    plt.savefig('Decision_tree_'+target+'.pdf')
    #y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #f.write('\n\n\n Decision Tree accuracy for discovering rules for '+target+'\n')
    #f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n\n")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        E = pd.read_csv(dataset_name)
        inference = Inference(dataset_name, pb=True)
        targets=eval(reader['NETWORK SETTING']['target'])
        f = open('results.txt', 'w')
        f.write('Effect;Cause;Eps\n')
        data = dataset(inference.source)
        data=combinations_dataset(data)
        A=E['observation'].unique()

        for t in targets:
            for
            inference.generate_hypotheses_for_effects(causes=inference.alphabet, effects=[t])
            inference.test_hypotheses()
            prima_facie = list()
            for i,c in enumerate(inference.prima_facie[t]):
                prima_facie.append(c[0])
                if(t=='X_C1s_is_down'):
                    prima_facie.append('X_C10_AND_X_C11_down')
                if (t == 'X_C2s_is_down'):
                    prima_facie.append('X_C20_AND_X_C21_down')

                #for j in range(i+1,len(inference.prima_facie[t])):

            #couples, data = dataset(inference.source, t, prima_facie)
            #models= building_network(inference.source,t,prima_facie)
            eps_avg=analysis(prima_facie,data,t)
            features = list()
            for i, c in enumerate(inference.prima_facie[t]):
                features.append(c[0])
            decisionTree(data,t,features,f)
            #eps_avg = analysis_smooth(prima_facie, data, t, 0)
        f.close()








