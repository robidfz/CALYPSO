import numpy as np
import Methodology.utils as utils


def epsilon_averages_old(prima_facie_causes,df,effect):

    dim=len(prima_facie_causes)
    Pcandx = np.zeros((dim, dim))
    Pcandnotx = np.zeros((dim, dim))
    concurrent_causes=np.zeros(dim)
    activities=df.columns.tolist()
    alpha=1
    beta=2
    eps_list=list()
    if(effect=='X_top_is_down'):
        print('ciao')
    for j,c in enumerate(prima_facie_causes):
        number_causes = 0
        f = effect
        for k in range(len(prima_facie_causes)):
            if(j!=k):
                x=prima_facie_causes[k]

                Ecandx = df[(df[f] == 1) & (df[c] == 1) & (df[x] == 1)].shape[0]
                candx = df[(df[c] == 1) & (df[x] == 1)].shape[0]
                xandnotc = df[(df[c] == 0) & (df[x] == 1)].shape[0]
                #if x and c are cofounding cause for E
                if(Ecandx!=0):
                    #if are not semantically impossible to compare
                    if(xandnotc!=0):
                        Pcandx[j, k] = (alpha+Ecandx)/(beta+candx)
                        Exandnotc = df[(df[f] == 1) & (df[c] == 0) & (df[x] == 1)].shape[0]
                        Pcandnotx[j, k] = (alpha + Exandnotc) / (beta + xandnotc)
                        number_causes += 1
                    else:
                        Pcandx[j, k] = 0
                        Pcandnotx[j, k] = 0
                    '''
                    if (xandnotc==0 and (c in x or x in c)):
                        if(c in x):
                            Pcandx[j, k] = 0
                            Pcandnotx[j, k] = 0
                            number_causes += 1
                        if(x in c):
                            Pcandx[j, k] = 1
                            Pcandnotx[j, k] = 0
                            number_causes += 1

                        
                        if(c in x):
                            Pcandx[j, k] = 0
                            Pcandnotx[j, k] = 0
                            number_causes += 1
                        if(x in c):
                            elems=utils.splitPredicate(c)
                            operator=utils.findOperator(c)
                            new_c=[e for e in elems if e not in x]
                            possible_causes=utils.combinations(new_c,operator)
                            causes = [e for e in possible_causes if e in activities]
                            cause=causes[0]
                            Ecandx = df[(df[f] == 1) & (df[cause] == 1) & (df[x] == 1)].shape[0]
                            candx = df[(df[cause] == 1) & (df[x] == 1)].shape[0]
                            xandnotc = df[(df[cause] == 0) & (df[x] == 1)].shape[0]
                            Pcandx[j, k] = (alpha + Ecandx) / (beta + candx)
                            Exandnotc = df[(df[f] == 1) & (df[cause] == 0) & (df[x] == 1)].shape[0]
                            Pcandnotx[j, k] = (alpha + Exandnotc) / (beta + xandnotc)
                            number_causes += 1
                            '''

                else:
                    Pcandx[j, k] = 0
                    Pcandnotx[j, k] = 0




        if(number_causes==0):
            Eandc = df[(df[c] == 1) & (df[f] == 1)].shape[0]
            only_c = df[(df[c] == 1)].shape[0]
            Eandnotc=df[(df[c] == 0) & (df[f] == 1)].shape[0]
            notc=df[(df[c] == 0)].shape[0]
            Pcandx[j, j] = ((alpha+Eandc) / (beta+only_c)) - ((alpha+Eandnotc)/(beta+notc))
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


def epsilon_averages(prima_facie_causes, df, effect,activity_col_name,caseIDs_col_name,timestamp_col_name,delta=None):
    dim = len(prima_facie_causes)
    P_f_given_a_and_x = np.zeros((dim, dim))
    P_f_given_nota_and_x = np.zeros((dim, dim))
    concurrent_causes = np.zeros(dim)
    activities = df.columns.tolist()
    if(delta==None):
        delta=max(df[timestamp_col_name].values)
    alpha = 1
    beta = 2
    eps_list = list()
    if (effect == 'X_top_is_down'):
        print('ciao')
    for j, a in enumerate(prima_facie_causes):
        not_a=utils.negateCause(a)
        number_causes = 0
        f = effect
        for k in range(len(prima_facie_causes)):
            if (j != k):
                x = prima_facie_causes[k]
                a_and_x=a+"_AND_"+x
                f_and_a_and_x =f+'_AND_'+ a + "_AND_" + x
                nota_and_x=x+'_AND_'+not_a
                cofunding_causes_frequence=utils.computingFrequencies(df,a_and_x,activity_col_name,caseIDs_col_name,timestamp_col_name,10,f)
                nota_and_x_freq=utils.computingFrequencies(df,nota_and_x,activity_col_name,caseIDs_col_name,timestamp_col_name)
                if(cofunding_causes_frequence!=0 and nota_and_x_freq!=0 ):

                    P_f_given_a_and_x[j, k] =utils.computingProbability(df,a_and_x,activity_col_name,caseIDs_col_name,timestamp_col_name,delta,f)
                    P_f_given_nota_and_x[j, k]=utils.computingProbability(df,nota_and_x,activity_col_name,caseIDs_col_name,timestamp_col_name,delta,f)
                    number_causes += 1
                else:
                    P_f_given_a_and_x[j, k] = 0
                    P_f_given_nota_and_x[j, k] = 0

        if (number_causes == 0):

            P_f_given_a_and_x[j, j] = utils.computingProbability(df, a, activity_col_name, caseIDs_col_name,timestamp_col_name, delta, f)
            P_f_given_nota_and_x[j, j] = utils.computingProbability(df, not_a, activity_col_name, caseIDs_col_name,timestamp_col_name, delta, f)
            number_causes = 1
        concurrent_causes[j] = number_causes

    for i in range(P_f_given_a_and_x.shape[0]):
        card = concurrent_causes[i]
        row = P_f_given_a_and_x[i] - P_f_given_nota_and_x[i]
        eps = row.sum()
        eps = eps / card
        eps_list.append([prima_facie_causes[i], eps])

    # eps_avg=np.sum(sum, axis=1)/(len(prima_facie_causes)-1)

    return eps_list