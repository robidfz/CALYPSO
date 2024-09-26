import numpy as np
import pandas as pd
from scipy import stats
import Causality_analysis_modules.utils as utils





def epsilon_averages(prima_facie_causes, df, effect,activity_col_name,caseIDs_col_name,timestamp_col_name,delta=None):
    errors=['X_C11_is_up\\is_down','X_C10_is_up\\is_down','X_C21_is_up\\is_down','X_C20_is_up\\is_down','X_C31_is_up\\is_down','X_C30_is_up\\is_down']
    if(effect in errors):
        print('Hello')
    dim = len(prima_facie_causes)
    P_f_given_a_and_x = np.zeros((dim, dim))
    P_f_given_nota_and_x = np.zeros((dim, dim))
    concurrent_causes = np.zeros(dim)
    p_value_dict=dict()
    filtered_causes=list()
    #epsilons_dataframe = pd.DataFrame(columns=['Cause'] + prima_facie_causes)
    if(delta==None):
        delta=max(df[timestamp_col_name].values)
    alpha = 1
    beta = 2
    eps_list = list()
    for j, a in enumerate(prima_facie_causes):
        p_value_list=list()

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
                    p_value_list.append(P_f_given_a_and_x[j, k]- P_f_given_nota_and_x[j, k])
                    #new_row[prima_facie_causes[j]]=P_f_given_a_and_x[j, k]- P_f_given_nota_and_x[j, k]
                else:
                    P_f_given_a_and_x[j, k] = 0
                    P_f_given_nota_and_x[j, k] = 0
                    #new_row[prima_facie_causes[j]]=None
            else:
                P_f_given_a_and_x[j, k] = 0
                P_f_given_nota_and_x[j, k] = 0
                #new_row[prima_facie_causes[j]]= None
        #new_row_df = pd.DataFrame([new_row])
        #epsilons_dataframe = pd.concat([epsilons_dataframe, new_row_df], ignore_index=True)

        if (number_causes == 0):

            P_f_given_a_and_x[j, j] = utils.computingProbability(df, a, activity_col_name, caseIDs_col_name,timestamp_col_name, delta, f)
            P_f_given_nota_and_x[j, j] = utils.computingProbability(df, not_a, activity_col_name, caseIDs_col_name,timestamp_col_name, delta, f)
            p_value_list.append(P_f_given_a_and_x[j, j] - P_f_given_nota_and_x[j, j])
            number_causes = 1
        concurrent_causes[j] = number_causes

        if (len(p_value_list)>0):

            std_dev = np.std(p_value_list)
            if (std_dev > 0):
                mu=0
                t_statistic, p_value = stats.ttest_1samp(p_value_list, mu)
                p_value_dict[a]=p_value
            else:
                filtered_causes.append(a)

    filtered_causes=filtered_causes+[keys for keys,values in p_value_dict.items() if values<=0.05]

    for i in range(P_f_given_a_and_x.shape[0]):
        card = concurrent_causes[i]
        row = P_f_given_a_and_x[i] - P_f_given_nota_and_x[i]
        eps = row.sum()
        eps = eps / card
        eps_list.append([prima_facie_causes[i], eps])
    eps_list=[e for e in eps_list if e[0] in filtered_causes]
    #safe_file_name = effect.replace('\\', '_')
    # eps_avg=np.sum(sum, axis=1)/(len(prima_facie_causes)-1)
    #epsilons_dataframe.to_csv(safe_file_name+'_epsilons.csv',index=False)

    return eps_list