import numpy as np
import Methodology.utils as utils


def epsilon_averages(prima_facie_causes,df,effect):

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
                if(candx!=0):
                    Pcandx[j, k] = (alpha+Ecandx)/(beta+candx)
                    Exandnotc = df[(df[f] == 1) & (df[c] == 0) & (df[x] == 1)].shape[0]
                    Pcandnotx[j, k] = (alpha + Exandnotc) / (beta + xandnotc)
                    number_causes += 1

                    if (xandnotc==0 and (c in x or x in c)):
                        Pcandx[j, k] = 0
                        Pcandnotx[j, k] = 0
                        number_causes += 1
                        '''
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