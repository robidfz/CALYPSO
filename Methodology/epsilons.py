import numpy as np


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