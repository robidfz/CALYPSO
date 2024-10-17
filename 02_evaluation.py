import sys
import pandas as pd
from Causality_analysis_modules import utils
import os
import glob
import matplotlib.pyplot as plt
from Workflow import utils as wutils
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def computeFalseAndTruePositive(dataset,ground_truth):
    effects=ground_truth['Effect'].unique()
    falsePositives=0
    truePositives=0
    for e in effects:
        ground_truth_causes=ground_truth[ground_truth['Effect']==e]['Cause'].values
        causes = dataset[dataset['Effect']==e]['Cause'].values
        for c in causes:
            if(c not in ground_truth_causes):
                falsePositives=falsePositives+1
            else:
                truePositives=truePositives+1
    return falsePositives,truePositives


def computeFalseAndTrureNegative(dataset,ground_truth):
    effects = ground_truth['Effect'].unique()
    falseNegatives = 0
    trueNegatives=0
    for e in effects:
        ground_truth_causes = ground_truth[ground_truth['Effect'] == e]['Cause'].values
        causes = dataset[dataset['Effect'] == e]['Cause'].values
        for c in ground_truth_causes:
            if (c not in causes):
                falseNegatives = falseNegatives + 1
    return falseNegatives

def computeRuleScore(dataset,ground_truth):
    effects = ground_truth['Effect'].unique()
    tot_pred=ground_truth.shape[0]
    score=0
    for e in effects:
        ground_truth_predicate = ground_truth[ground_truth['Effect'] == e]['Cause'].values[0]
        inferred_predicate = dataset[dataset['Effect'] == e]['Cause'].values
        if(inferred_predicate!=None):
            inferred_predicate=inferred_predicate[0]
            if(utils.comparePredicates(ground_truth_predicate,inferred_predicate)):
                    score+=1
    score=score/tot_pred


    return score

def computeHeuristics(dataset,k):
    values=dataset['Epsilon']
    values_number=dataset.shape[0]
    sum=0
    for v in values:
        if(v>=0.5):
            sum=sum+(v*k)
        else:
            sum=sum+(v/k)
    sum=sum/(values_number*k)
    return sum

def computeMetrics(dataset,dataset_preds,ground_truth_structure,groud_truth_rules,k_values):
    falsePositive, truePositive = computeFalseAndTruePositive(dataset, ground_truth_structure)
    falseNegative = computeFalseAndTrureNegative(dataset, ground_truth_structure)
    accuracy = truePositive  / (falsePositive + falseNegative + truePositive)
    recall = truePositive / (truePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    pred_score = computeRuleScore(dataset_preds, groud_truth_rules)
    metrics=dict()
    metrics['ACCURACY']=accuracy
    metrics['RECALL']=recall
    metrics['PRECISION']=precision
    metrics['RULE SCORE']=pred_score
    dataset_h= pd.concat([dataset, dataset_preds], axis=0, ignore_index=True)

    for k in k_values:
        heuristic_value=computeHeuristics(dataset_h, k)
        metrics['HEURISTIC WITH k='+str(k)] = heuristic_value


    return metrics






def plot(df,columns,graph_name,y_attribute,filtered_attribute=None,v=None):

    df = df.sort_values(by=['Noise inter-processes','Noise intra-process'], ascending=[True,True])
    df_original=df.iloc[0]
    df_plot = df.iloc[1:]
    if(filtered_attribute!=None):
        df_plot = df_plot[df_plot[filtered_attribute] == v]
    df_plot = df_plot.groupby([y_attribute]).mean().reset_index()


    x=df_plot.shape[0]
    if(x<10):
        x=x+5
    else:
        x=x-10
    plt.figure(figsize=(x, 8))
    colors = wutils.generate_colors(len(columns))
    n = len(columns)
    ind = np.arange(len(df_plot[y_attribute]))
    width = 0.4 / n
    for i,col in enumerate(columns):
        value=df_original[col]
        plt.axhline(y=value, color=colors[i], linestyle='--')

    data =df_plot[columns].values

    for j in range(n):

        plt.bar(ind + j * width, df_plot[columns[j]], width, label=columns[j] , color=colors[j])
    if(np.min(data)>0.5):
        plt.ylim(0.5, 1)
    else:
        plt.ylim(0, 0.5)


    plt.xticks(ind + width * (n - 1) / 2, df_plot[y_attribute],rotation=45)

    if(v==None):
        s="Barplot_ "+graph_name
    else:
        s="Barplot_ "+graph_name+"_by_"+y_attribute+"_"+str(v)

    plt.xlabel(y_attribute)
    plt.ylabel("Metrics values")
    plt.legend(title="Metrics", loc='lower left', fontsize='small')
    plt.savefig(s+'.pdf')



def computeDeltas(dataset,columns):
    deltas_df=dataset[columns]
    deltas_df = deltas_df.sub(deltas_df.iloc[0], axis=1).abs().drop(index=0)
    deltas_df['Noise ']=dataset['Noise ']
    deltas_df['Noise inter-processes'] = dataset['Noise inter-processes']
    deltas_df['Noise intra-process'] = dataset['Noise intra-process']
    return deltas_df


def boxPlot(df, columns, graph_name):

    df = df.sort_values(by=['Noise inter-processes', 'Noise intra-process'], ascending=[True, True])
    df_long = pd.melt(df, id_vars=['Noise '], value_vars=columns, var_name='Metrics', value_name='Values')
    plt.figure(figsize=(20, 6))
    sns.boxplot(x='Noise ', y='Values', hue='Metrics', data=df_long, palette='Set2')
    plt.title("Box Plot of Metrics Values by Noise ")
    plt.xlabel("(NOISE  INTER-PROCESSES, NOISE  INTRA-PROCESS)")
    plt.ylabel("GROUP METRICS MEAN VALUE")
    plt.legend(title="Metrics")
    plt.savefig('boxPlot_'+graph_name+'.pdf')


def regressionPlane(X,Z):
    model = LinearRegression()
    model.fit(X, Z)
    A, B = model.coef_
    C = model.intercept_
    xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 10),
                         np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 10))
    zz = A * xx + B * yy + C
    return xx,yy,zz


def tridimensionalPlane(df, column, original_metrics):
    df = df.sort_values(by=['Noise inter-processes', 'Noise intra-process'], ascending=[True, True])

    # Converting columns to numeric without forcing them into integers
    df['Noise inter-processes'] = pd.to_numeric(df['Noise inter-processes'], errors='coerce')
    df['Noise intra-process'] = pd.to_numeric(df['Noise intra-process'], errors='coerce')

    X = df ['Noise intra-process','Noise inter-process']


    planes_features = [column] + original_metrics
    colors = wutils.generate_colors(len(planes_features))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For legend
    legend_elements = []

    for i, c in enumerate(planes_features):
        Z = df[c]
        xx, yy, zz = regressionPlane(X, Z)
        surface = ax.plot_surface(x, yy, zz, alpha=0.5, rstride=100, cstride=100, color=colors[i])

        # Add legend entry for each surface
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Surface regression for {c}'))

    # Scatter plot of the original points
    ax.scatter(df['Noise inter-processes'], df['Noise intra-process'], df[column], color='blue', s=50)

    # Axes labels
    ax.set_xlabel('Noise inter-processes')
    ax.set_ylabel('Noise intra-process')
    ax.set_zlabel(column + ' values')

    ax.legend(handles=legend_elements, loc='upper left', fontsize='small',
              bbox_to_anchor=(-0.3, 1.15))

    plt.savefig(column + '_3D.pdf')
    plt.show()

def tridimensionalPlot(df, column):
    df = df.sort_values(by=['Noise inter-processes', 'Noise intra-process'], ascending=[True, True])

    # Converting columns to numeric without forcing them into integers
    df['Noise inter-processes'] = pd.to_numeric(df['Noise inter-processes'], errors='coerce')
    df['Noise intra-process'] = pd.to_numeric(df['Noise intra-process'], errors='coerce')

    Y = df['Noise intra-process']
    X = df['Noise inter-processes']



    colors = wutils.generate_colors(len(column))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For legend
    legend_elements = []

    for i, c in enumerate(column):
        Z = df[c]
        # xx, yy, zz = regressionPlane(X, Z)
        ax.bar3d(X, Y, Z,0.5, 0.5, 0.5, color=colors[i])

        # Add legend entry for each surface
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Surface regression for {c}'))

    # Scatter plot of the original points
   # ax.scatter(df['Noise inter-processes'], df['Noise intra-process'], df[column], color='blue', s=50)

    # Axes labels
    ax.set_xlabel('Noise inter-processes')
    ax.set_ylabel('Noise intra-process')
    ax.set_zlabel('Metrics values')

    ax.legend(handles=legend_elements, loc='upper left', fontsize='small',
              bbox_to_anchor=(-0.3, 1.15))

    plt.savefig( 'Metrics_3D.pdf')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 6:
        test_name=sys.argv[1]
        structures_name = sys.argv[2]
        predicates_name=sys.argv[3]
        ground_truth_structure= sys.argv[4]
        ground_truth_predicate= sys.argv[5]
        files_dict=wutils.file_to_dict(test_name)
        k_values=[1,2,5,7]
        classical_metrics=['ACCURACY','RECALL','PRECISION','RULE SCORE']
        columns= ['TEST','Noise inter-processes','Noise intra-process']
        columns=columns+ classical_metrics
        heuristics=list()
        for k in k_values:
            columns.append('HEURISTIC WITH k='+str(k))
            heuristics.append('HEURISTIC WITH k='+str(k))
        ground_truth = pd.read_csv(ground_truth_structure, sep=',')
        ground_truth_preds = pd.read_csv(ground_truth_predicate, sep=',')
        results=pd.DataFrame(columns=columns)
        folder=os.getcwd()
        folder=folder+('\\TEST')
        pattern = os.path.join(folder,  "*"+structures_name )
        file_list = glob.glob(pattern)
        pattern= os.path.join(folder,  "*"+predicates_name )
        file_list_preds = glob.glob(pattern)
        deltas=dict()
        row=dict()
        row['TEST']='original_dataset'
        row['Noise inter-processes']=0
        row['Noise intra-process']=0
        original_dataset= pd.read_csv(file_list[0], sep=',')
        original_dataset_preds=pd.read_csv(file_list_preds[0],sep=',')
        original_metric=computeMetrics(original_dataset,original_dataset_preds,ground_truth,ground_truth_preds,k_values)
        row=row | original_metric
        results.loc[len(results)]=row
        for i in range(1,len(file_list)):
            row=dict()
            filename=file_list[i]
            number=utils.numberTest(filename)
            row['TEST']=number
            row['Noise inter-processes'] = files_dict[number][0]
            row['Noise intra-process'] = files_dict[number][1]
            dataset = pd.read_csv(file_list[i])
            dataset_pred=pd.read_csv(file_list_preds[i])
            metrics=computeMetrics(dataset,dataset_pred,ground_truth,ground_truth_preds,k_values)
            row = row | metrics
            results.loc[len(results)] = row
        results['Noise'] = list(zip(results['Noise inter-processes'], results['Noise intra-process']))
        graph_name='classical_metrics'
        plot(results,classical_metrics,graph_name,'Noise')
        values=results['Noise inter-processes'].unique()
        values=values[1:]
        for v in values:
            plot(results, classical_metrics, graph_name, 'Noise intra-process','Noise inter-processes',v)
        values = results['Noise intra-process'].unique()
        values = values[1:]
        for v in values:
            plot(results, classical_metrics, graph_name,'Noise inter-processes', 'Noise intra-process',v)
        tridimensionalPlot(results,['RECALL'])
        #boxPlot(results, classical_metrics, graph_name)
        #deltas_df=computeDeltas(results,heuristics)
        #graph_name = 'heuristics_metrics'
        #plot(results, heuristics, graph_name,'Noise ')
        #boxPlot(results, heuristics, graph_name)
        #for h in heuristics:
            #tridimensionalPlot(results,h,classical_metrics)
        grouped_df = results.groupby('Noise').agg(['mean', 'var'])
        grouped_df.to_csv('accuracy.csv', index=False)



