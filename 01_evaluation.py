import sys
import pandas as pd
from Causality_analysis_modules import utils
import os
import glob
import matplotlib.pyplot as plt
from Workflow import utils as wutils
import numpy as np
import seaborn as sns

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
    for k in k_values:
        heuristic_value=computeHeuristics(dataset, k)
        metrics['HEURISTIC WITH k='+str(k)] = heuristic_value


    return metrics






def plot(df,columns,graph_name):
    df = df.sort_values(by=['% Cases Affected by noise','% Noise added in a single Case'], ascending=[True,True])
    df_mean=df.groupby(['Noise %']).mean().reset_index()
    df['# TEST'] = df.groupby('Noise %')['Noise %'].transform('count')
    df_mean['# TEST'] = df.groupby('Noise %').size().values
    df_mean.to_csv(graph_name+'.csv',index=False)


    plt.figure(figsize=(10, 6))
    colors = wutils.generate_colors(len(columns))
    n = len(columns)
    ind = np.arange(len(df_mean['Noise %']))
    width = 0.8 / n

    for j in range(n):

        plt.bar(ind + j * width, df_mean[columns[j]], width, label=columns[j] , color=colors[j])


    plt.xticks(ind + width * (n - 1) / 2, df_mean['Noise %'])

    plt.title("Bar Plot of Metrics Mean Values by Noise %")
    plt.xlabel("(NOISE % INTER-PROCESSES, NOISE % INTRA-PROCESS)")
    plt.ylabel("GROUP METRICS MEAN VALUE")
    plt.legend(title="Metrics")
    plt.savefig(graph_name+'.pdf')



def computeDeltas(dataset,columns):
    deltas_df=dataset[columns]
    deltas_df = deltas_df.sub(deltas_df.iloc[0], axis=1).abs().drop(index=0)
    deltas_df['Noise %']=dataset['Noise %']
    deltas_df['% Cases Affected by noise'] = dataset['% Cases Affected by noise']
    deltas_df['% Noise added in a single Case'] = dataset['% Noise added in a single Case']
    return deltas_df


def boxPlot(df, columns, graph_name):

    df = df.sort_values(by=['% Cases Affected by noise', '% Noise added in a single Case'], ascending=[True, True])
    df_long = pd.melt(df, id_vars=['Noise %'], value_vars=columns, var_name='Metrics', value_name='Values')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Noise %', y='Values', hue='Metrics', data=df_long, palette='Set2')
    plt.title("Box Plot of Metrics Values by Noise %")
    plt.xlabel("(NOISE % INTER-PROCESSES, NOISE % INTRA-PROCESS)")
    plt.ylabel("GROUP METRICS MEAN VALUE")
    plt.legend(title="Metrics")
    plt.savefig('boxPlot_'+graph_name+'.pdf')


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
        columns= ['TEST','% Cases Affected by noise','% Noise added in a single Case']
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
        row['% Cases Affected by noise']=0
        row['% Noise added in a single Case']=0
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
            row['% Cases Affected by noise'] = files_dict[number][0]
            row['% Noise added in a single Case'] = files_dict[number][1]
            dataset = pd.read_csv(file_list[i])
            dataset_pred=pd.read_csv(file_list_preds[i])
            metrics=computeMetrics(dataset,dataset_pred,ground_truth,ground_truth_preds,k_values)
            row = row | metrics
            results.loc[len(results)] = row
        results['Noise %'] = list(zip(results['% Cases Affected by noise'], results['% Noise added in a single Case']))
        graph_name='classical_metrics'
        plot(results,classical_metrics,graph_name)
        boxPlot(results, classical_metrics, graph_name)
        deltas_df=computeDeltas(results,heuristics)
        graph_name = 'heuristics_metrics'
        plot(deltas_df, heuristics, graph_name)
        boxPlot(deltas_df, heuristics, graph_name)
        results.to_csv('accuracy.csv', index=False)



