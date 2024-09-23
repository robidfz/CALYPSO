import sys
import pandas as pd
from Methodology import utils
import os
import glob
from configparser import ConfigParser

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

def computePredicateScore(dataset,ground_truth,file):
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
    file.write('PREDICATE SCORE: '+str(score)+'\n')

    return score

def computeMetrics(dataset,ground_truth,file):
    falsePositive, truePositive = computeFalseAndTruePositive(dataset, ground_truth)
    falseNegative = computeFalseAndTrureNegative(dataset, ground_truth)
    accuracy = truePositive  / (falsePositive + falseNegative + truePositive)
    recall = truePositive / (truePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    metrics=dict()
    metrics['ACCURACY']=accuracy
    metrics['RECALL']=recall
    metrics['PRECISION']=precision
    file.write('ACCURACY: ' + str(accuracy) + '\n')
    file.write('RECALL: ' + str(recall) + '\n')
    file.write('PRECISION: ' + str(precision) + '\n')
    metrics=computeHeuristics(dataset, k_values, file, metrics)
    return metrics


def heuristicsDefinition(dataset,k):
    values=dataset['Epsilon']
    values_number=dataset.shape[0]
    sum=0
    for v in values:
        if(v>=0.5):
            sum=sum+(v*k)
        else:
            sum=sum+(v/k)
    sum=sum/values_number
    return sum

def computeHeuristics(dataset,k_value,file,metric):

    for k in k_value:
        value=heuristicsDefinition(dataset,k)
        key='HEURISTIC WITH '+str(k)
        metric[key]=value
        file.write(key+': ' + str(value) + '\n')
    key='MEAN'
    value=dataset['Epsilon'].mean()
    metric[key]=value
    file.write('MEAN: ' + str(value) + '\n')
    return metric

def computeDeltas(metric_original,metric,file):
    result = {key: (metric_original[key] - metric[key])/original_metric[key] for key in metric_original if key in metric}
    file.write('DELTA FOR METRIC:\n')
    for key,values in result.items():
        file.write(key+': '+str(values)+'\n')
    return result



if __name__ == "__main__":
    if len(sys.argv) == 6:
        file=open('report_evaluation.txt','w')
        configfile_name=sys.argv[1]
        structures_name = sys.argv[2]
        predicates_name=sys.argv[3]
        ground_truth_structure= sys.argv[4]
        ground_truth_predicate= sys.argv[5]
        reader = ConfigParser()
        reader.read(configfile_name)
        metrics= eval(reader['EVALUATION']['metrics'])
        metrics.insert(0,'TEST')
        results=pd.DataFrame(columns=metrics)
        folder=os.getcwd()
        pattern = os.path.join(folder, structures_name + "*")
        file_list = glob.glob(pattern)
        pattern= os.path.join(folder, predicates_name + "*")
        file_list_preds = glob.glob(pattern)
        deltas=dict()
        ground_truth = pd.read_csv(ground_truth_structure, sep=',')
        ground_truth_preds=pd.read_csv(ground_truth_predicate, sep=',')
        file.write('TEST0:\n')
        row=dict()
        row['TEST']='original_dataset'
        original_dataset= pd.read_csv(file_list[0], sep=',')
        original_dataset_preds=pd.read_csv(file_list_preds[0],sep=',')
        k_values=[1,2,5,7]
        original_metric=computeMetrics(original_dataset,ground_truth,file)
        pred_score=computePredicateScore(original_dataset_preds,ground_truth_preds,file)
        row=row | original_metric
        row['PREDICATE SCORE']=pred_score
        results.loc[len(results)]=row
        file.write('\n\n\n')
        for i in range(1,len(file_list)):
            row=dict()
            filename=file_list[i]
            file.write(file_list[i]+':\n')
            number,test_name_csv=utils.numberTest(filename)
            row['TEST']='TEST'+str(number)
            #ground_truth_name='ground_truth.csv'
            #dataset_name_new=dataset_name+test+'.csv'
            dataset = pd.read_csv(file_list[i])
            dataset_pred=pd.read_csv(file_list_preds[i])
            metrics=computeMetrics(dataset,ground_truth,file)
            row = row | metrics
            pred_score=computePredicateScore(dataset_pred,ground_truth_preds,file)
            row['PREDICATE SCORE'] = pred_score
            results.loc[len(results)] = row
            res=computeDeltas(original_metric, metrics, file)
            deltas[file_list[i]]=res
            file.write('\n\n\n\n\n')
        results.to_csv('accuracy.csv', index=False)



        file.close()

