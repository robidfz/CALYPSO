import pandas as pd
import numpy as np
import itertools
from numpy import array, linspace
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np
import re
import random
from sklearn.cluster import DBSCAN

def addNoise(df,activities_col_name, caseIDs_col_name, timestamps_col_name,percentage_in_caseID, percentage_of_caseID):
    percentage_in_caseID=int(percentage_in_caseID)
    percentage_of_caseID=int(percentage_of_caseID)
    activities=df[activities_col_name].unique().tolist()
    cases=df[caseIDs_col_name].unique().tolist()
    times= int((len(cases)*percentage_of_caseID)/100)
    for i in range(times):
        case = random.choice(cases)
        timestamps=df[df[caseIDs_col_name]==case][timestamps_col_name]
        timestamp_min = min(timestamps)
        timestamp_max = max(timestamps)
        number_rows_tot=len(timestamps)
        number_new_rows=int((number_rows_tot*percentage_in_caseID)/100)
        for j in range(number_new_rows):
            act = random.choice(activities)
            timestamp=random.uniform(timestamp_min,timestamp_max)
            new_line = {caseIDs_col_name:case,timestamps_col_name:timestamp,activities_col_name:act}
            df = df.append(new_line, ignore_index=True)
    df=df.sort_values(by=timestamps_col_name)
    #df.to_csv('Threat_to_validity_TEST'+str(number_test)+'.csv',index=False)
    return df

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

def addColumn1(dataset,column_dict,features,output):
    new_col = list()
    for i, row in dataset.iterrows():
        for key,values in column_dict.items():
            if(row[features].startswith(key)):
                new_val = row[features]+"_"+str(row[values])
                new_val=new_val.replace(' ','_')
                new_col.append(new_val)
    dataset[output] = new_col
    return dataset
def addColumn2(dataset,feature,value,output):
    new_col=list()
    for i, row in dataset.iterrows():
        val = row[feature].replace(" ", "_")
        flag=0
        for v in value:
            if(val.endswith(v[1])):
                elem= v[0]
                new_col.append(elem)
                flag=1
        if(flag==0):
            elem=value[2][0]
            new_col.append(elem)

    dataset[output] = new_col
    return dataset

def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return value

def addingUpState(df,activities_col_name,caseIDs_col_name,timestamps_col_name):
    act = df[activities_col_name].unique()
    components = extractComponents(act)
    cases = df.groupby(caseIDs_col_name)
    col=df.columns
    new_rows=list()
    for case, activities in cases:

        timestamp=min(activities[timestamps_col_name].values)
        caseID=activities[caseIDs_col_name].unique()[0]

        for c in components:
                new_row = dict()
                for column in col:
                    new_row[column]=None

                new_row[activities_col_name] = c+"_is_up"
                new_row[timestamps_col_name]=timestamp-1
                new_row[caseIDs_col_name]=caseID
                new_rows.append(new_row)
    new_rows_df = pd.DataFrame(new_rows, columns=col)
    df = pd.concat([df, new_rows_df], ignore_index=True)
    df = df.sort_values(by=timestamps_col_name)
    return df
def extractPossibleTransition(df,activity_col_name,caseID_col_name,components):
    possible_activities = df[activity_col_name].unique()

    transition_dict=dict()
    for c in components:
        component_activities=[a for a in possible_activities if a.startswith(c)]
        filtered_df=df[df[activity_col_name].isin( component_activities)]
        grouped_df=filtered_df.groupby(caseID_col_name)
        possible_transitions = list()

        for group_name,group in grouped_df:
            index=0
            while index < len(group)-1:
                row = group.iloc[index]
                following_row=group.iloc[index+1]
                act_initial=row[activity_col_name]
                component,state_initial=splitActivity(act_initial)
                act_final=following_row[activity_col_name]
                component,state_final=splitActivity(act_final)
                if(state_final!=state_initial):
                    transition=[state_initial,state_final]
                    transition_actv=defineTransitionActivities(component,transition)
                    if(transition_actv not in possible_transitions):
                        possible_transitions.append(transition_actv)
                index=index+1
        transition_dict[c]=possible_transitions



    return transition_dict


def extractPossibleSignalsEvents(df,dynamics_name,activity_col_name):
    retval=list()
    all_activities=df[activity_col_name].unique()
    for d in dynamics_name:
        retval=retval+[a for a in all_activities if a.startswith(d)]
    return retval


def computingProbabilityold(df, cause, effect=None):
    if (effect != None):
        c_and_x = df[(df[effect] == 1) & (df[cause] == 1)].shape[0]
        c = df[df[cause] == 1].shape[0]
        probability = c_and_x / c
    else:
        c = df[df[cause] == 1].shape[0]
        all = df.shape[0]
        probability = c / all
    return probability

def findTransition(group,transition,activity_col_name,timestamp_col_name):
    state_initial,state_final=splitTransitionActivities(transition)
    values=[state_initial,state_final]
    if(all([x in group[activity_col_name].values for x in values])):
        state_initial_timestamp=min(group[group[activity_col_name]==state_initial][timestamp_col_name].values)
        state_final_timestamp = max(group[group[activity_col_name] == state_final][timestamp_col_name].values)
        if(state_initial_timestamp<=state_final_timestamp):
            transition_timestamp=state_final_timestamp
        else:
            transition_timestamp=-1
    else:
        transition_timestamp=-1
    return transition_timestamp


def computingFrequencies(df, cause,activity_col_name,caseID_col_name, timestamp_col_name,delta=None,effect=None):
    grouped_df = df.groupby(caseID_col_name)
    cause_freq = 0
    for group_name,group in grouped_df:
        cause_timestamp = findPredicateTimestamp(group, cause, activity_col_name, timestamp_col_name)
        if (cause_timestamp != -1):
            if (effect != None):
                effect_timestamp = findPredicateTimestamp(group, effect, activity_col_name, timestamp_col_name)
                if (effect_timestamp != -1 and (effect_timestamp - cause_timestamp >= 1 and effect_timestamp - cause_timestamp < delta)):
                    cause_freq = cause_freq + 1
            else:
                cause_freq = cause_freq + 1
    return cause_freq
def computingProbability(df, cause,activity_col_name,caseID_col_name, timestamp_col_name, delta=None,effect=None):

    n_cases=len(df[caseID_col_name].unique())
    if(delta==None):
        delta=max(df[timestamp_col_name].values)
    cause_freq=computingFrequencies(df,cause,activity_col_name,caseID_col_name,timestamp_col_name)
    if(effect!=None):
        causeandeffect_freq = computingFrequencies(df,cause,activity_col_name,caseID_col_name,timestamp_col_name,delta,effect)
        if(cause_freq!=0):
            probability = causeandeffect_freq / cause_freq
        else:
            probability=-1
    else:
        probability=cause_freq/n_cases
    return probability


def findComponent(components, name):
    for component in components:
        if component.name == name:
            return component
    return None

def matchingStrings(s1, s2, operator):
    elements = s1.split("_" + operator + "_")
    components=list()
    for e in elements:
        component,actv=splitActivity(e)
        components.append(component)
    check = any(e in s2 for e in components)
    return check

def matchingElements(s1, s2, operator):
    elements1 = s1.split("_" + operator + "_")
    elements2 = s2.split("_" + operator + "_")
    check = (set(elements1) == set(elements2))
    return check



def findOperator(s):
    possible_operatores=['AND','OR','NOT','>']
    i=0
    index=-1
    while(i<len(possible_operatores) and index==-1):
        index = s.find(possible_operatores[i])
        operator=possible_operatores[i]
        i=i+1
    if (index == -1):
        operator = -1
    return operator
def composedPredicate(elems,operator):
    result = elems[0]
    for s in elems[1:]:
        result = definePredicate(result, s,operator)
    return result
def combinations(elem,operator):
    permutations_list = list(itertools.permutations(elem))
    results = [composedPredicate(perm,operator) for perm in permutations_list]
    return results


def ends_with_any(value, suffixes):
    if isinstance(value, str):
        return any(value.endswith(suffix) for suffix in suffixes)
    return False
def definePredicate(cause1,cause2, operand):
    predicate = cause1 + '_' + operand + '_' + cause2
    return predicate

def splitPredicate(s):
    operator = findOperator(s)
    if(operator==-1):
        elements =[s]
    else:
        if(operator=='NOT'):
            elements = s.split(operator + "_")
        else:
            elements = s.split('_'+operator + "_")
    return elements

def splitActivity(activity):
    index0 = activity.find('_')
    if(index0!=-1):

        elem = activity[index0 + 1:]
        index = elem.find('_') + index0 + 1

    else:
        index = activity.find('>')
    elem = activity[:index]
    occurence_name = activity[index + 1:]
    return elem, occurence_name

def defineActivity(elem, occurrence):
    return elem + '_' + occurrence

def defineTransitionActivities(component, transition):
    initial_state = transition[0]
    final_state = transition[1]
    t = component + "_" + initial_state + "\\" + final_state
    return t

def splitTransitionActivities(transition_activity):

    transition_activity_index = transition_activity.rfind('\\')

    if(transition_activity_index!=-1):
        c, act = splitActivity(transition_activity)
        final_state = c + "_" + transition_activity[transition_activity_index + 1:]
        index = transition_activity[:transition_activity_index].rfind("_")
        index_1 = transition_activity[:index].rfind("_")
        initial_state = c + "_" + transition_activity[index_1 + 1:transition_activity_index]
    else:
        initial_state = transition_activity
        final_state =transition_activity
    return initial_state, final_state

def extractTransition(effect):
    initial_state, final_state = splitTransitionActivities(effect)
    e_comp, initial_state = splitActivity(initial_state)
    e_comp, final_state = splitActivity(final_state)
    transition_effect = (initial_state, final_state)
    return transition_effect

def extractComponents(activities):
    activities = pd.Series(activities)
    result = activities.apply(splitActivity)
    components = result.apply(lambda x: x[0]).drop_duplicates()
    mask = ~components.str.startswith('sig')
    components = components[mask]
    return components
'''
def extractDynamics(activities):
    activities = pd.Series(activities)
    result = activities.apply(splitActivity)
    dynamics = result.apply(lambda x: x[0]).drop_duplicates()
    dynamics =dynamics[dynamics.str.startswith('sig')]

    return dynamics
    '''

def splitDynamicCauses(cause):
    index = cause.find('>')
    threshold = cause[index + 1:]
    elem = cause[:index]

    return elem, threshold





def filteringSignificantCauses(eps_avg_dict, maximum=False):
    cause_effect_dict = dict()
    for effect, values in eps_avg_dict.items():
        causes = [c[0] for c in values]
        eps = [c[1] for c in values]
        if (maximum):
            try:
                max_value = max(eps)
            except:
                max_value = 0
            new_causes = [(value[0], value[1]) for value in values if value[1] == max_value]
        else:
            mean = np.mean(eps)
            std_dev = np.std(eps)
            filtered_causes = list()
            threshold = 0
            if (std_dev > 0.1):
                threshold = mean
                new_causes = [(value[0], value[1]) for value in values if value[1] >= threshold]
            else:
                new_causes=[(value[0], value[1]) for value in values ]
        cause_effect_dict[effect] = new_causes

    return cause_effect_dict

def filteringLatestCausesold(eps_avg,effect,df,activities_col_name,caseIDs_col_name, timestamps_col_name):
    filtered_eps = eps_avg.copy()
    for i, eps in enumerate(eps_avg):
        latest_cause = eps[0]
        for j in range(i + 1, len(eps_avg)):

            retval = checkLatestCause(latest_cause, eps_avg[j][0], effect, df, activities_col_name,caseIDs_col_name, timestamps_col_name)
            if(retval!=-1):
                latest_cause=retval
                if(latest_cause!=eps_avg[j][0] and eps_avg[j][0] in filtered_eps):
                    filtered_eps.remove([eps_avg[j][0],eps_avg[j][1]])
                else:
                    if(eps_avg[i][0] in filtered_eps):
                        filtered_eps.remove([eps_avg[i][0], eps_avg[i][1]])


    return filtered_eps







def filteringLatestCauses(A, effect, df, activities_col_name, caseIDs_col_name, timestamps_col_name):
    filtered_eps = A.copy()
    i = 0
    while i < len(filtered_eps):
        j = i + 1
        while j < len(filtered_eps):
            retval = checkLatestCause(filtered_eps[i], filtered_eps[j], effect, df, activities_col_name, caseIDs_col_name, timestamps_col_name)
            if retval != -1:
                latest_cause = retval
                if latest_cause == filtered_eps[j]:
                    filtered_eps.pop(i)
                    break  # Restart from the beginning after removing an element
                elif latest_cause == filtered_eps[i]:
                    filtered_eps.pop(j)
                else:
                    j += 1
            else:
                j += 1
        else:
            i += 1
    return filtered_eps

def findPredicateTimestamp(group,cause,activities_col_name,timestamps_col_name):
    operator=findOperator(cause)
    preds = splitPredicate(cause)
    retval=None
    j = 0
    times=list()
    if(operator==-1 or operator=='>'):
        cause_i, cause_f = splitTransitionActivities(preds[0])
        if(cause_i==cause_f):
            check1 = cause_i in group[activities_col_name].values
            if(check1):
                times_f = min(group[group[activities_col_name] == cause_i][timestamps_col_name].values)
                retval = times_f
            else:
                retval=-1
        else:
            check1 = cause_i in group[activities_col_name].values
            check2 = cause_f in group[activities_col_name].values
            if (check1 and check2):
                times_i = min(
                    group[group[activities_col_name] == cause_i][timestamps_col_name].values)
                times_f = max(
                    group[group[activities_col_name] == cause_f][timestamps_col_name].values)

                if (times_i < times_f):
                    retval=times_f
                else:
                    retval=-1
            else:
                retval=-1

    if(operator=='OR'):
        while (j < len(preds)):
            single_cause_timestamp=findPredicateTimestamp(group,preds[j],activities_col_name,timestamps_col_name)
            if(single_cause_timestamp!=-1):
                times.append(single_cause_timestamp)
            j = j + 1
        if (len(times) > 0):
            retval = min(times)
        else:
            retval=-1
    if(operator=='AND'):
        andcondition = True
        j = 0
        times = list()
        while (andcondition and j < len(preds)):
            single_cause_timestamp=findPredicateTimestamp(group,preds[j],activities_col_name,timestamps_col_name)
            times.append(single_cause_timestamp)
            andcondition=(single_cause_timestamp!=-1)
            j = j + 1
        if andcondition:
            retval = max(times)
        else:
            retval = -1
    if (operator == 'NOT'):
        causes=splitPredicate(cause)
        cause=causes[-1]
        single_cause_timestamp = findPredicateTimestamp(group, cause, activities_col_name,timestamps_col_name)
        if(single_cause_timestamp==-1):
            retval=min(group[timestamps_col_name].values)
        else:
            retval=-1



    return retval


def findPredicateTimestamp_old(group,cause,activities_col_name,timestamps_col_name):
    operator=findOperator(cause)
    preds = splitPredicate(cause)
    causes_initials=list()
    causes_finals=list()
    components=list()
    for e in preds:
        cause1_initial, cause1_final = splitTransitionActivities(e)
        causes_initials.append(cause1_initial)
        causes_finals.append(cause1_final)
        component, act = splitActivity(e)
        components.append(component)
    j = 0
    times=list()
    if(operator=='OR' or operator==-1):
        while (j < len(causes_finals)):
            cause_f = causes_finals[j]
            cause_i = causes_initials[j]
            check1 = cause_i in group[activities_col_name].values
            check2 = cause_f in group[activities_col_name].values
            if (check1 and check2):
                times_i = min(
                    group[group[activities_col_name] == cause_i][timestamps_col_name].values)
                times_f = max(
                    group[group[activities_col_name] == cause_f][timestamps_col_name].values)
                if(cause_i==cause_f):
                    times(times_f)
                else:
                    if (times_i < times_f):
                        times.append(times_f)
            j = j + 1
        if (len(times) > 0):
            retval = min(times)
        else:
            retval=-1
    if(operator=='AND'):
        andcondition = True
        j = 0
        times = list()
        while (andcondition and j < len(causes_finals)):
            cause_f = causes_finals[j]
            cause_i = causes_initials[j]
            check1 = cause_i in group[activities_col_name].values
            check2 = cause_f in group[activities_col_name].values
            if (check1 and check2):
                times_i = min(group[group[activities_col_name] == cause_i][timestamps_col_name].values)
                times_f = max(group[group[activities_col_name] == cause_f][timestamps_col_name].values)
                if (times_i < times_f):
                    times.append(times_f)
                else:
                    andcondition = False
            else:
                andcondition = False
            j = j + 1
        if andcondition:
            retval = max(times)
        else:
            retval = -1
    return retval

def checkLatestCauseold(cause1,cause2,effect,df,activity_col_name,caseID_col_name,timestamp_col_name):


    effect_intial,effect_final=splitTransitionActivities(effect)
    values={effect_intial,effect_final}
    cases=df[caseID_col_name].unique()
    previous_activity=cause1
    real_possible_cause=cause2
    condition=True
    i=0
    find_previous=0
    while(condition and i<len(cases)):
        group=df[df[caseID_col_name]==cases[i]]
        if( all(v in group[activity_col_name].values for v in values)):
            effect_final_timestamp = max(group[group[activity_col_name] == effect_final][timestamp_col_name])
            effect_initial_timestamp = min(group[group[activity_col_name] == effect_intial][timestamp_col_name])
            if(effect_final_timestamp>effect_initial_timestamp):
                previous_activity_timestamp = findPredicateTimestamp(group, previous_activity, activity_col_name,timestamp_col_name)
                real_possible_cause_timestamp = findPredicateTimestamp(group, real_possible_cause, activity_col_name,timestamp_col_name)
                if(previous_activity_timestamp!=-1 and real_possible_cause_timestamp!=-1):
                    if(find_previous == 0):
                        #find the possible latest cause
                        if (real_possible_cause_timestamp < previous_activity_timestamp):
                            temp =previous_activity
                            temp_timestamp=previous_activity_timestamp
                            previous_activity_timestamp=real_possible_cause_timestamp
                            real_possible_cause_timestamp=temp_timestamp
                            previous_activity = real_possible_cause
                            real_possible_cause = temp
                        find_previous = find_previous + 1
                    condition = previous_activity_timestamp <= real_possible_cause_timestamp
                elif((previous_activity_timestamp != -1) and (real_possible_cause_timestamp == -1) and find_previous!=0):
                    condition = False
                #elif (previous_activity_timestamp != -1) and (real_possible_cause_timestamp == -1):
        i = i + 1


    if(condition and find_previous!=0):
        retval=real_possible_cause
    else:
        retval=-1

    return retval


def checkLatestCause(cause1,cause2,effect,df,activity_col_name,caseID_col_name,timestamp_col_name):



    cases=df[caseID_col_name].unique()
    previous_activity=cause1
    real_possible_cause=cause2
    condition=True
    i=0
    find_previous=0
    while(condition and i<len(cases)):
        group=df[df[caseID_col_name]==cases[i]]
        effect_timestamp=findPredicateTimestamp(group,effect,activity_col_name,timestamp_col_name)
        if(effect_timestamp!=-1):
            previous_activity_timestamp = findPredicateTimestamp(group, previous_activity, activity_col_name,timestamp_col_name)
            real_possible_cause_timestamp = findPredicateTimestamp(group, real_possible_cause, activity_col_name,timestamp_col_name)
            if(previous_activity_timestamp!=-1 and real_possible_cause_timestamp!=-1):
                if(find_previous == 0):
                    #find the possible latest cause
                    if (real_possible_cause_timestamp < previous_activity_timestamp):
                        temp =previous_activity
                        temp_timestamp=previous_activity_timestamp
                        previous_activity_timestamp=real_possible_cause_timestamp
                        real_possible_cause_timestamp=temp_timestamp
                        previous_activity = real_possible_cause
                        real_possible_cause = temp
                    find_previous = find_previous + 1
                condition = previous_activity_timestamp < real_possible_cause_timestamp
            elif((previous_activity_timestamp != -1) and (real_possible_cause_timestamp == -1) and find_previous!=0):
                condition = False
            #elif (previous_activity_timestamp != -1) and (real_possible_cause_timestamp == -1):
        i = i + 1


    if(condition and find_previous!=0):
        retval=real_possible_cause
    else:
        retval=-1

    return retval


def checkPossibleTransition(transition,activities):
    inital_activity,final_activity=splitTransitionActivities(transition)
    condition=(inital_activity in activities) and (final_activity in activities)
    return condition


def negateOperator(operator):
    if (operator == 'AND'):
        new_operator = 'OR'
    else:
        new_operator = 'AND'
    return new_operator
def negateCause(cause):
    operator=findOperator(cause)
    if operator!=-1:
        elements = splitPredicate(cause)
        neg_elements=['NOT_'+ e for e in elements ]
        new_operator=negateOperator(operator)
        connector='_'+new_operator+'_'
        not_cause = connector.join(neg_elements)

    else:
        not_cause='NOT_'+cause
    return not_cause


def difference_between_rows(row1, row2):

    return [
        a - b if a is not None and b is not None else None
        for a, b in zip(row1, row2)
    ]

def comparePredicates(pred1,pred2):
    retval=False
    operator_pred1 = findOperator(pred1)
    operator_pred2 = findOperator(pred2)
    if (operator_pred1 == operator_pred2):
        elements_pred1 = splitPredicate(pred1)
        elements_pred2 = splitPredicate(pred2)
        if (sorted(elements_pred1) == sorted(elements_pred2)):
            retval=True
    return retval


def numberTest(filename):
    match = re.search(r'TEST(\d+)', filename)

    if match:
        numero = match.group(1)

    else:
        numero=None
    return numero