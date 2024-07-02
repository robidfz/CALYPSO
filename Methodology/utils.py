import pandas as pd
import numpy as np


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
            #if (not any(c in element for element in activities[activities_col_name])):
            new_row = dict()
            for column in col:
                new_row[column]=None

            new_row[activities_col_name] = c+"_is_up"
            new_row[timestamps_col_name]=timestamp
            new_row[caseIDs_col_name]=caseID
            new_rows.append(new_row)
    new_rows_df = pd.DataFrame(new_rows, columns=col)
    df = pd.concat([df, new_rows_df], ignore_index=True)
    df = df.sort_values(by=timestamps_col_name)
    return df
def computingProbability(df, cause, effect=None):
    if (effect != None):
        c_and_x = df[(df[effect] == 1) & (df[cause] == 1)].shape[0]
        c = df[df[cause] == 1].shape[0]
        probability = c_and_x / c
    else:
        c = df[df[cause] == 1].shape[0]
        all = df.shape[0]
        probability = c / all
    return probability

def findComponent(components, name):
    for component in components:
        if component.name == name:
            return component
    return None

def matchingStrings(s1, s2, operator):
    elements = s1.split("_" + operator + "_")
    check = any(e in s2 for e in elements)
    return check

def matchingElements(s1, s2, operator):
    elements1 = s1.split("_" + operator + "_")
    elements2 = s2.split("_" + operator + "_")
    check = (set(elements1) == set(elements2))
    return check

def findOperator(s):
    operator = 'AND'
    index = s.find(operator)
    if (index == -1):
        operator = 'OR'
    return operator

def ends_with_any(value, suffixes):
    if isinstance(value, str):
        return any(value.endswith(suffix) for suffix in suffixes)
    return False
def definePredicate(cause1,cause2, operand):
    predicate = cause1 + '_' + operand + '_' + cause2
    return predicate

def splitPredicate(s):
    operator = findOperator(s)
    elements = s.split("_" + operator + "_")
    return elements

def splitActivity(activity):
    index0 = activity.find('_')
    elem = activity[index0 + 1:]
    index = elem.find('_') + index0 + 1
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
        cause_effect_dict[effect] = new_causes

    return cause_effect_dict