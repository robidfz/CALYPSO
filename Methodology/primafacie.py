import Methodology.utils as utils
class PrimaFacie:
    def __init__(self,df,caseIDs_col_name,activities_col_name,timestamps_col_name,window_matrix):
        self.df=df
        self.caseIDs_col_name=caseIDs_col_name
        self.activities_col_name=activities_col_name
        self.timestamps_col_name=timestamps_col_name
        self.window_matrix=window_matrix

    def primafacie_condition1(self,effect, cause):
        cases = self.df[self.caseIDs_col_name].unique()
        condition1 = True
        flag = False
        i = 0
        effect_initial, effect_final = utils.splitTransitionActivities(effect)
        if(effect_initial==effect_final):
            effect=effect_final
            while (condition1 and i < len(cases)):
                group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                if (cause in group[self.activities_col_name].values):
                    cause_timestamp = group[group[self.activities_col_name] == cause][self.timestamps_col_name].values[0]
                    if (effect_initial in group[self.activities_col_name].values and effect in group[self.activities_col_name].values):
                        flag = True
                        effect_timestamp = group[group[self.activities_col_name] == effect][self.timestamps_col_name].values[0]
                        condition1 = (cause_timestamp < effect_timestamp)
                i = i + 1
            condition1 = (condition1 and flag)
        else:
            operator = utils.findOperator(cause)
            preds = utils.splitPredicate(cause)
            components = list()
            causes_initials = list()
            causes_finals = list()
            for e in preds:
                cause_initial, cause_final = utils.splitTransitionActivities(e)
                causes_initials.append(cause_initial)
                causes_finals.append(cause_final)
                component, act = utils.splitActivity(e)
                components.append(component)

            cases = self.df[self.caseIDs_col_name].unique()
            condition1 = True
            flag = False
            i = 0
            if (operator == 'AND'):
                while (condition1 and i < len(cases)):
                    group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                    times = list()
                    if ((effect_initial in group[self.activities_col_name].values) and (
                            effect_final in group[self.activities_col_name].values)):
                        # selecting effect transition timestamp
                        effect_timestamp = max(
                            group[group[self.activities_col_name] == effect_final][self.timestamps_col_name].values)

                        if (all([a in group[self.activities_col_name].values for a in causes_initials]) and all(
                                [a in group[self.activities_col_name].values for a in causes_finals])):
                            flag = True
                            for a in causes_finals:
                                # selecting cause transition timestamp
                                cause_timestamp = max(group[group[self.activities_col_name] == a][
                                    self.timestamps_col_name].values)
                                times.append(cause_timestamp)
                            cause_timestamp = max(times)
                            condition1 = (cause_timestamp < effect_timestamp)
                    i = i + 1
            else:
                while (condition1 and i < len(cases)):
                    group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                    times = list()
                    if ((effect_initial in group[self.activities_col_name].values) and (
                            effect_final in group[self.activities_col_name].values)):
                        # selecting effect transition timestamp
                        effect_timestamp = max(group[group[self.activities_col_name] == effect_final][self.timestamps_col_name].values)
                        if (any([a in group[self.activities_col_name].values for a in causes_initials]) and any(
                                [a in group[self.activities_col_name].values for a in causes_finals])):
                            flag = True
                            for e in causes_finals:
                                if (e in group[self.activities_col_name].values):
                                    cause_timestamps = max(group[group[self.activities_col_name] == e][self.timestamps_col_name].values)
                                    times.append(cause_timestamps)
                            cause_timestamp = min(times)
                            condition1 = (cause_timestamp < effect_timestamp)
                    i = i + 1
            condition1 = (condition1 and flag)
        return condition1



    def findComponentState(self, group,component):



        filtered_group = group[group[self.activities_col_name].str.startswith(component)]

        if (len(filtered_group) > 0):
            # filtered_group=filtered_group[filtered_group[timestamps_col_name]<timestamp]
            closest_timestamp = filtered_group.iloc[
                (filtered_group[self.timestamps_col_name]).abs().argmax()]
            state=closest_timestamp[self.activities_col_name]
            timestamp = closest_timestamp[self.timestamps_col_name]
        else:
            closest_timestamp = group.iloc[group[self.timestamps_col_name].abs().argmax()]
            state=component + "_is_up"
            timestamp= closest_timestamp[self.timestamps_col_name]

        return state,timestamp

    def primafacie_condition1_predicates_old_old(self, effect, cause):
        operator = utils.findOperator(cause)

        preds = utils.splitPredicate(cause)
        effect_initial, effect_final = utils.splitTransitionActivities(effect)
        components = list()
        causes_initials = list()
        causes_finals = list()
        for e in preds:
            cause_initial, cause_final = utils.splitTransitionActivities(e)
            causes_initials.append(cause_initial)
            causes_finals.append(cause_final)
            component, act = utils.splitActivity(e)
            components.append(component)

        cases = self.df[self.caseIDs_col_name].unique()
        condition1 = True
        flag = False
        i = 0
        if (operator == 'AND'):
            while (condition1 and i < len(cases)):
                group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                times = list()
                if ((effect_initial in group[self.activities_col_name].values) and (
                        effect_final in group[self.activities_col_name].values)):
                    time_start=group[group[self.activities_col_name]==effect_initial][self.timestamps_col_name][0]
                    time_end=group[group[self.activities_col_name]==effect_final][self.timestamps_col_name][0]
                    # selecting effect transition timestamp
                    group=group[group[self.timestamps_col_name]>time_start & group[self.timestamps_col_name]<time_end]
                    retval=True
                    for a in preds:
                        cause_name,cause_state=utils.splitActivity(a)
                        state,timestamp=self.findComponentState(group,cause_name)
                        if(retval==True and a==state):
                            times.append(timestamp)
                        else:
                            retval=False

                    if(retval):
                        cause_timestamp = max(times)
                        condition1 = (cause_timestamp < time_end and cause_timestamp>time_start)
                    else:
                        condition1=False
                i = i + 1
        else:
            while (condition1 and i < len(cases)):
                group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                times = list()
                if ((effect_initial in group[self.activities_col_name].values) and (
                        effect_final in group[self.activities_col_name].values)):
                    #cutting in the timewindow of interest
                    time_start = group[group[self.activities_col_name] == effect_initial][self.timestamps_col_name][0]
                    time_end = group[group[self.activities_col_name] == effect_final][self.timestamps_col_name][0]
                    # selecting effect transition timestamp
                    group = group[
                        group[self.timestamps_col_name] > time_start & group[self.timestamps_col_name] < time_end]
                    retval = False
                    for a in preds:
                        cause_name, cause_state = utils.splitActivity(a)
                        state, timestamp = self.findComponentState(group, cause_name)
                        if (a == state):
                            times.append(timestamp)
                            retval = True

                    if (retval):
                        cause_timestamp = min(times)
                        condition1 = (cause_timestamp < time_end and cause_timestamp > time_start)
                    else:
                        condition1 = False
                i = i + 1
        condition1 = (condition1)
        return condition1

    def primafacie_condition2and3(self,effect, cause):

        condition3 = False
        cause_probability = utils.computingProbability(self.window_matrix, cause)
        condition2 = (cause_probability > 0)
        if (condition2):
            conditional_probability = utils.computingProbability(self.window_matrix, cause, effect)
            effect_probability = utils.computingProbability(self.window_matrix, effect)
            condition3 = (conditional_probability >= effect_probability)
        pf = (condition2 and condition3)
        return pf

    def primafacie(self,effect, cause):

        cond1 = self.primafacie_condition1(effect, cause)
        if (cond1):
            cond2_3 = self.primafacie_condition2and3(effect, cause)
        else:
            cond2_3 = False
        pf = (cond1 and cond2_3)

        return pf
