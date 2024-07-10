import Methodology.utils as utils
class PrimaFacie:
    def __init__(self,df,caseIDs_col_name,activities_col_name,timestamps_col_name,window_matrix):
        self.df=df
        self.caseIDs_col_name=caseIDs_col_name
        self.activities_col_name=activities_col_name
        self.timestamps_col_name=timestamps_col_name
        self.window_matrix=window_matrix



    def primafacie_condition1(self, effect, cause,delta=None):
        cases = self.df[self.caseIDs_col_name].unique()

        condition=False
        i=0
        flag=False
        while(condition==False and i<len(cases)):
            group=self.df[self.df[self.caseIDs_col_name]==cases[i]]
            effect_timestamp = utils.findPredicateTimestamp(group, effect, self.activities_col_name, self.timestamps_col_name)
            cause_timestamp = utils.findPredicateTimestamp(group, cause, self.activities_col_name,self.timestamps_col_name)
            if (effect_timestamp != -1 and cause_timestamp!=-1):
                flag=True
                condition=(effect_timestamp - cause_timestamp>= 1) and (effect_timestamp - cause_timestamp < delta)
            i=i+1
        #condition=(condition and flag)
        return condition






    def primafacie_condition1_old(self, effect, cause):
        cases = self.df[self.caseIDs_col_name].unique()
        condition1 = True
        flag = False
        i = 0
        effect_initial, effect_final = utils.splitTransitionActivities(effect)
        if (effect_initial == effect_final):
            effect = effect_final
            while (condition1 and i < len(cases)):
                group = self.df[self.df[self.caseIDs_col_name] == cases[i]]
                if (effect_initial in group[self.activities_col_name].values and effect in group[
                    self.activities_col_name].values):
                    if (cause in group[self.activities_col_name].values):
                        flag = True
                        cause_timestamp = \
                        group[group[self.activities_col_name] == cause][self.timestamps_col_name].values[0]
                        effect_timestamp = \
                        group[group[self.activities_col_name] == effect][self.timestamps_col_name].values[0]
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
                        andcondition=True
                        j=0
                        while (andcondition and j<len(causes_finals)):
                            cause_f= causes_finals[j]
                            cause_i=causes_initials[j]
                            check1=cause_i in group[self.activities_col_name].values
                            check2=cause_f in group[self.activities_col_name].values
                            if(check1 and check2):
                                times_i=min(group[group[self.activities_col_name]==cause_i][self.timestamps_col_name].values)
                                times_f = max(group[group[self.activities_col_name] == cause_f][self.timestamps_col_name].values)
                                if(times_i<times_f):
                                    times.append(times_f)
                                else:
                                    andcondition=False
                            else:
                                andcondition=False
                            j=j+1
                        if(andcondition):
                            flag=True
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
                        effect_timestamp = max(
                            group[group[self.activities_col_name] == effect_final][self.timestamps_col_name].values)
                        j = 0
                        while (j < len(causes_finals)):
                            cause_f = causes_finals[j]
                            cause_i = causes_initials[j]
                            check1 = cause_i in group[self.activities_col_name].values
                            check2 = cause_f in group[self.activities_col_name].values
                            if (check1 and check2):
                                times_i = min(
                                    group[group[self.activities_col_name] == cause_i][self.timestamps_col_name].values)
                                times_f = max(
                                    group[group[self.activities_col_name] == cause_f][self.timestamps_col_name].values)
                                if (times_i < times_f):
                                    times.append(times_f)
                            j = j + 1
                        if (len(times)>0):
                            flag = True
                            cause_timestamp = min(times)
                            condition1 = (cause_timestamp < effect_timestamp)
                    i = i + 1
            condition1 = (condition1 and flag)
        return condition1


    def primafacie_condition2and3(self,effect, cause,delta=None):

        condition3 = False
        cause_probability = utils.computingProbability(self.df,cause,self.activities_col_name,self.caseIDs_col_name,self.timestamps_col_name,delta)
        condition2 = (cause_probability > 0)
        if (condition2):
            conditional_probability = utils.computingProbability(self.df,cause,self.activities_col_name,self.caseIDs_col_name,self.timestamps_col_name,delta,effect)
            effect_probability = utils.computingProbability(self.df,effect,self.activities_col_name,self.caseIDs_col_name,self.timestamps_col_name,delta)
            condition3 = (conditional_probability >= effect_probability)
        pf = (condition2 and condition3)
        return pf

    def primafacie(self,effect, cause,delta=None):
        if (delta == None):
            delta = max(self.df[self.timestamps_col_name].values)

        cond1 = self.primafacie_condition1(effect, cause,delta)
        if (cond1):
            cond2_3 = self.primafacie_condition2and3(effect, cause,delta)
        else:
            cond2_3 = False
        pf = (cond1 and cond2_3)

        return pf

    def updateWindowMatrix(self,window_matrix):
        self.window_matrix=window_matrix