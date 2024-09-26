import Causality_analysis_modules.utils as utils
class PrimaFacie:
    def __init__(self,df,caseIDs_col_name,activities_col_name,timestamps_col_name):
        self.df=df
        self.caseIDs_col_name=caseIDs_col_name
        self.activities_col_name=activities_col_name
        self.timestamps_col_name=timestamps_col_name




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

