import pandas as pd
import Methodology.utils as utils
class WindowMatrix:
    def __init__(self,df,caseIDs_col_name,activities_col_name,timestamps_col_name):
        self.df = df
        self.caseIDs_col_name = caseIDs_col_name
        self.activities_col_name = activities_col_name
        self.timestamp_col_name = timestamps_col_name

    def windowMatrixGeneration(self):
        col = self.df[self.activities_col_name].unique()
        cases = self.df.groupby(self.caseIDs_col_name)
        new_rows = list()
        for case, activities in cases:
            new_row = dict()
            new_row['CaseID'] = activities[self.caseIDs_col_name].unique()
            for idx, row in activities.iterrows():
                new_row[row[self.activities_col_name]] = 1
            new_rows.append(new_row)
        window_matrix = pd.DataFrame(new_rows)
        window_matrix.fillna(0, inplace=True)
        window_matrix = window_matrix[col]
        return window_matrix



    def updatingWindowMatrixPredicates(self, cause,window_matrix):
        logic_operator=utils.findOperator(cause)
        elements=utils.splitPredicate(cause)
        if (cause not in window_matrix.columns):
            values = list()
            if (logic_operator == 'AND'):
                for i, row in window_matrix.iterrows():
                    condition=True
                    j=0
                    while(condition and j<len(elements)):
                        condition=(row[elements[j]]==1)
                        j=j+1
                    if(condition):
                        values.append(1)
                    else:
                        values.append(0)

            if (logic_operator == 'OR'):
                for i, row in window_matrix.iterrows():
                    condition = True
                    j = 0
                    while (condition and j < len(elements)):
                        condition = (row[elements[j]] == 0)
                        j = j + 1
                    if (condition==False):
                        values.append(1)
                    else:
                        values.append(0)
            window_matrix[cause] = values
        #window_matrix.to_csv('window_matrix.csv', index=False)
        return window_matrix


    def updatingWindowMatrixTransitions(self, transition,window_matrix):
        initial_state = transition[0]
        final_state = transition[1]
        activities = window_matrix.columns
        components = utils.extractComponents(activities)
        groups=self.df.groupby(self.caseIDs_col_name)
        for c in components:
            initial_activity = c + "_" + initial_state
            final_activity = c + "_" + final_state
            if(initial_activity in activities and final_activity  in activities):
                new_col=list()
                for group, row in groups:
                    considered_activities=row[self.activities_col_name].values
                    if ((initial_activity in considered_activities) & (final_activity in considered_activities)):
                        group_initial_activities=row[row[self.activities_col_name]==initial_activity]
                        initial_timestamp=group_initial_activities[self.timestamp_col_name].min()
                        group_final_activities = row[row[self.activities_col_name] == final_activity]
                        final_timestamp = group_final_activities[self.timestamp_col_name].max()
                        if(initial_timestamp<final_timestamp):
                            new_col.append(1)
                        else:
                            new_col.append(0)

                    else:
                        new_col.append(0)
                col_name=utils.defineTransitionActivities(c,transition)
                window_matrix[col_name]=new_col

        #window_matrix.to_csv("window_matrix.csv", index=False)
        return window_matrix