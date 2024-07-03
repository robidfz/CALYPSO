import pandas as pd
import Methodology.utils as utils
class WindowMatrix:
    def __init__(self,df,caseIDs_col_name,activities_col_name,timestamps_col_name):
        self.df = df
        self.caseIDs_col_name = caseIDs_col_name
        self.activities_col_name = activities_col_name
        self.timestamp_col_name = timestamps_col_name
        self.window_matrix=None

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
        self.window_matrix=window_matrix
        return window_matrix

    def updatingWindowMatrixPredicates(self, logic_operator, operand1, operand2):
        new_cause = utils.definePredicate(operand1,operand2,logic_operator)
        if (new_cause not in self.window_matrix.columns):
            values = list()
            if (logic_operator == 'AND'):
                for i, row in self.window_matrix.iterrows():
                    if (row[operand1] == 1 and row[operand2] == 1):
                        values.append(1)
                    else:
                        values.append(0)
            if (logic_operator == 'OR'):
                for i, row in self.window_matrix.iterrows():
                    if (row[operand1] == 0 and row[operand2] == 0):
                        values.append(0)
                    else:
                        values.append(1)
            self.window_matrix[new_cause] = values
        self.window_matrix.to_csv('window_matrix.csv', index=False)
        return self.window_matrix


    def updatingWindowMatrixTransitions(self, transition):
        initial_state = transition[0]
        final_state = transition[1]
        activities = self.window_matrix.columns
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
                self.window_matrix[col_name]=new_col

        self.window_matrix.to_csv("window_matrix.csv", index=False)
        return self.window_matrix