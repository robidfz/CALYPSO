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
        new_cause = operand1 + '_' + logic_operator + '_' + operand2
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

    def updatingWindowMatrixTransitions_old(self, transition):
        initial_state = transition[0]
        final_state = transition[1]
        activities = self.window_matrix.columns
        components = utils.extractComponents(activities)
        for i, row in self.window_matrix.iterrows():
            for c in components:
                initial_activity = c + "_" + initial_state
                final_activity = c + "_" + final_state
                transition_activity = utils.defineTransitionActivities(c, transition)
                if ((row[initial_activity] == 1) and (row[final_activity] == 1)):
                    self.window_matrix.at[i, transition_activity] = 1
                else:
                    self.window_matrix.at[i, transition_activity] = 0

        self.window_matrix.to_csv("window_matrix.csv", index=False)
        return self.window_matrix

    def updatingWindowMatrixTransitions(self, transition):
        initial_state = transition[0]
        final_state = transition[1]
        activities = self.window_matrix.columns
        components = utils.extractComponents(activities)
        groups=self.df.groupby(self.caseIDs_col_name)
        for group, row in groups:
            for c in components:
                initial_activity = c + "_" + initial_state
                final_activity = c + "_" + final_state
                transition_activity = utils.defineTransitionActivities(c, transition)
                considered_activities=row[self.activities_col_name].values
                if ((initial_activity in considered_activities) & (final_activity in considered_activities)):
                    initial_timestamp=min(row[row[self.activities_col_name]==initial_activity][self.timestamp_col_name].values)
                    final_timestamp=max(row[row[self.activities_col_name]==final_activity][self.timestamp_col_name].values)
                    if(initial_timestamp<final_timestamp):
                        self.window_matrix.at[group, transition_activity] = 1
                else:
                    self.window_matrix.at[group, transition_activity] = 0

        self.window_matrix.to_csv("window_matrix.csv", index=False)
        return self.window_matrix