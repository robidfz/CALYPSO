import numpy as np
import pandas as pd
from itertools import product
from configparser import ConfigParser
import sys
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from PdFT_syntax_Elements.component import Component
from PdFT_syntax_Elements.structure import PdFT
from PdFT_syntax_Elements.event import Event
from PdFT_syntax_Elements.dynamic import Dynamic
from Causality_analysis_modules.primafacie import PrimaFacie
import Causality_analysis_modules.utils as utils
import Causality_analysis_modules.epsilons as epsilon



class Inference():
    def __init__(self,activities_col_name,caseIDs_col_name,timestamps_col_name,transition_names=None,component_set=None,dynamic=None):
        self.activities_col_name = activities_col_name
        self.caseIDs_col_name = caseIDs_col_name
        self.timestamps_col_name =timestamps_col_name
        self.transition_names = transition_names
        self.component_set=component_set
        self.dynamic=dynamic

    def modelTemplate(self,df,activity_col_name):
        structure=PdFT()

        activities=df[activity_col_name].unique()
        if self.component_set is None:
            df_component = df['Element_ID'].astype(str).unique()
            self.component_set = [element for element in df_component if element.startswith('X')]

        for i, c in enumerate(self.component_set):
            states=list()
            comp_obj = Component(c, "c_" + str(i))
            states_activities=[s for s in activities if s.startswith(c)]
            for sa in states_activities:
                comp_name,state=utils.splitActivity(sa)
                states.append(state)
            comp_obj.setStates(states)
            structure.addComponent(comp_obj)
        if self.dynamic is None:
            df_dynamic = df['Element_ID'].astype(str).unique()
            self.dynamic = [element for element in df_dynamic if element.startswith('sig')]
        for j,d in enumerate(self.dynamic):
            dyn_obj=Dynamic(d,"d_"+str(j))
            structure.addDynamic(dyn_obj)

        if self.transition_names is None:
            transitions = df['Message_description'].astype(str).unique()
            self.transition_names = [element for element in transitions if element.startswith('is')]

        return structure




    def structureInference(self,dataset,structure,t,pf):
        cause_effect_dict=dict()
        components = structure.getComponentsNames()
        dynamics=structure.getDynamicsNames()
        transition_dict=utils.extractPossibleTransition(dataset,self.activities_col_name,self.caseIDs_col_name,components)
        transitions=[value for values in transition_dict.values() for value in values]
        signals_events=utils.extractPossibleSignalsEvents(dataset,dynamics,self.activities_col_name)
        effects=list()
        for c in components:
            effects.append(utils.defineTransitionActivities(c,t))
        A = transitions+signals_events
        for f in effects:
            prima_facie = list()
            effect_component,actv=utils.splitActivity(f)
            for a in A:
                if(a.startswith(effect_component)==False):
                    if (pf.primafacie(f, a,10)):
                        prima_facie.append(a)
            eps_list = epsilon.epsilon_averages(prima_facie, dataset, f, self.activities_col_name, self.caseIDs_col_name,self.timestamps_col_name, 10)
            if eps_list:
                cause_effect_dict[f] = eps_list



        return cause_effect_dict

    def rulesInference(self,df,pf,cause_effect_dict):
        operators=['AND','OR']
        eps_avg_dict=dict()

        for f,causes in cause_effect_dict.items():
            composed_causes=list()
            causes_name=[c[0] for c in causes]
            for operator in operators:
                if(len(causes_name)>0):
                    new_predicate=causes_name[0]
                    for i in range(1,len(causes_name)):
                        new_predicate=new_predicate+f"_{operator}_"+causes_name[i]
                    composed_causes.append(new_predicate)
            composed_causes=list(set(composed_causes))
            A=list()
            for a in composed_causes:
                if(pf.primafacie(f, a,10)):
                    A.append(a)
            eps_avg = epsilon.epsilon_averages(A,df,f,self.activities_col_name,self.caseIDs_col_name,self.timestamps_col_name,10)
            eps_avg_dict[f] = eps_avg
        eps_avg_dict=utils.filteringSignificantCauses(eps_avg_dict,True)

        return eps_avg_dict






    def structure2PdFT(self,structure,structure_df,structure_dict):
        active_transitions = [activity[0] for actvity_list in structure_dict.values() for activity in actvity_list]
        col=structure_df.columns
        significant_causes_dict=dict()

        counter=0
        for effect, values in structure_dict.items():

            significant_causes=list()
            effect_comp, occ_name = utils.splitActivity(effect)
            transition_effect=utils.extractTransition(effect)
            effect_obj=structure.findComponent(effect_comp)
            active_flag=(effect in active_transitions)
            effect_obj.setAlpha(transition_effect,active_flag)



            causes = [c[0] for c in values]
            eps = [c[1] for c in values]
            connected_components=structure.findConnectedComponents(effect_comp)
            connected_objects=[c.name for c in connected_components if len(connected_components)>0]
            connected_dynamics=effect_obj.dynamics_set
            connected_objects=connected_objects+[d.name for d in connected_dynamics if len(connected_dynamics)>0]
            for k,cause in enumerate(causes):
                operator=utils.findOperator(cause)
                cause_comp, occ_name = utils.splitActivity(cause)



                if(operator=='>'):
                    cause_obj=structure.findDynamic(cause_comp)
                    if (cause_comp not in connected_objects):
                        effect_obj.addDynamic(cause_obj,occ_name)
                    input_port=None
                    event_name=None
                    cause_output_port_value=None

                else:

                    cause_obj = structure.findComponent(cause_comp)
                    transition_cause = utils.extractTransition(cause)
                    active_flag = (cause in active_transitions)
                    cause_obj.setAlpha(transition_cause, active_flag)
                    input_port='p_{' + str(effect_obj.extractPedix())+','+str(cause_obj.extractPedix())+'}'
                    if (cause_comp not in connected_objects):
                        effect_obj.addInputPort(input_port)
                        event=Event('e_' + str(counter),cause_obj.output_port,input_port,eps[k])
                        structure.addEvent(event)
                    cause_output_port_value = cause_obj.alpha[transition_cause]
                    event_name='e_' + str(counter)


                row=dict()
                row[col[0]]=effect
                row[col[1]]=effect_obj.code
                row[col[2]] = cause
                row[col[3]] = cause_obj.code
                row[col[4]] = input_port
                row[col[5]] = effect_obj.alpha[transition_effect]
                row[col[6]] = cause_output_port_value
                row[col[7]] = event_name
                row[col[8]] = eps[k]
                structure_df.loc[len(structure_df)] = row
                significant_causes.append([cause_comp,eps[k]])
                counter+=1


            significant_causes_dict[effect_comp]=significant_causes

        return structure,structure_df




    def rules2PdFT(self,structure, predicate_df,rules_results):



        col=predicate_df.columns
        for effect, values in rules_results.items():
            effect_comp, effect_state = utils.splitActivity(effect)
            transition_effect=utils.extractTransition(effect)
            effect_obj = structure.findComponent(effect_comp)
            causes = [c[0] for c in values]
            eps=[c[1] for c in values]
            for k, cause in enumerate(causes):
                elems = utils.splitPredicate(cause)
                predicate = ''
                for e in elems:
                    operator=utils.findOperator(e)
                    if(operator=='>'):
                        predicate=predicate+e
                    else:
                        cause_comp, occ_name = utils.splitActivity(e)
                        cause_obj = structure.findComponent(cause_comp)
                        transition_cause = utils.extractTransition(e)
                        input_port = effect_obj.getInputPort(cause_obj)


                        if(cause_obj.alpha[transition_cause]==0):
                            input_port='NOT_'+input_port

                        predicate=predicate+input_port+"_"+str(operator)+"_"
                        predicate=predicate[:-1]
                        index=predicate.rfind("_")
                        predicate=predicate[:index]


                effect_obj.setTrigger(transition_effect,predicate,eps[k])
                row=dict()
                row[col[0]]=effect
                row[col[1]]=effect_obj.code
                row[col[2]]=cause
                row[col[3]]=effect_obj.trigger[transition_effect]
                row[col[4]]=effect_obj.alpha[transition_effect]
                row[col[5]]=eps[k]
                predicate_df.loc[len(predicate_df)] = row

        return structure,predicate_df





    def inference(self, E,output_name):

        #E = E.head(100)
        structure = self.modelTemplate( E, self.activities_col_name)
        pf = PrimaFacie(E, self.caseIDs_col_name, self.activities_col_name, self.timestamps_col_name)
        col = ['Effect', 'Effect Component', 'Cause', 'Cause Component', 'Effect Input Port',
               'Effect Output Port Value',
               'Cause Output Port Value', 'Event', 'Epsilon']
        col_pred = ['Effect', 'Component Effect', 'Cause', 'Trigger', 'Alpha', 'Epsilon']
        rules_df = pd.DataFrame(columns=col_pred)
        structure_df = pd.DataFrame(columns=col)
        structure_results = dict()
        rules_results = dict()

        for t in self.transition_names:
            cause_effect_dict = self.structureInference(E, structure, t, pf)
            structure_results = structure_results | cause_effect_dict
            rules_dict = self.rulesInference(E, pf, cause_effect_dict)
            rules_results = rules_results | rules_dict

        structure, structure_df = self.structure2PdFT(structure, structure_df, structure_results)
        structure, predicate_df = self.rules2PdFT(structure, rules_df, rules_results)
        structure_df.to_csv(output_name+'_PdFT_structure.csv', index=False)
        predicate_df.to_csv(output_name+'_PdFT_predicate.csv', index=False)












