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
from Methodology.primafacie import PrimaFacie
import Methodology.utils as utils
import Methodology.epsilons as epsilon
import os
import glob
import time
from memory_profiler import memory_usage




def structureDefinition(reader,df,activity_col_name):
    structure=PdFT()
    component_set = eval(reader['STRUCTURE_SETTING']['components_set'])
    dynamic = eval(reader['STRUCTURE_SETTING']['dynamics_set'])
    activities=df[activity_col_name].unique()
    for i, c in enumerate(component_set):
        states=list()
        comp_obj = Component(c, "c_" + str(i))
        states_activities=[s for s in activities if s.startswith(c)]
        for sa in states_activities:
            comp_name,state=utils.splitActivity(sa)
            states.append(state)
        comp_obj.setStates(states)
        structure.addComponent(comp_obj)
    for j,d in enumerate(dynamic):
        dyn_obj=Dynamic(d,"d_"+str(j))
        structure.addDynamic(dyn_obj)

    return structure


def overallWorkflow(dataset,structure,effect_names,filename):
    start_time = time.time()
    start_memory=memory_usage()[0]

    pf = PrimaFacie(dataset, caseIDs_col_name, activities_col_name, timestamps_col_name)
    col = ['Effect', 'Effect Component', 'Cause', 'Cause Component', 'Effect Input Port', 'Effect Output Port Value',
           'Cause Output Port Value', 'Event', 'Epsilon']
    col_pred = ['Effect', 'Component Effect', 'Cause', 'Trigger', 'Alpha', 'Epsilon']
    rules_df = pd.DataFrame(columns=col_pred)
    structure_df = pd.DataFrame(columns=col)
    structure_results=dict()
    rules_results=dict()

    for t in effect_names:
        cause_effect_dict=structureInference(dataset,structure,t,pf)
        structure_results= structure_results | cause_effect_dict
        rules_dict=rulesInference(dataset,pf,cause_effect_dict)
        rules_results= rules_results | rules_dict

    structure, structure_df = structure2PdFT(structure, structure_df, structure_results)
    structure, predicate_df = rules2PdFT(structure, rules_df,rules_results)


    number,test_name=utils.numberTest(filename)
    end_time = time.time()
    end_memory = memory_usage()[0]
    execution_time=end_time-start_time
    mem_usage=end_memory-start_memory
    structure_df.to_csv('PdFT_structure'+str(test_name), index=False)
    predicate_df.to_csv('PdFT_predicate'+str(test_name),index=False)
    return execution_time,mem_usage

def structureInference(dataset,structure,t,pf):
    cause_effect_dict=dict()
    components = structure.getComponentsNames()
    dynamics=structure.getDynamicsNames()
    transition_dict=utils.extractPossibleTransition(dataset,activities_col_name,caseIDs_col_name,components)
    transitions=[value for values in transition_dict.values() for value in values]
    signals_events=utils.extractPossibleSignalsEvents(dataset,dynamics,activities_col_name)
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

        #filtered_causes=utils.filteringLatestCauses(prima_facie,f,dataset,activities_col_name,caseIDs_col_name,timestamps_col_name)
        eps_list = epsilon.epsilon_averages(prima_facie, dataset, f, activities_col_name, caseIDs_col_name,timestamps_col_name, 10)
        if eps_list:
            cause_effect_dict[f] = eps_list
    #cause_effect_dict=utils.filteringSignificantCauses(cause_effect_dict)


    return cause_effect_dict

def rulesInference(df,pf,cause_effect_dict):
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
        eps_avg = epsilon.epsilon_averages(A,df,f,activities_col_name,caseIDs_col_name,timestamps_col_name,10)
        eps_avg_dict[f] = eps_avg
    eps_avg_dict=utils.filteringSignificantCauses(eps_avg_dict,True)

    return eps_avg_dict






def structure2PdFT(structure,structure_df,structure_dict):
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




def rules2PdFT(structure, predicate_df,rules_results):



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


            effect_obj.setTrigger(transition_effect,predicate)
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]]=cause
            row[col[3]]=effect_obj.trigger[transition_effect]
            row[col[4]]=effect_obj.alpha[transition_effect]
            row[col[5]]=eps[k]
            predicate_df.loc[len(predicate_df)] = row

    return structure,predicate_df







if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset = sys.argv[2]
        folder = os.getcwd()
        pattern = os.path.join(folder, dataset+"*")
        file_list = glob.glob(pattern)
        results=pd.DataFrame(columns=['TEST','TIME','MEMORY'])
        #file_list = [file for file in file_list if not (file.endswith('TEST1') or file.endswith('TEST2') or file.endswith('TEST3') or file.endswith('TEST0') or file.endswith('TEST4') or file.endswith('TEST5'))]
        for filename in file_list:
            row = dict()
            reader = ConfigParser()
            reader.read(configfile_name)
            E = pd.read_csv(filename)
            activities_col_name = eval(reader['DATASET']['activities_col'])[0]
            structure_effects_names = eval(reader['INFERENCE']['structure_effects'])
            caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
            timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
            predicates_effects_names = eval(reader['INFERENCE']['predicates_effects'])
            dynamics_effect_names=eval(reader['INFERENCE']['dynamics_effects'])
            #E = E.head(100)
            structure = structureDefinition(reader, E, activities_col_name)
            exe_time,mem_usage=overallWorkflow(E,structure,predicates_effects_names,filename)
            test_name='TEST'+str(utils.numberTest(filename))
            row['TEST']=test_name
            row['TIME']=exe_time
            row['MEMORY']=mem_usage
            results.loc[len(results)]=row
        results.to_csv('performances.csv',index=False)








