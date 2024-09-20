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


def preprocessing(reader,df, template):
    property = reader[template]['property'].split(',')
    features = eval(reader[template]['feature'])
    value = eval(reader[template]['value'])
    output = eval(reader[template]['output'])
    filename = eval(reader[template]['filename'])[0]

    for i, p in enumerate(property):
        actual_features = features[i]
        actual_values = value[i]
        actual_output = output[i]

        if (p == 'filtering1'):
            # filter excluding debug records from simulator dataset that are stored in Warning level
            inference_df= utils.parsing_strings(df, actual_features[0], actual_values[0])
        if(p=='filtering2'):
            groups=df.groupby(caseIDs_col_name)
            col_name = actual_features[0]
            col_name2 = actual_features[1]
            for index, group in groups:
                for i, values in enumerate(actual_values):
                    col_value = values[0]
                    col_value2 = values[1]
                    df_1=utils.parsing_strings(group, col_name, col_value)
                    df_2=utils.parsing_strings(df_1, col_name2, col_value2)
                    df_2.sort_values([timestamps_col_name],inplace=True, ascending=True)
                    if(df_2.shape[0]>0):
                        adding_row=df_2.iloc[0]
                        col_to_rename=actual_output[i][0]
                        new_name=actual_output[i][1]
                        adding_row[col_to_rename] = new_name
                        inference_df.loc[len(inference_df)]=adding_row
                        #inference_df=pd.concat([inference_df,adding_row], axis=0)




    inference_df.sort_values([timestamps_col_name],inplace=True)
    inference_df=utils.addingUpState(inference_df,activities_col_name,caseIDs_col_name,timestamps_col_name)
    inference_df.to_csv(filename, index=False)

    return inference_df






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
    col = ['Effect', 'Effect Component', 'Cause', 'Cause Component', 'Effect Input Port', 'Effect Output Port',
           'Cause Output Port', 'Event', 'Epsilon']
    col_pred = ['Effect', 'Component Effect', 'Cause', 'Trigger', 'Alpha', 'Epsilon']
    predicate_df = pd.DataFrame(columns=col_pred)
    structure_df = pd.DataFrame(columns=col)

    for t in effect_names:
        cause_effect_dict=discoveringStructure(dataset,structure,t,pf)
        #cause_effect_dict={'X_top_is_up\\is_down':[['X_C1s_is_up\\is_down',0.5],['X_C2s_is_up\\is_down',0.5],['X_C3s_is_up\\is_down',0.5]],'X_C1s_is_up\\is_down':[['X_C11_is_up\\is_down',1],['X_C10_is_up\\is_down',1]],'X_C2s_is_up\\is_down':[['X_C21_is_up\\is_down',1],['X_C20_is_up\\is_down',1]],'X_C3s_is_up\\is_down':[['X_C31_is_up\\is_down',1],['X_C30_is_up\\is_down',1]],'X_C11_is_up\\is_down':[['sigA>70',0.7]],'X_C10_is_up\\is_down':[['sigA>70',0.7]],'X_C21_is_up\\is_down':[['sigB>3.99999',0.7]],'X_C20_is_up\\is_down':[['sigB>3.99999',0.7]],'X_C30_is_up\\is_down':[['sigC>85',0.7]],'X_C31_is_up\\is_down':[['sigC>85',0.7]]}
        structure,structure_df=structureVisualisation(structure,structure_df,cause_effect_dict)
        predicates_dict=discoveringPredicates(dataset,pf,cause_effect_dict)
        structure,predicate_df=buildingSemantics(predicates_dict,structure,predicate_df)



    test_name=utils.numberTest(filename)
    end_time = time.time()
    end_memory = memory_usage()[0]
    execution_time=end_time-start_time
    mem_usage=end_memory-start_memory
    structure_df.to_csv('PdFT_structure_TTV_'+str(test_name)+'.csv', index=False)
    predicate_df.to_csv('PdFT_predicate_TTV_'+str(test_name)+'.csv',index=False)
    return execution_time,mem_usage

def discoveringStructure(dataset,structure,t,pf):
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

        cause_effect_dict[f] = eps_list
    #cause_effect_dict=utils.filteringSignificantCauses(cause_effect_dict)


    return cause_effect_dict








def structureVisualisation(structure,structure_df,cause_effect_dict):

    G = nx.DiGraph()
    col=structure_df.columns
    significant_causes_dict=dict()

    counter=0
    for effect, values in cause_effect_dict.items():
        significant_causes=list()
        effect_comp, occ_name = utils.splitActivity(effect)
        effect_obj=structure.findComponent(effect_comp)

        causes = [c[0] for c in values]
        eps = [c[1] for c in values]
        cons = effect
        connected_components=structure.findConnectedComponents(effect_comp)
        connected_objects=[c.name for c in connected_components if len(connected_components)>0]
        connected_dynamics=effect_obj.dynamics_set
        connected_objects=connected_objects+[d.name for d in connected_dynamics if len(connected_dynamics)>0]
        for k,cause in enumerate(causes):
            operator=utils.findOperator(cause)
            cause_comp, occ_name = utils.splitActivity(cause)
            #controllare se la connessione nella struttura non è stata già trovata con la componente o la dinamica


            if(operator=='>'):
                cause_obj=structure.findDynamic(cause_comp)
                if (cause_comp not in connected_objects):
                    effect_obj.addDynamic(cause_obj,occ_name)
                input_port=None
                event_name=None
                cause_output_port=None

            else:
                cause_obj = structure.findComponent(cause_comp)
                input_port='p_{' + str(effect_obj.extractPedix())+','+str(cause_obj.extractPedix())+'}'
                if (cause_comp not in connected_objects):
                    effect_obj.addInputPort(input_port)
                    event=Event('e_' + str(counter),cause_obj.output_port,input_port,eps[k])
                    structure.addEvent(event)
                cause_output_port = cause_obj.output_port
                event_name='e_' + str(counter)


            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]] = cause
            row[col[3]] = cause_obj.code
            row[col[4]] = input_port
            row[col[5]] = effect_obj.output_port
            row[col[6]] = cause_output_port
            row[col[7]] = event_name
            row[col[8]] = eps[k]
            structure_df.loc[len(structure_df)] = row
            ant=cause
            G.add_node(ant,name=cause_obj.code)
            G.add_node(cons,name=effect_obj.code)
            G.add_edge(ant, cons, name='e_' + str(counter) ,weight=round(eps[k],4))
            significant_causes.append([cause_comp,eps[k]])
            counter+=1


        significant_causes_dict[effect_comp]=significant_causes

    return structure,structure_df








def discoveringPredicates(df,pf,cause_effect_dict):
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








def buildingSemantics(cause_effect_dict,structure,predicate_df):

    col=predicate_df.columns
    for effect, values in cause_effect_dict.items():
        effect_comp, effect_state = utils.splitActivity(effect)
        initial_state,final_state=utils.splitTransitionActivities(effect)
        e_comp,initial_state=utils.splitActivity(initial_state)
        e_comp,final_state=utils.splitActivity(final_state)
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
                    input_port=effect_obj.getInputPort(cause_obj)
                    predicate=predicate+input_port+"_"+str(operator)+"_"
                    predicate=predicate[:-1]
                    index=predicate.rfind("_")
                    predicate=predicate[:index]


            t=effect_obj.findTransition(initial_state,final_state)
            t.setPredicate(predicate)
            t.setPort(effect_obj.output_port)
            t.setProbability(eps[k])
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]]=cause
            row[col[3]]=t.triggerFunction()
            row[col[4]]=t.alphaFunction()
            row[col[5]]=t.rhoFunction()
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

        #dataset_name='causality_dataset.csv'
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
            template='PREPROCESSING_SETTING'
            #E=preprocessing(reader,E,template)
            structure = structureDefinition(reader, E, activities_col_name)
            exe_time,mem_usage=overallWorkflow(E,structure,predicates_effects_names,filename)
            test_name='TEST'+str(utils.numberTest(filename))
            row['TEST']=test_name
            row['TIME']=exe_time
            row['MEMORY']=mem_usage
            results.loc[len(results)]=row
        results.to_csv('performances.csv',index=False)








