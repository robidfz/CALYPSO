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
from Methodology.windowmatrix import WindowMatrix
import Methodology.epsilons as epsilon
import itertools

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
    for d in dynamic:
        dyn_obj=Dynamic(d)
        structure.addDynamic(dyn_obj)

    return structure




def discoveringStructure(dataset,structure,effect_names):


    #wm=WindowMatrix(dataset,caseIDs_col_name,activities_col_name,timestamps_col_name)
    #window_matrix = wm.windowMatrixGeneration()
    pf=PrimaFacie(dataset,caseIDs_col_name,activities_col_name,timestamps_col_name,window_matrix=None)
    components=structure.getComponentsNames()
    cause_effect_dict=dict()
    for e in effect_names:
        A=[utils.defineActivity(a,e) for a in components]
        for f in A:
            prima_facie = list()
            for a in A:
                if(f!=a):
                    if (pf.primafacie(f, a,10)):
                        prima_facie.append(a)
            #eps_avg=epsilon.epsilon_averages(prima_facie,window_matrix,f)
            eps_avg=epsilon.epsilon_averages(prima_facie,dataset,f,activities_col_name,caseIDs_col_name,timestamps_col_name,10)
            cause_effect_dict[f] = eps_avg
    cause_effect_dict=utils.filteringSignificantCauses(cause_effect_dict)
    structure=structureVisualisation(structure,cause_effect_dict)
    return None,structure












def structureVisualisation(structure,cause_effect_dict):

    G = nx.DiGraph()
    col=['Effect','Effect Component','Cause','Cause Component','Effect Input Port','Effect Output Port','Cause Output Port','Event','Weight value']
    structure_df=pd.DataFrame(columns=col)
    significant_causes_dict=dict()

    counter=0
    for effect, values in cause_effect_dict.items():
        significant_causes=list()
        effect_comp, occ_name = utils.splitActivity(effect)
        effect_obj=structure.findComponent(effect_comp)

        causes = [c[0] for c in values]
        eps = [c[1] for c in values]
        cons = effect

        for k,cause in enumerate(causes):
            cause_comp, occ_name = utils.splitActivity(cause)
            cause_obj=structure.findComponent(cause_comp)



            input_port='p_{' + str(effect_obj.extractPedix())+','+str(cause_obj.extractPedix())+'}'
            effect_obj.addInputPort(input_port)
            event=Event('e_' + str(counter),cause_obj.output_port,input_port,eps[k])
            structure.addEvent(event)
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]] = cause
            row[col[3]] = cause_obj.code
            row[col[4]] = event.structure[1]
            row[col[5]] = effect_obj.output_port
            row[col[6]] = event.structure[0]
            row[col[7]] = event.name
            row[col[8]] = event.weight
            structure_df.loc[len(structure_df)] = row
            ant=cause
            G.add_node(ant,name=cause_obj.code)
            G.add_node(cons,name=effect_obj.code)
            G.add_edge(ant, cons, name='e_' + str(counter) ,weight=round(eps[k],4))
            significant_causes.append([cause_comp,eps[k]])
            counter+=1


        significant_causes_dict[effect_comp]=significant_causes
    structure_df.to_csv('PdFT_structure.csv',index=False)
    graphVisualization(G,structure,'PdFT_structure.pdf')
    return structure





def graphVisualization(G,structure,name_fig):

    fig, ax = plt.subplots(figsize=(40, 40))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos,node_shape='s',node_color='lightblue',node_size=30000)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    node_labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=50, font_color='black')
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12,font_color='black')

    for (u, v, d) in G.edges(data=True):
        effect_name,act=utils.splitActivity(v)
        cause_name,act_cause=utils.splitActivity(u)
        effect_comp = structure.findComponent(effect_name)
        cause_comp=structure.findComponent(cause_name)
        p_name_in = effect_comp.getInputPort(cause_comp)
        p_name_out = cause_comp.output_port

        x_start, y_start = pos[u]
        x_end, y_end = pos[v]

        # Calcolare la direzione dell'arco
        dx = x_end - x_start
        dy = y_end - y_start
        angle = np.arctan2(dy, dx)

        # Offset per posizionare il quadrato sul bordo del nodo
        offset_x = 0.600167 * np.cos(angle)
        offset_y = 0.600167 * np.sin(angle)

        # Posizionare il quadrato sul bordo del nodo di partenza
        box_start_x = x_start + offset_x
        box_start_y = y_start + offset_y
        plt.plot([box_start_x], [box_start_y], marker='s', markersize=60, color='grey')
        plt.text(box_start_x, box_start_y, p_name_out, fontsize=20, ha='center', va='center', color='black')

        # Posizionare il quadrato sul bordo del nodo di arrivo
        box_end_x = x_end - offset_x
        box_end_y = y_end - offset_y
        plt.plot([box_end_x], [box_end_y], marker='s', markersize=60, color='grey')
        plt.text(box_end_x, box_end_y, p_name_in, fontsize=20, ha='center', va='center', color='black')

        # Disegnare l'arco dai quadrati
        #plt.arrow(box_start_x, box_start_y, box_end_x - box_start_x, box_end_y - box_start_y,length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
    #labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(name_fig)
    plt.close()



def discoveringPredicates(df,window_matrix,structure):
    eps_avg_dict=dict()
    components=structure.getComponentsNames()
    operands=['AND','OR']
    local_eps=dict()
    #defining all the possible transitions observed in the dataset
    possible_transitions_dict=utils.extractPossibleTransition(df,activities_col_name,caseIDs_col_name,components)
    pf = PrimaFacie(df, caseIDs_col_name, activities_col_name, timestamps_col_name, window_matrix)
    #selecting a component
    for effect_component in components:
        effect_transitions=possible_transitions_dict[effect_component]
        effect_obj=structure.findComponent(effect_component)
        #selecting the connected components in the structure
        connected_components_obj = structure.findConnectedComponents(effect_obj.name)
        if(len(connected_components_obj)>0):
            connected_components = sorted([c.name for c in connected_components_obj])
            possible_single_causes=list()
            for c in connected_components:
                #selecting all the possible transitions associated to the connected components
                possible_single_causes=possible_single_causes+possible_transitions_dict[c]

            #Selecting an observed effect: one of the possible transition on the selected component
            for f in effect_transitions:
                A = list()
                #find what transition among all the possible transition observed in the connected components is a prima facie for f
                for a in possible_single_causes:
                    if (pf.primafacie(f, a, 10)):
                        A.append(a)
                #compute all the epsilon and select the most significant causes
                local_eps_list = epsilon.epsilon_averages(A, df, f, activities_col_name, caseIDs_col_name,timestamps_col_name, 10)
                local_eps[f] = local_eps_list
                cause_effect_dict = utils.filteringSignificantCauses(local_eps, False)
                selected_components=[x[0] for x in cause_effect_dict[f]]
                selected_transitions=dict()
                for c in connected_components:
                    selected_transitions[c]=[x for x in selected_components if x.startswith(c)]
                A = list()
                #composing the most significant causes with the operands
                combinations = list(itertools.product(*[selected_transitions[component] for component in connected_components]))
                total_composed_causes=list()
                for operand in operands:
                    composed_causes=[f"_{operand}_".join(combination) for combination in combinations]
                    total_composed_causes=total_composed_causes+composed_causes
                # find what composed cause among all the possible composed causes is a prima facie for f
                for a in total_composed_causes:
                    if (pf.primafacie(f, a,10)):
                            A.append(a)
                eps_avg = epsilon.epsilon_averages(A,df,f,activities_col_name,caseIDs_col_name,timestamps_col_name,10)
                eps_avg_dict[f] = eps_avg


    cause_effect_dict=utils.filteringSignificantCauses(eps_avg_dict,False)
    structure=buildingSemantics(cause_effect_dict,structure)
    return window_matrix,structure





def buildingSemantics(cause_effect_dict,structure):

    col=['Effect','Component Effect','Cause','Trigger','Alpha','Rho']
    predicate_df=pd.DataFrame(columns=col)
    for effect, values in cause_effect_dict.items():
        effect_comp, effect_state = utils.splitActivity(effect)
        initial_state,final_state=utils.splitTransitionActivities(effect)
        e_comp,initial_state=utils.splitActivity(initial_state)
        e_comp,final_state=utils.splitActivity(final_state)
        effect_obj = structure.findComponent(effect_comp)
        causes = [c[0] for c in values]
        eps=[c[1] for c in values]
        for k, cause in enumerate(causes):
            operator=utils.findOperator(cause)
            elems=utils.splitPredicate(cause)
            predicate=''
            for e in elems:
                cause_comp, occ_name = utils.splitActivity(e)
                cause_obj = structure.findComponent(cause_comp)
                input_port=effect_obj.getInputPort(cause_obj)
                predicate=predicate+input_port+"_"+operator+"_"
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
    predicate_df.to_csv('predicate_PdFT.csv',index=False)
    return structure

def discoveringThresholds(df,window_matrix,structure,dynamic_effect_names):
    pf=PrimaFacie(df, caseIDs_col_name,activities_col_name,timestamps_col_name,window_matrix)
    #possible_activities=window_matrix.columns
    possible_activities=df[activities_col_name].unique()
    eps_avg_dict=dict()
    # selecting the components on which we have dynamics installed
    selected_components=structure.components
    selected_components=[c.name for c in selected_components]
    dynamics_names=structure.dynamics
    dynamics_names = [d.name for d in dynamics_names]
    for t in dynamic_effect_names:
        #defining the transition effect on the selected components
        effects=[utils.defineTransitionActivities(a,t) for a in selected_components]
        thresholds_predicates=list()
        for effect in effects:
            for d in dynamics_names:
                thresholds_predicates =thresholds_predicates+ [a for a in possible_activities if a.startswith(d)]
            A=list()
            for a in thresholds_predicates:
                if(pf.primafacie(effect, a,10)):
                    A.append(a)
            eps_avg = epsilon.epsilon_averages(A,df,effect,activities_col_name,caseIDs_col_name,timestamps_col_name,10)

            eps_avg_dict[effect]=eps_avg
    #window_matrix.to_csv("window_matrix.csv", index=False)
    cause_effect_dict=utils.filteringSignificantCauses(eps_avg_dict,False)
    structure=buildingDynamics(cause_effect_dict,structure)
    return structure




def buildingDynamics(cause_effect_dict,structure):

    col=['Effect','Component Effect','Cause','Dynamic','Threshold','Trigger','Alpha','Rho']
    predicate_df=pd.DataFrame(columns=col)
    for effect, values in cause_effect_dict.items():
        effect_comp, effect_state = utils.splitActivity(effect)
        initial_state, final_state = utils.splitTransitionActivities(effect)
        name,initial_state=utils.splitActivity(initial_state)
        name,final_state=utils.splitActivity(final_state)
        effect_obj = structure.findComponent(effect_comp)
        causes = [c[0] for c in values]
        eps=[c[1] for c in values]
        for k, cause in enumerate(causes):
            dynamic_name,threshold=utils.splitDynamicCauses(cause)
            dyn=structure.findDynamic(dynamic_name)
            dyn.threshold=threshold
            predicate=cause
            t=effect_obj.findTransition(initial_state,final_state)
            t.setPredicate(predicate)
            t.setPort(effect_obj.output_port)
            t.setProbability(eps[k])
            row=dict()
            row[col[0]]=effect
            row[col[1]]=effect_obj.code
            row[col[2]]=cause
            row[col[3]]=dyn.name
            row[col[4]]=dyn.threshold
            row[col[5]]=t.triggerFunction()
            row[col[6]]=t.alphaFunction()
            row[col[7]]=t.rhoFunction()
            predicate_df.loc[len(predicate_df)] = row
    predicate_df.to_csv('thresholds_PdFT.csv',index=False)
    return structure

if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        reader = ConfigParser()
        reader.read(configfile_name)
        E = pd.read_csv(dataset_name)
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        structure_effects_names = eval(reader['INFERENCE']['structure_effects'])
        caseIDs_col_name = eval(reader['DATASET']['caseID_col'])[0]
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        predicates_effects_names = eval(reader['INFERENCE']['predicates_effects'])
        dynamics_effect_names=eval(reader['INFERENCE']['dynamics_effects'])
        #E = E[(~E[activities_col_name].str.endswith('under_repair'))]
        template='PREPROCESSING_SETTING'
        E=preprocessing(reader,E,template)
        structure = structureDefinition(reader, E, activities_col_name)
        wm,structure=discoveringStructure(E,structure,structure_effects_names)
        wm,structure=discoveringPredicates(E,wm,structure)
        discoveringThresholds(E,wm,structure,dynamics_effect_names)








