from PdFT_syntax_Elements.transition import Transition
from PdFT_syntax_Elements.dynamic import Dynamic

class Component:
    def __init__(self,name, code):
        self.name=name
        self.code=code
        self.input_ports=list()
        self.output_port='p_{'+str(self.extractPedix())+'}'
        self.states=list()
        self.statePriorities = {'is_up': 3, 'is_failing': 2, 'is_down': 1}
        self.transitions=list()
        self.transitionPriorities=dict()
        self.dynamics_set=dict()





    def extractPedix(self):
        component_pedix_index = self.code.rfind("_")
        component_pedix = self.code[component_pedix_index + 1:]
        return component_pedix

    def defineTransitions(self):
        for i,s1 in enumerate(self.states):
            for j in range(i+1,len(self.states)):
                s2=self.states[j]
                self.transitions.append(Transition(s1,s2))
                self.transitions.append(Transition(s2, s1))
        self.defineTransitionPriorities()

    def findTransition(self,s1,s2):
        struct=(s1,s2)
        retval=None
        for t in self.transitions:
            if(t.structure==struct):
                retval=t
        return retval
    def addInputPort(self,input_port):
        self.input_ports.append(input_port)
    def getInputPort(self,input_component):
        j=input_component.extractPedix()
        for p in self.input_ports:
            index=p.find(',')
            pedix=p[index+1:-1]
            if(pedix==j):
                port=p
        return port



    def setStates(self,states):
        self.states=states
        self.defineTransitions()

    def setStatePriority(self,state,priority):
        self.statePriorities[state]=priority


    def getTransitionPriority(self,t):
        initial_state_priority=self.statePriorities[t[0]]
        final_state_priority=self.statePriorities[t[1]]
        transition_priority= initial_state_priority-final_state_priority
        return transition_priority

    def defineTransitionPriorities(self):
        for t in self.transitions:
            priority=self.getTransitionPriority(t.structure)
            self.transitionPriorities[t]=priority


    def addDynamic(self, dynamic_obj,threshold):
        self.dynamics_set[dynamic_obj]=threshold

    def getDynamicThreshold(self, dyn_name):
        retval = None
        for dynamic_obj, threshold in self.dynamics_set.items():
            if (dynamic_obj.name == dyn_name):
                retval = threshold
        return retval





    '''
    def getOutputPort(self):
        return self.output_port
    def getName(self):
        return self.name

    def getCode(self):
        return self.code

'''

