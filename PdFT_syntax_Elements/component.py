from PdFT_syntax_Elements.transition import Transition
from PdFT_syntax_Elements.dynamic import Dynamic

class Component:
    def __init__(self,name, code):
        self.name=name
        self.code=code
        self.input_ports=list()
        self.output_port='p_{'+str(self.extractPedix())+'}'
        self.states=list()
        self.transitions=list()
        self.dynamics_set=dict()
        self.trigger=dict()
        self.alpha=dict()





    def extractPedix(self):
        component_pedix_index = self.code.rfind("_")
        component_pedix = self.code[component_pedix_index + 1:]
        return component_pedix

    def defineTransitions(self):
        for i,s1 in enumerate(self.states):
            for j in range(i+1,len(self.states)):
                s2=self.states[j]
                self.transitions.append((s1,s2))
                self.transitions.append((s2, s1))

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



    def addDynamic(self, dynamic_obj,threshold):
        self.dynamics_set[dynamic_obj]=threshold

    def getDynamicThreshold(self, dyn_name):
        retval = None
        for dynamic_obj, threshold in self.dynamics_set.items():
            if (dynamic_obj.name == dyn_name):
                retval = threshold
        return retval

    def setTrigger(self, transition,predicate):
        if(transition in self.transitions):
            self.trigger[transition]=predicate


    def setAlpha(self, transition, active_flag):
        if active_flag:
            if ((transition in self.transitions) and (transition not in self.alpha.keys())):
                self.alpha[(transition[0],transition[1])]=1
                self.alpha[(transition[1],transition[0])]=0
        else:
            self.alpha[transition] = -1






    '''
    def getOutputPort(self):
        return self.output_port
    def getName(self):
        return self.name

    def getCode(self):
        return self.code

'''

