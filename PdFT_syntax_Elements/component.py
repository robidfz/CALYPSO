from PdFT_syntax_Elements.transition import Transition
from PdFT_syntax_Elements.dynamic import Dynamic
class Component:
    def __init__(self,name, code):
        self.name=name
        self.code=code
        self.input_ports=list()
        self.output_port='p_{'+str(self.extractPedix())+'}'
        self.states=['is_up','is_down','is_failing']
        self.transitions=self.defineTransitions()
        self.dynamics=list()



    def extractPedix(self):
        component_pedix_index = self.code.rfind("_")
        component_pedix = self.code[component_pedix_index + 1:]
        return component_pedix

    def defineTransitions(self):
        transitions=list()
        for i,s1 in enumerate(self.states):
            for j in range(i+1,len(self.states)):
                s2=self.states[j]
                transitions.append(Transition(s1,s2))
                transitions.append(Transition(s2, s1))
        return transitions

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
    def setDynamic(self,name,threshold):
        dynamic= Dynamic(name,threshold)
        self.dynamics.append(dynamic)

    def getDynamic(self,name):
        retval=None
        for d in self.dynamics:
            if(d.name==name):
                retval=d
        return retval

    '''
    def getOutputPort(self):
        return self.output_port
    def getName(self):
        return self.name

    def getCode(self):
        return self.code

'''

