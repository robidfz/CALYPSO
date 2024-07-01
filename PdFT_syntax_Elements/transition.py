class Transition:
    def __init__(self, initial_state, final_state):
        self.structure=(initial_state,final_state)
        self.predicate=None
        self.port=None
        self.probability=None

    def setPredicate(self,predicate):
        self.predicate=predicate
    def setPort(self,port):
        self.port=port

    def setProbability(self,probability):
        self.probability=probability
    def triggerFunction(self):
        return self.predicate

    def alphaFunction(self):
        return self.port

    def rhoFunction(self):
        return self.probability