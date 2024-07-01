class Event:
    def __init__(self,name,o_port,i_port,eps):
        self.name = name
        self.structure = (o_port,i_port)
        self.weight=eps
    def weightFunction(self):
        return self.weight