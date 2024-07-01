class PdFT:
    def __init__(self):
        self.components=list()
        self.events=list()


    def addComponent(self, component):
        self.components.append(component)

    def addEvent(self,event):
        self.events.append(event)

    def findComponent(self,comp_name):
        retval=None
        for component in self.components:
            if component.name == comp_name:
                retval= component
        return retval