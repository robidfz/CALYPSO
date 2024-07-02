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
    def findComponentByOutputPort(self,output_port):
        pedix_index=output_port.find("{")
        pedix=output_port[pedix_index+1:-1]
        component_code="c_"+pedix
        retval=None
        for c in self.components:
            if c.code == component_code:
                retval = c
        return retval


    def componentList(self):
        c_list=list()
        for c in self.components:
            c_list.append(c.name)
        return c_list

    def findConnectedComponents(self,c_name):
        comp_input=self.findComponent(c_name)
        port_input=comp_input.input_ports
        connected_components=list()
        for ip in port_input:
            events_considered = [event for event in self.events if event.structure[1] == ip]
            for e in events_considered:
                pout=e.structure[0]
                if(self.findComponentByOutputPort(pout)!=None):
                    connected_components.append(self.findComponentByOutputPort(pout))
        return connected_components

    def findComponentWithDynamics(self):
        selected_components=list()
        for c in self.components:
            if (len(c.dynamics) != 0):
                selected_components.append(c)
        return selected_components