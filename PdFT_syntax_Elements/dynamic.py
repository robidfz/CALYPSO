class Dynamic:
    def __init__(self,name):
        self.name = name
        self.threshold = None


    def setThreshold(self,threshold):
        self.threshold=threshold

    def getName(self):
        return self.name