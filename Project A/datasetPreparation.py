from sklearn import datasets

class Dataset:

    def __init__(self, mode='train'):
        self.iris = datasets.load_iris()
        self.mode = mode
        pass

    def getInputs(self):
        inputs = self.iris.data
        if self.mode == 'train':
            return inputs[::2] #I use every even record from dataset for training
        if self.mode == 'test':
            return inputs[1::2] #every odd record from dataset is used for testing

    def getTargets(self):
        iris_targets = self.iris.target
        targets = []
        for i in range(150): #recreating targets array because default values are from 0 to 2, not 0 to 1
            if iris_targets[i] == 0:
                 targets.append([1, 0, 0])
            if iris_targets[i] == 1:
                 targets.append([0, 1, 0])
            if iris_targets[i] == 2:
                 targets.append([0, 0, 1])
        if self.mode == 'train':
            return targets[::2]
        if self.mode == 'test':
            return targets[1::2]
