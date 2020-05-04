import neuralNetworkModel as NN
import datasetPreparation

#initialize neural NeuralNetwork
learningRate = 0.002
biasNeuronsValue = 0
architecture = [4, 100, 49, 3]
n = NN.NeuralNetwork(learningRate, biasNeuronsValue, architecture)

#prepare training dataset
data = datasetPreparation.Dataset('train')
inputs = data.getInputs()
targets = data.getTargets()

#prepare testing dataset
testData = datasetPreparation.Dataset('test')
testInputs = testData.getInputs()
testTargets = testData.getTargets()

for epoch in range(600):
    for i in range(len(inputs)):
        n.train(inputs[i], targets[i])


#count accuarcy
score = 0
for record in range(len(inputs)):
    #check if the training record is equal testing record
    if targets[record] == [round(float(n.query(testInputs[record])[i])) for i in range(3)]:
        score += 1
    else:
        print(f"{record} record was classified incorrectly!")

print(f"Correct answers: {score}/{len(inputs)}")
