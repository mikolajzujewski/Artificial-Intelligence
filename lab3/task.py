import numpy as np
import random


def ReLuFunc(x):
    if x > 0:
        return x
    else:
        return 0

def ReLuFuncPrime(x):
    if x > 0:
        return 1
    else:
        return 0

def heavisideFunc(x):
    if x < 0:
        return 0
    else:
        return 1

def heavisideFuncPrime(x):
    return 1

def sigmoidFunc(x):
    return 1 / (1 + np.exp(-x))

def sigmoidFuncPrime(x):
    return sigmoidFunc(x) * (1 - sigmoidFunc(x))

def sinFunc(x):
    return np.sin(x)

def sinFuncPrime(x):
    return np.cos(x)

def tanhFunc(x):
    return np.tanh(x)

def tanhFuncPrime(x):
    return 1 - tanhFunc(x)**2

def signFunc(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1

def signFuncPrime(x):
    return 1

def leakyReLuFunc(x):
    if x > 0:
        return x
    else:
        return x / 100

def leakyReLuFuncPrime(x):
    if x > 0:
        return 1
    else:
        return 0.01

np.random.seed(0)

X = [[1, 2],
     [2, 2],
     [-1.5, 2.7]]

redXs = []
redYs = []
greenXs = []
greenYs = []

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, func, funcPrime):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.func = func
        self.funcPrime = funcPrime
    def forward(self, inputs):
        self.inputs = inputs
        self.states = np.dot(self.inputs, self.weights) + self.biases
        state = np.dot(self.inputs, self.weights)
        print(state)
        print(self.states)
        return self.activate()
    def activate(self):
        self.output = []
        for s in self.states:
            out = []
            for i in s:
                out.append(self.func(i))
            self.output.append(out)
        return self.output[0]
    def backward(self, grad):
        self.gradPrime = []
        for s in self.states:
            out = []
            for i in s:
                out.append(self.funcPrime(i))
            self.gradPrime.append(out)
        return self.backwardActivate()
    def backwardActivate(self):
        self.grad = self.gradPrime
        self.gradOut = np.dot(self.weights, np.transpose(self.grad))
        return self.gradOut[0]
    def adjust(self, eta):
        correction = eta * np.dot(np.array(self.grad).T, [self.inputs])
        """print(correction)
        #self.weights += (eta * np.dot(np.array(self.grad).T, self.inputs))
        #self.weights += correction
        print(self.weights)
        self.weights = np.add(self.weights, correction.T)
        print("ok")
        print(self.weights)"""
        self.weights = np.add(self.weights, correction.T)




def generateCords():
    points = 1
    cords = []
    for i in range(0, points):
        randCord = np.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    samps = 20
    sampsCords = []
    for i in cords:
        for j in range(0, samps):
            var = np.random.uniform(0.005, 0.035)
            randCord = np.random.normal(loc=i, scale=var)
            sampsCords.append(randCord)
    return sampsCords

def prepareInputs(cords):
    inputs = []
    for (x, y) in cords:
        pair = []
        pair.append(x)
        pair.append(y)
        inputs.append(pair)
    return inputs

def prepareInputsLabels(cords, label):
    inputs = []
    for (x, y) in cords:
        pair = []
        labeledPair = []
        pair.append(x)
        pair.append(y)
        labeledPair.append(pair)
        labeledPair.append(label)
        inputs.append(labeledPair)
    return inputs

def prepareTrainingData(redIns, greenIns):
    inputs = []
    for r in redIns:
        inputs.append(r)
    for g in greenIns:
        inputs.append(g)
    random.shuffle(inputs)
    return inputs





def train(data, epochs):
    global layers
    for e in range(0, epochs):
        for (ins, label) in data:
            for layer in layers:
                #print("layer")
                ins = layer.forward(ins)
                #print(layer.inputs)

            #print(label)
            #print(ins)
            
            grad = 2 * np.subtract(label, ins)
            #print(label)
            #print(ins)
            #print(grad)

            for layer in layers[::-1]:
                grad = layer.backward(grad)
                #print(layer.inputs)
                #print(layer.grad)
                #print(layer.weights)
            
            for layer in layers:
                #print("lel")
                """print(layer.grad)
                print([layer.inputs])
                print(np.array(layer.grad).T)
                print(np.dot(np.array(layer.grad).T, [layer.inputs]))
                print(layer.inputs)
                print(np.transpose(layer.inputs))
                print(np.array([layer.inputs]).T)
                print(layer.weights)"""

                
                #print(np.dot(layer.grad, layer.inputs))
                layer.adjust(0.01)
    print("done")



redXs = generateCords()
redYs = generateCords()
greenXs = generateCords()
greenYs = generateCords()

redInputs = prepareInputsLabels(zip(redXs, redYs), [1, 0])
greenInputs = prepareInputsLabels(zip(greenXs, greenYs), [0, 1])
greenInputsTest = prepareInputs(zip(greenXs, greenYs))
redInputsTest = prepareInputs(zip(redXs, redYs))

"""redInputs = prepareInputs(zip(redXs, redYs))
greenInputs = prepareInputs(zip(greenXs, greenYs))"""

trainingData = prepareTrainingData(redInputs, greenInputs)

layers = []

testG = greenInputsTest[0]
testR = redInputsTest[0]
print(testG)
print(testR)

active = sinFunc
activePrime = sinFuncPrime

layer1 = Layer_Dense(2, 2, active, activePrime)
layers.append(layer1)
layer2 = Layer_Dense(2, 4, active, activePrime)
layers.append(layer2)
layer3 = Layer_Dense(4, 2, active, activePrime)
layers.append(layer3)

for layer in layers:
    print("\n")
    print(layer.weights)

print("\n\n")

train(trainingData, 20)

for layer in layers:
    testG = layer.forward(testG)
    print(layer.output)

for layer in layers:
    testR = layer.forward(testR)
    #print(layer.output)
    print("\n")
    print(layer.weights)


#print(trainingData)

"""ins = redInputs
for i in ins:
    for layer in layers:
        print(i)
        i = layer.forward(i)
        print(layer.output, "\n\n")"""






"""layer1.forward(redInputs)
print(layer1.output, "\n")
layer2.forward(layer1.output)
print(layer2.output, "\n")
layer3.forward(layer2.output)
print(layer3.output, "\n\n")
print(redInputs)
print("\n\n")"""
#print(X)