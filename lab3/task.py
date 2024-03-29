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
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.func = func
        self.funcPrime = funcPrime
    def forward(self, inputs):
        self.inputs = inputs
        self.states = np.dot(self.inputs, (self.weights).T)
        return self.activate()
    def activate(self):
        self.output = []
        for s in self.states:
            self.output.append(self.func(s))
        return self.output
    def backward(self, grad):
        self.statesPrime = []
        for s in self.states:
            self.statesPrime.append(self.funcPrime(s))
        self.gradPrime = []
        for i in range(0, len(grad)):
            self.gradPrime.append(self.statesPrime[i] * grad[i])
        return self.backwardActivate()
    def backwardActivate(self):
        self.grad = self.gradPrime
        self.gradOut = np.dot(self.grad, self.weights)
        return self.gradOut
    def adjust(self, eta):
        correction = np.dot(np.array([self.grad]).T, np.array([self.inputs])) * eta
        self.weights = np.add(self.weights, correction)





def generateCords():
    points = 1
    cords = []
    for i in range(0, points):
        randCord = np.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    samps = 30
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
            #print(ins)
            for layer in layers:
                #print("layer")
                ins = layer.forward(ins)
                #print(layer.inputs)

            #print(label)
            #print(ins)
            
            #grad = np.subtract(label, ins)

            #grad = np.array([label[0] - ins[0], label[1] - ins[1]]) * 2
            grad = np.subtract(np.array(label), np.array(ins)) * 2
            
            

            """print(label)
            print(ins)
            print("grad ->", grad)"""

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
                layer.adjust(0.25)
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

testG = greenInputsTest[5]
testR = redInputsTest[0]


active = sigmoidFunc
activePrime = sigmoidFuncPrime

layer1 = Layer_Dense(2, 4, active, activePrime)
layers.append(layer1)
layer2 = Layer_Dense(4, 3, active, activePrime)
layers.append(layer2)
layer3 = Layer_Dense(3, 5, active, activePrime)
layers.append(layer3)
layer4 = Layer_Dense(5, 2, active, activePrime)
layers.append(layer4)

"""for layer in layers:
    print("\n")
    print(layer.weights)"""

print("\n\n")

train(trainingData, 400)

print("green")

for layer in layers:
    testG = layer.forward(testG)
    print("<")
    print(layer.output)
    print(">")

print("red")

for layer in layers:
    testR = layer.forward(testR)
    print("<")
    print(layer.output)
    print(">")
    
"""


#print(trainingData)

ins = redInputs
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