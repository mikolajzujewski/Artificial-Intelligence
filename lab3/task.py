import numpy as np


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
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.func = func
        self.funcPrime = funcPrime
    def forward(self, inputs):
        self.inputs = inputs
        self.states = np.dot(self.inputs, self.weights) + self.biases
        return self.activate()
    def activate(self):
        self.output = []
        for s in self.states:
            out = []
            for i in s:
                out.append(self.func(i))
            self.output.append(out)
        return self.output


def generateCords():
    points = 2
    cords = []
    for i in range(0, points):
        randCord = np.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    samps = 5
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


#def train():


redXs = generateCords()
redYs = generateCords()
greenXs = generateCords()
greenYs = generateCords()

redInputs = prepareInputs(zip(redXs, redYs))
greenInputs = prepareInputs(zip(greenXs, greenYs))

layers = []



active = ReLuFunc
activePrime = ReLuFuncPrime

layer1 = Layer_Dense(2, 2, active, activePrime)
layers.append(layer1)
layer2 = Layer_Dense(2, 4, active, activePrime)
layers.append(layer2)
layer3 = Layer_Dense(4, 2, active, activePrime)
layers.append(layer3)

ins = redInputs
for i in ins:
    for layer in layers:
        i = layer.forward(i)
        print(layer.output, "\n\n")






"""layer1.forward(redInputs)
print(layer1.output, "\n")
layer2.forward(layer1.output)
print(layer2.output, "\n")
layer3.forward(layer2.output)
print(layer3.output, "\n\n")
print(redInputs)
print("\n\n")"""
#print(X)