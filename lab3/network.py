import numpy as np
import random

import tkinter
import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


redXs = []
redYs = []
greenXs = []
greenYs = []

class Layer:
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
        return self.backwardLinear()
    def backwardLinear(self):
        self.grad = self.gradPrime
        self.gradOut = np.dot(self.grad, self.weights)
        return self.gradOut
    def adjust(self):
        correction = np.dot(np.array([self.grad]).T, np.array([self.inputs])) * ETA
        self.weights = np.add(self.weights, correction)



def _quit():
    root.quit()

def _update():
    try:
        global redXs, redYs, greenXs, greenYs
        redXs = generateCords()
        redYs = generateCords()
        greenXs = generateCords()
        greenYs = generateCords()
        drawModes()
        canvas.get_tk_widget().pack()
        canvas.draw()
    except:
        print("Something is wrong, update canceled")

def drawModes():
    plot.cla()
    plot.axis([0, 1, 0, 1])
    plot.plot(redXs, redYs, 's', color="red", marker="o")
    plot.plot(greenXs, greenYs, 's', color="green", marker="o")
    canvas.get_tk_widget().pack()
    canvas.draw()
    



def generateCords():
    points = MODES
    cords = []
    for i in range(0, points):
        randCord = np.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    samps = SAMPS
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

def _drawContour():
    global layers
    drawModes()
    X = np.arange(0.0, 1.01, 0.01)
    Y = np.arange(0.0, 1.01, 0.01)
    Z = []
    for (i, x) in enumerate(X):
        Z.append([])
        for y in Y:
            inp = [x, y]
            for layer in layers:
                inp = layer.forward(inp)
            Z[i].append(layers[-1].output[0])

    contour = plot.contourf(X, Y, Z)

    contour.clabel(colors="black")
    canvas.get_tk_widget().pack()
    canvas.draw()




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
                ins = layer.forward(ins)

            grad = np.subtract(np.array(label), np.array(ins)) * 2
            
            for layer in layers[::-1]:
                grad = layer.backward(grad)
            
            for layer in layers:
                layer.adjust()
    print("done")


def _training():
    global redInputs, greenInputs, redXs, redYs, greenXs, greenYs
    redInputs = prepareInputsLabels(zip(redXs, redYs), [1, 0])
    greenInputs = prepareInputsLabels(zip(greenXs, greenYs), [0, 1])
    trainingData = prepareTrainingData(redInputs, greenInputs)
    rawEpochs = epochs.get()
    n_epochs = int(rawEpochs)
    train(trainingData, n_epochs)



#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

active = tanhFunc
activePrime = tanhFuncPrime

ETA = 0.25
MODES = 1
SAMPS = 40

# Layer = (inputs, neurons, func, funcPrime)


layers = []


layer1 = Layer(2, 2, active, activePrime)
layers.append(layer1)
layer2 = Layer(2, 4, active, activePrime)
layers.append(layer2)
layer3 = Layer(4, 4, active, activePrime)
layers.append(layer3)
layer4 = Layer(4, 2, active, activePrime)
layers.append(layer4)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



root = tkinter.Tk()
root.wm_title("Sample genrator")

figure = Figure(figsize=(5, 5), dpi=120)
plot = figure.add_subplot(1,1,1)
plot.axis([0, 1, 0, 1])

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack()

modes = tkinter.Entry(root)
samples = tkinter.Entry(root)
epochs = tkinter.Entry(root)




labelModsText = tkinter.StringVar()
labelSamplesText = tkinter.StringVar()
labelEpochsText = tkinter.StringVar()
labelEpochsText.set("Number of epochs:")
labelMods = tkinter.Label(root, textvariable=labelModsText)
labelSamples = tkinter.Label(root, textvariable=labelSamplesText)
labelEpochs = tkinter.Label(root, textvariable=labelEpochsText)
labelEpochs.pack()
epochs.pack()
updateButton = tkinter.Button(master=root, text="Update", command=_update)
exitButton = tkinter.Button(master=root, text="Exit", command=_quit)
testButton = tkinter.Button(master=root, text="Train", command=_training)
pletButton = tkinter.Button(master=root, text="Draw", command=_drawContour)
updateButton.pack()
testButton.pack()
pletButton.pack()
exitButton.pack()


root.mainloop()