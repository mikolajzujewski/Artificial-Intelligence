import tkinter
import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy


def heavisideFunc(x):
    if x < 0:
        return 0
    else:
        return 1

def heavisideFuncPrime(x):
    return 1

def sigmoidFunc(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoidFuncPrime(x):
    return sigmoidFunc(x) * (1 - sigmoidFunc(x))

def sinFunc(x):
    return numpy.sin(x)

def sinFuncPrime(x):
    return numpy.cos(x)

def tanhFunc(x):
    return numpy.tanh(x)

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




class Neuron():
    def __init__(self, func, funcPrime, rate):
        #self.weights = [numpy.random.rand(), -numpy.random.rand()]
        self.weights = [1, -0.5]
        self.states = []
        self.func = func
        self.funcPrime = funcPrime
        self.rate = rate
        self.bias = 0

    def count(self, inputs):
        outputs = []
        for i in inputs:
            state = numpy.dot(i, self.weights) + self.bias
            self.states.append(state)
            y = self.func(state)
            outputs.append(y)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    
    def practice(self, inputs, outputs, label):
        corrections = []
        for o, i in zip(outputs, inputs):
            delta = numpy.dot(i, (self.rate*(label - o)*self.funcPrime(numpy.dot(i, self.weights) + self.bias)))
            self.weights += delta
            corrections.append(delta)

            """
            delta = []
            delta1 = self.rate * (label - o)* self.funcPrime(i[0] * self.weights[0]) * i[0]
            delta2 = self.rate * (label - o)* self.funcPrime(i[1] * self.weights[1]) * i[1]
            delta.append(delta1)
            delta.append(delta2)
            corrections.append(delta)
            self.weights[0] += delta1
            self.weights[1] += delta2

            """



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
    points = 2
    cords = []
    for i in range(0, points):
        randCord = numpy.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    samps = 100
    sampsCords = []
    for i in cords:
        for j in range(0, samps):
            var = numpy.random.uniform(0.005, 0.035)
            randCord = numpy.random.normal(loc=i, scale=var)
            sampsCords.append(randCord)
    return sampsCords

def _training():
    rawTrain = trainings.get()
    reps = int(rawTrain)
    for r in range(0, reps):
        redOut = neuron.count(zip(redXs, redYs))
        greenOut = neuron.count(zip(greenXs, greenYs))
        neuron.practice(zip(redXs, redYs), redOut, 1)
        neuron.practice(zip(greenXs, greenYs), greenOut, 0)


def _drawContour():
    drawModes()
    X = numpy.arange(0.0, 1.01, 0.01)
    Y = numpy.arange(0.0, 1.01, 0.01)
    Z = []
    for (i, x) in enumerate(X):
        Z.append([])
        for y in Y:
            Z[i].append(neuron.count(zip([x], [y])))

    contour = plot.contourf(X, Y, Z)

    contour.clabel(colors="black")
    canvas.get_tk_widget().pack()
    canvas.draw()



redXs = []
redYs = []
greenXs = []
greenYs = []

#################################/////////////////////////////////////////////////////////////

neuron = Neuron(ReLuFunc, ReLuFuncPrime, 0.01)

#################################/////////////////////////////////////////////////////////////



root = tkinter.Tk()
root.wm_title("Sample genrator")

figure = Figure(figsize=(5, 5), dpi=120)
plot = figure.add_subplot(1,1,1)
plot.axis([0, 1, 0, 1])

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack()

modes = tkinter.Entry(root)
samples = tkinter.Entry(root)
trainings = tkinter.Entry(root)




labelModsText = tkinter.StringVar()
labelSamplesText = tkinter.StringVar()
labelTrainingsText = tkinter.StringVar()
labelModsText.set("Number of modes:")
labelSamplesText.set("Number of samples:")
labelTrainingsText.set("Number of trainings:")
labelMods = tkinter.Label(root, textvariable=labelModsText)
labelSamples = tkinter.Label(root, textvariable=labelSamplesText)
labelTrainings = tkinter.Label(root, textvariable=labelTrainingsText)
labelTrainings.pack()
trainings.pack()
updateButton = tkinter.Button(master=root, text="Update", command=_update)
exitButton = tkinter.Button(master=root, text="Exit", command=_quit)
testButton = tkinter.Button(master=root, text="Train", command=_training)
pletButton = tkinter.Button(master=root, text="Draw", command=_drawContour)
updateButton.pack()
testButton.pack()
pletButton.pack()
exitButton.pack()


root.mainloop()

