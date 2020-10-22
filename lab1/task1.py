import tkinter
import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy

root = tkinter.Tk()
root.wm_title("Sample genrator")

figure = Figure(figsize=(5, 5), dpi=150)
plot = figure.add_subplot(1,1,1)
plot.axis([0, 1.1, 0, 1.1])

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack()

mods = tkinter.Entry(root)
samples = tkinter.Entry(root)


def _quit():
    root.quit()

def _update():
    try:
        redXs = generateCords()
        redYs = generateCords()
        greenXs = generateCords()
        greenYs = generateCords()
        plot.cla()
        plot.axis([0, 1.1, 0, 1.1])
        plot.plot(redXs, redYs, 's', color="red", marker="o")
        plot.plot(greenXs, greenYs, 's', color="green", marker="o")
        canvas.get_tk_widget().pack()
        canvas.draw()
    except:
        print("Something is wrong, update canceled")

def generateCords():
    rawPoints = mods.get()
    points = int(rawPoints)
    cords = []
    for i in range(0, points):
        randCord = numpy.random.rand()
        cords.append(randCord)
    return generateSamples(cords)

def generateSamples(cords):
    rawSamps = samples.get()
    samps = int(rawSamps)
    var = numpy.random.uniform(0.005, 0.035)
    sampsCords = []
    for i in cords:
        for j in range(0, samps):
            randCord = numpy.random.normal(loc=i, scale=var)
            sampsCords.append(randCord)
    return sampsCords


labelModsText = tkinter.StringVar()
labelSamplesText = tkinter.StringVar()
labelModsText.set("Number of mods:")
labelSamplesText.set("Number of samples:")
labelMods = tkinter.Label(root, textvariable=labelModsText)
labelSamples = tkinter.Label(root, textvariable=labelSamplesText)
labelMods.pack()
mods.pack()
labelSamples.pack()
samples.pack()
updateButton = tkinter.Button(master=root, text="Update", command=_update)
exitButton = tkinter.Button(master=root, text="Exit", command=_quit)
updateButton.pack()
exitButton.pack()


root.mainloop()

