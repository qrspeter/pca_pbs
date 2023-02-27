# initial idea from  https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter.filedialog import askopenfile
from numpy import exp, array, random, dot
import numpy as np
# import pandas as pd

inputs = 551 
trainings = 3 #10000


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with "inputs" input connections and 1 output connection.
        # We assign random weights to a "inputs" x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((inputs, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


def opening(file_name):
    spectra = []
    with open(file_name, 'r', encoding='cp1251') as fin:
        for line in fin:
            if line[0] != '#':
                if line == '\n':
                    continue

                now = line.strip().split('\t')
                spectra.append(list(map(float, now)))
                numspectra = np.array(spectra)
    # нормализация по амплитуде
    intense = [i[1] for i in numspectra] # а если сразу открыть файл в нумпай массив и транспонировать, то обращались бы просто по имени к строкам
    normalizedIntense = (intense - np.min(intense)) / (np.max(intense) - np.min(intense))
#    print(np.max(intense), np.min(intense), np.max(intense) - np.min(intense))
    length = [j[0] for j in numspectra]
    # нормализация по длине волны от 800 до 1900 нм
    n_before = (int(length[0]) - 800)//2
    n_after = (1900 - int(length[-1]))//2
    zeroes_before = np.zeros(n_before)
    zeroes_after = np.zeros(n_after)
    result = np.concatenate([zeroes_before, normalizedIntense, zeroes_after])
    return result


def test_file():
    
    filetypes = (
        ('ascii files', '*.arc_data'), # расширение можно сменить на *.arc_data, чтобы сразу были видны нужные файлы
        ('text files', '*.txt'), # расширение можно сменить на *.arc_data, чтобы сразу были видны нужные файлы
        ('All files', '*.*')
    )
    
    
    
    f = fd.askopenfile(filetypes=filetypes)
    arr = opening(f.name)  # считать файл f собственной функцией открытие_файла(), а не через readlines
    
    result = neural_network.think(arr)
    
    if result < 0.5:
        mess = '1060'
    else:
        mess = '1640'
    
    showinfo(title='Result', message=mess) # message="Результат")


if __name__ == "__main__":


    data_dir = './data/' # лучше делать перебор.. хотя тогда не понять какой это образец.. или из разных папок брать
    
    training_set_inputs = np.zeros((12, inputs))
    training_set_inputs[0] = opening(data_dir + 'PbS1060 +iso on glass drop.arc_data')
    training_set_inputs[1] = opening(data_dir + 'PbS1060 on glass drop.arc_data')
    training_set_inputs[2] = opening(data_dir + 'PbS1060 on micadrop.arc_data')
    training_set_inputs[3] = opening(data_dir + 'PbS1060+acet drop.arc_data')
    training_set_inputs[4] = opening(data_dir + 'PbS1060+met drop.arc_data')
    training_set_inputs[5] = opening(data_dir + 'PbS1060+iso drop.arc_data')

    training_set_inputs[6] = opening(data_dir + 'PbS1640 on glass drop.arc_data')
    training_set_inputs[7] = opening(data_dir + 'PbS1640 on mica drop.arc_data')
    training_set_inputs[8] = opening(data_dir + 'PbS1060+iso drop.arc_data')
    training_set_inputs[9] = opening(data_dir + 'PbS1640+acet drop.arc_data')
    training_set_inputs[10] = opening(data_dir + 'PbS1640+all drop.arc_data')
    training_set_inputs[11] = opening(data_dir + 'PbS1640+iso drop.arc_data')
    
    print(training_set_inputs)
    

    neural_network = NeuralNetwork()
    training_set_outputs = array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T #нужно знать
    neural_network.train(training_set_inputs, training_set_outputs, trainings) #тренируем

#    df = pd.DataFrame(training_set_inputs)
#    wavelengths = list(range(800, 1901, 2))
#    print(len(wavelengths))
#    print(wavelengths)
#    df.columns =  map(str, wavelengths)
#    df['PbS'] = array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T
#    df.to_csv('./data/pbs.csv')


    # create the root window
    root = tk.Tk()
    root.title("Let's check a spectrum!")
    root.resizable(False, False)
    root.geometry('300x150')

    open_button = ttk.Button(
        root,
        text='Open a File',
        command=test_file
    )  # вызывается команда загрузки-обработки-классификации спектра
    open_button.pack(expand=True)

    # run the application
    root.mainloop()

