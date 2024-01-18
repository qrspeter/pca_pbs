# Made by first-year students of the educational program "Physics of Nanostructures" ITMO (Ivan, Tigran, Igor and Roman).

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import os
import numpy as np

def tigrilla(path, lmin, lmax, step):
    arr = np.loadtxt(path)
    normarr = np.array(arr[:,1]/np.max(arr[:,1]))
    x = np.arange(lmin, lmax, step)
    interp = np.interp(x, arr[:,0], normarr, left=0, right=0)
    return interp

def activ(x):
    if x>0.5:
        return "Спектр класса А (левый)"
    else:
        return "Спектр класса В (правый)"
        
def tutor(list_X,list_Y,iterations=10000,tolerance=0.0001,learn_rate=0.001):
    weights = np.zeros((list_X.shape[1],1),float)
    iteration=0
    while iteration<iterations:
        error=list_Y-np.dot(list_X,weights)
        if abs(np.mean(list_Y))<tolerance:
            print('Выпускной!')
            break
        else:
            weights+=np.dot(list_X.T,error*learn_rate)
        iteration+=1
    return weights

def test_file():
    filetypes = (
        ('Данные древних Римлян', '*.arc_data*'),
        ('Текстовые файлы', '*.txt'),
        ('Уверен что нужны все файлы?', '*.*'))    
    filename = fd.askopenfilename( # получить имя файла
        title='Откройте файл',
        initialdir=' ',
        filetypes=filetypes)
    y_final=tigrilla(filename, lmin, lmax, step)
    y_final=y_final.reshape(1,1100)
    msg=activ(np.dot(y_final,weights))
    showinfo(
        title='заголовок',
        message=msg)

lmin = 800
lmax = 1900
step = 1
y=np.array([[1],[1],[1],[1],[1],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]])

path_input = r"./csv/"
files = os.listdir(path_input)
result_matrix = np.zeros((len(files),(lmax-lmin)//step))
for n,elem in enumerate(files):
    this_path = path_input + r'' + '/' + elem
    this_file = open(this_path)
    result_y = tigrilla(this_path, lmin, lmax, step)
    result_matrix[n]=(result_y)    
if __name__ == "__main__":
    root = tk.Tk()
    root.title('Распределитель')
    root.resizable(False, False)
    root.geometry('300x250')
    weights=tutor(result_matrix,y)
    open_button = ttk.Button(   # open button
        root,
        text='Open a File',
        command=test_file)    
    open_button.pack(expand=True)    
    # run the application
    root.mainloop()
