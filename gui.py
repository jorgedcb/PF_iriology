from tkinter import ttk
from tkinter import *
from tkinter.messagebox import showinfo
import tkinter as tk                    
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from iris_detection import IrisDetection
import cv2

class Table:
     
    def __init__(self,root,lst):
        total_rows = len(lst)
        total_columns = len(lst[0])
        
        for i in range(total_rows):
            for j in range(total_columns):
                 
                self.e = Entry(root, width=20, fg='blue',
                               font=('Arial',16,'bold'))
                 
                self.e.grid(row=i, column=j)
                self.e.insert(END, lst[i][j])


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self._eye = None
        self._img = None
        self.title("Tab Widget")
        tabControl = ttk.Notebook(self)
        
        tab1 = ttk.Frame(tabControl)
        self.tab1 = tab1
        tab2 = ttk.Frame(tabControl)
        self.tab2 = tab2
        tab3 = ttk.Frame(tabControl)
        self.tab3 = tab3
        tabControl.add(tab1, text ='Carga')
        tabControl.add(tab2, text ='Resultados')
        tabControl.add(tab3, text ='Tabla Resultados')
        tabControl.pack(expand = 1, fill ="both")

        my_font1=('times', 18, 'bold')
        l1 = tk.Label(tab1,text='Seleccione imagen para empezar',width=30,font=my_font1)  
        l1.grid(row=1,column=1)
        b1 = tk.Button(tab1, text='Cargar foto', 
        width=20,command = lambda:self.upload_file())
        b1.grid(row=2,column=1) 

        my_font1=('times', 18, 'bold')
        l1 = tk.Label(tab2,text='Seleccione imagen para empezar',width=30,font=my_font1)  
        l1.grid(row=1,column=1)

    def upload_file(self):
        #f_types = [('Jpg Files', '*.jpg')]
        f_types = [('Png Files', '*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        print(filename)
        img = ImageTk.PhotoImage(file=filename)
        self._img = img
        b2 =tk.Button(self.tab1,image=img) # using Button 
        b2.grid(row=3,column=1)
        self._eye = IrisDetection(filename)
        self._eye.run_diagnostic()
        original_image = self._eye.color_diagnostic_image
        segmetation_image = self._eye.gray_scale_segmentate_image
        baw_detection = self._eye.gray_scale_diagnostic_image
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        original = Image.fromarray(original_image)
        img_original = ImageTk.PhotoImage(image=original)
        
        
        detection = Image.fromarray(baw_detection)
        img_detection = ImageTk.PhotoImage(image=detection)

        segmetation = Image.fromarray(segmetation_image)
        img_segmetation = ImageTk.PhotoImage(image=segmetation)

        b2 =tk.Button(self.tab2,image=img_original) # using Button 
        b2.grid(row=2,column=1)
        b3 =tk.Button(self.tab2,image= img_segmetation) # using Button 
        b3.grid(row=3,column=1)
        b4 =tk.Button(self.tab2,image=img_detection) # using Button 
        b4.grid(row=2,column=2)
        b5 = Table(self.tab3,self._eye.results())
        self.mainloop()  


    def button_clicked(self):
        showinfo(title='Information', message='Hello, Tkinter!')

