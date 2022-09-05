from tkinter import ttk
from tkinter.messagebox import showinfo
import tkinter as tk                    
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from iris_detection import Iris_detection

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self._iris = None
        self._img = None
        self.title("Tab Widget")
        tabControl = ttk.Notebook(self)
        
        tab1 = ttk.Frame(tabControl)
        self.tab1 = tab1
        tab2 = ttk.Frame(tabControl)
        self.tab2 = tab2
        tabControl.add(tab1, text ='Carga')
        tabControl.add(tab2, text ='Resultados')
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
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        print(filename)
        img = ImageTk.PhotoImage(file=filename)
        self._img = img
        b2 =tk.Button(self.tab1,image=img) # using Button 
        b2.grid(row=3,column=1)
        self._iris = Iris_detection(filename)
        img = self._iris.start_detection()
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        b2 =tk.Button(self.tab2,image= self._img) # using Button 
        b2.grid(row=2,column=1)
        b3 =tk.Button(self.tab2,image= self._img) # using Button 
        b3.grid(row=3,column=1)
        b4 =tk.Button(self.tab2,image=imgtk) # using Button 
        b4.grid(row=2,column=2)
        self.mainloop()  


    def button_clicked(self):
        showinfo(title='Information', message='Hello, Tkinter!')

if __name__ == "__main__":
  app = App()
  app.mainloop()