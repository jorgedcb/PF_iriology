from tkinter import ttk
from tkinter import *
from tkinter.messagebox import showinfo
import tkinter as tk                    
from tkinter import filedialog
from PIL import Image, ImageTk
from iris_detection import IrisDetection
import cv2
import customtkinter as cstk

cstk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
cstk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Table:
     
    def __init__(self,root,lst):
        total_rows = len(lst)
        total_columns = len(lst[0])
        
        for i in range(total_rows):
            for j in range(total_columns):
                 
                self.e = Entry(root, width=45, fg='black', bg = '#C5C5C5',
                               font=('satoshi',14,'bold'))
                 
                self.e.grid(row=i, column=j)
                self.e.insert(END, lst[i][j])


class App(cstk.CTk):

    def __init__(self):
        super().__init__()

        style = ttk.Style()
        settings = {"TNotebook.Tab": {"configure": {"padding": [5, 1],
                                                    "background": "gray",
                                                    "bordercolor":"black"
                                                    }
                                     },
                        "TNotebook": {"configure": {"background": "#696969",
                                                    "foreground":"#696969"
                                                    }
                                     },
                        "TFrame": {"configure": {"background": "#696969",
                                                    "foreground":"#696969"
                                                    }
                                     }
           }  
        style.theme_create("mi_estilo", parent="alt", settings=settings)
        style.theme_use("mi_estilo")

        self._eye = None
        self._img = None
        self.title("Herramienta de apoyo iridológica")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry("%dx%d" % (screen_width/1.7, screen_height))

        tabControl = ttk.Notebook(self, style="TNotebook")
        tab1 = ttk.Frame(tabControl, style="TFrame")
        self.tab1 = tab1
        tab2 = ttk.Frame(tabControl, style="TFrame")
        self.tab2 = tab2
        tab3 = ttk.Frame(tabControl, style="TFrame")
        self.tab3 = tab3
        tabControl.add(tab1, text ='   Carga y Ajustes   ')
        tabControl.add(tab2, text ='     Resultados      ')
        tabControl.add(tab3, text ='   Tabla Resultados  ')
        tabControl.pack(expand = 1, fill ="both")
    	
        framet1 = cstk.CTkFrame(self.tab1)
        framet1.pack()
        l1 = cstk.CTkLabel(framet1, text='  Seleccione imagen para empezar  ',width=40,text_font=('satoshi', 18))  
        l1.grid(row=1,column=1)
        b1 = cstk.CTkButton(framet1, text='Cargar foto', 
        width=30,text_font=('satoshi', 14),command = lambda:self.upload_file(), hover_color="#808080", border_color="#3B3B3B",fg_color="#696969")
        b1.grid(row=2,column=1)
        lesp = cstk.CTkLabel(framet1, text=' ',width=40,text_font=('satoshi', 4))  
        lesp.grid(row=3,column=1)
        self._framet1 = None
        self._framet2 = None
        self._framet3 = None
        self._framet4 = None #para arreglar el problema

    def upload_file(self):
        #f_types = [('Jpg Files', '*.jpg')]
        f_types = [('Png Files', '*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        print(filename)
        percentil = 53.3 #aqui le pones el valor que tu quieras
        self._eye = IrisDetection(filename,percentil)
        self._eye.run_diagnostic()
        image=Image.open(filename)

        w_o, h_o = image.size
        rel = self.winfo_screenheight()/h_o
        iwi = int(rel*w_o/1.8)
        ihi = int(rel*h_o/1.8)

        imgrz=image.resize((iwi, ihi))
        img = ImageTk.PhotoImage(imgrz)
        self._img = img
        if self._framet1: #para arreglar el problema
            self._framet1.destroy()
        self._framet1 = cstk.CTkFrame(self.tab1)
        
        self._framet1.pack()
        b =tk.Button(self._framet1,image=img) # using CTkButton 
        b.grid(row=4,column=1)
        original_image = self._eye.color_diagnostic_image
        segmetation_image = self._eye.gray_scale_segmentate_image
        baw_detection = self._eye.gray_scale_diagnostic_image
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        chart_image = cv2.imread("Chart.jpg")

        iw = int(rel*w_o/2.2)
        ih = int(rel*h_o/2.2)

        chart = Image.fromarray(chart_image).resize((iw, ih))
        img_chart = ImageTk.PhotoImage(image=chart)

        original = Image.fromarray(original_image).resize((iw, ih))
        img_original = ImageTk.PhotoImage(image=original)
        
        detection = Image.fromarray(baw_detection).resize((iw, ih))
        img_detection = ImageTk.PhotoImage(image=detection)

        segmetation = Image.fromarray(segmetation_image).resize((iw, ih))
        img_segmetation = ImageTk.PhotoImage(image=segmetation)
        if self._framet2:
            self._framet2.destroy()
        self._framet2 = cstk.CTkFrame(self.tab2)
        self._framet2.pack()
        b1 =tk.Button(self._framet2,image=img_chart) # using Button 
        b1.grid(row=1,column=1)
        b2 =tk.Button(self._framet2,image=img_original) # using Button 
        b2.grid(row=1,column=2)
        b3 =tk.Button(self._framet2,image= img_segmetation) # using Button 
        b3.grid(row=2,column=1)
        b4 =tk.Button(self._framet2,image=img_detection) # using Button 
        b4.grid(row=2,column=2)

        if self._framet3:
            self._framet3.destroy()
        self._framet3 = cstk.CTkFrame(self.tab3)
        self._framet3.pack()
        l2 = cstk.CTkLabel(self._framet3, text='   Análisis Preeliminar   ',width=30,text_font=('satoshi', 18))  
        l2.grid(row=1,column=1)
        lesp2 = cstk.CTkLabel(self._framet3, text=' ',width=40,text_font=('satoshi', 4))  
        lesp2.grid(row=2,column=1)
        
        if self._framet4:
            self._framet4.destroy()
        self._framet4 = cstk.CTkFrame(self.tab3)
        self._framet4 .pack()
        b5 = Table(self._framet4 , self._eye.results())
        self.mainloop()  
        

    def button_clicked(self):
        showinfo(title='Information', message='Hello, Tkinter!')