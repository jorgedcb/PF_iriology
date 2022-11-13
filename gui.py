from tkinter import ttk, CENTER, NO
from tkinter.messagebox import showinfo
import tkinter as tk                    
from tkinter import filedialog
from PIL import Image, ImageTk
from iris_detection import IrisDetection
import cv2
import customtkinter as cstk

cstk.set_appearance_mode("Dark")
cstk.set_default_color_theme("dark-blue")

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
        style.configure("mi_estilo.Treeview", highlightthickness=0, bd=0, font=('satoshi', 16), rowheight=40, background="#c9c9c9" ) 
        style.configure("mi_estilo.Treeview.Heading", font=('satoshi', 17,'bold'), rowheight=45, background="#bdbdbd")

        self._eye = None
        self._img = None
        self.title("Herramienta de apoyo iridológica")
        self.state('zoomed')

        tab_control = ttk.Notebook(self, style="TNotebook")
        tab1 = ttk.Frame(tab_control, style="TFrame")
        self.tab1 = tab1
        tab2 = ttk.Frame(tab_control, style="TFrame")
        self.tab2 = tab2
        tab_control.add(tab1, text ='   Carga y Ajustes   ')
        tab_control.add(tab2, text ='     Resultados      ')
        tab_control.pack(expand = 1, fill ="both")

        framet1 = cstk.CTkFrame(self.tab1)
        framet1.pack()

        lper = cstk.CTkLabel(framet1, text='  Seleccione el umbral de detección  ',width=40,text_font=('satoshi', 12))  
        lper.grid(row=1,column=1)
        segemented_button_var = cstk.StringVar(value="5%")  # set initial value
        segemented_button = cstk.CTkComboBox(master=framet1,
                                              values=["3%", "5%", "7%", "9%"],
                                              variable=segemented_button_var)
        segemented_button.grid(row=2,column=1) 

        l1 = cstk.CTkLabel(framet1, text='  Seleccione imagen para empezar  ',width=40,text_font=('satoshi', 18))  
        l1.grid(row=3,column=1)
        b1 = cstk.CTkButton(framet1, text='Cargar foto', 
        width=30,text_font=('satoshi', 14),command = lambda : self.upload_file(percental_string=segemented_button.get()), hover_color="#808080", border_color="#3B3B3B",fg_color="#696969")
        b1.grid(row=4,column=1)
        lesp = cstk.CTkLabel(framet1, text=' ',width=40,text_font=('satoshi', 4))  
        lesp.grid(row=5,column=1)   
        self._framet1 = None
        self._framet2 = None
        self._framet3 = None
        self._framet4 = None

    def upload_file(self, percental_string):
        f_types = [('Png Files', '*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        print(filename)
        if percental_string == "3%": 
            percetage_value = 3
        elif percental_string == "5%": 
            percetage_value = 5
        elif percental_string == "7%": 
            percetage_value = 7
        elif percental_string == "9%": 
            percetage_value = 9
        percentil = percetage_value
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
        if self._framet1:
            self._framet1.destroy()
        self._framet1 = cstk.CTkFrame(self.tab1)
        
        self._framet1.pack()
        b =tk.Button(self._framet1,image=img)
        b.grid(row=6,column=1)
        original_image = self._eye.color_diagnostic_image
        segmetation_image = self._eye.gray_scale_segmentate_image
        baw_detection = self._eye.gray_scale_diagnostic_image
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        chart_image = cv2.imread("Chart.jpg")

        iw = int(rel*w_o/2.3)
        ih = int(rel*h_o/2.3)

        chart = Image.fromarray(chart_image).resize((iw, ih))
        img_chart = ImageTk.PhotoImage(image=chart)

        original = image.resize((iw, ih))
        img_original = ImageTk.PhotoImage(image=original)
        
        detection = Image.fromarray(baw_detection).resize((iw, ih))
        img_detection = ImageTk.PhotoImage(image=detection)

        segmetation = Image.fromarray(segmetation_image).resize((iw, ih))
        img_segmetation = ImageTk.PhotoImage(image=segmetation)
        if self._framet2:
            self._framet2.destroy()
        self._framet2 = cstk.CTkFrame(self.tab2)
        self._framet2.pack()
        b2 =tk.Button(self._framet2,image=img_original) 
        b2.grid(row=1,column=1)
        b3 =tk.Button(self._framet2,image= img_segmetation)
        b3.grid(row=1,column=2)
        b4 =tk.Button(self._framet2,image=img_detection)
        b4.grid(row=1,column=3)
        b1 =tk.Button(self._framet2,image=img_chart) 
        b1.grid(row=2,column=1)  
        tab_w1 = int(self.winfo_screenwidth()/2.4)
        tab_w2 = int(self.winfo_screenwidth()/6.4)
        columns = ('ZA', 'NA')
        tree = ttk.Treeview(self._framet2, columns=columns, show='headings', style="mi_estilo.Treeview", height=7)
        tree.grid(row=2,column=2,columnspan=2)
        tree.column("# 1",anchor=CENTER, stretch=NO, width=tab_w1)
        tree.heading('ZA', text='Sección Afectada')
        tree.column("# 2",anchor=CENTER, stretch=NO, width=tab_w2)
        tree.heading('NA', text='Nivel')
        tuples = self._eye.results()
        index = iid = 0
        for row in tuples:
            tree.insert("", index, iid, values=row)
            index = iid = index + 1
        self.mainloop()  

    def button_clicked(self):
        showinfo(title='Information', message='Hello, Tkinter!')