import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats as st
#from tkinter import ttk
#from tkinter.messagebox import showinfo
#import tkinter as tk                    
from tkinter import filedialog
#from tkinter.filedialog import askopenfile
#from PIL import Image, ImageTk


class Iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_path = image_path
        self._pupil = None
        self._iris = None
        self._imgbaw = None

    def load_image(self):

        self._img = cv2.imread(self._img_path)
        #self._img = cv2.imread(self._img_path,1) #read in gray scale
        self._original = self._img.copy()
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale(self):
        #self._img =  cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        imagen = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('original.jpg', imagen)
        self._imgbaw = imagen.copy()
        self._img  = cv2.GaussianBlur(imagen,(9,9), cv2.BORDER_DEFAULT)
        #cv2.imwrite('opencv_th_tz.jpg', t)
        #self._img  = cv2.GaussianBlur(imagen,(3,3), cv2.BORDER_DEFAULT)

        #plt.rcParams["figure.figsize"]=(16,9)
        #plt.imshow(imagen,cmap='gray')

        #cv2.imshow("Result",imagen)
        #cv2.waitKey(0)

    def detect_pupil(self):

        _, thresh = cv2.threshold(self._img, 70, 255, cv2.THRESH_BINARY) #70
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite('iris_tresh.jpg', thresh)
        img_with_contours = np.copy(self._img)
        
        #cv2.drawContours(self._original, contours, -2, (0, 255, 0))
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        #cv2.imwrite('iris_tresh.jpg', img_with_contours)
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, 400, maxRadius=100) #80
        len_c = len(c)
        for l in c:
            if len_c !=1 | len(l) !=1:
                print('More than 1 circle found for the pupil')
            for circle in l:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                #print('wtf1',center)
                cv2.circle(self._original, center, radius, (0, 0, 255), thickness = 1)
                cv2.circle(self._original, center, 3, (255, 0, 0), thickness = -1)
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
                cv2.circle(self._imgbaw, center, radius, (0, 0, 0), thickness=-1)
                self._pupil = (center[0], center[1], radius)

    def detect_iris(self):
        _, t = cv2.threshold(self._img, 125, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print('jorge',len(contours))
        #cv2.imwrite('opencv_th_tz.jpg', t)
        img_with_contours = np.copy(self._img)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        cv2.imwrite('opencv_th_tz.jpg', img_with_contours)
        
    
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 3, maxRadius=280, minRadius=200)
        len_c = len(c)
        for l in c:
            if len_c !=1 | len(l) !=1:
                print('More than 1 circle found for the iris')
            for circle in l:
                
                center = (int(circle[0]), int(circle[1]))
                #center = (self._pupil[0], self._pupil[1])
                radius = int(circle[2])
                
                cv2.circle(self._original, center, radius, (0, 0, 255), thickness = 1)
                cv2.circle(self._original, center, 3, (0, 0, 255), thickness = -1)
                self._iris = (center[0], center[1], radius)
                mask = np.zeros((self._img.shape[0], self._img.shape[1], 1), np.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
                self._img = cv2.bitwise_and(self._img, mask)
                self._imgbaw = cv2.bitwise_and(self._imgbaw, mask)
                
                

    def detect_anomalies(self):

        _, thresh = cv2.threshold(self._img,85 , 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite('anomalies.jpg', thresh)

        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) #this is to find circles, dont need it right know
            area = cv2.contourArea(contour)
            #print('area',len(approx),area)
            if  area < 1000: #avoid find countours to big maybe y have to increase this value
                #print('area',len(approx),area)
                contour_list.append(contour)
            # else:
            #     #contour_list.append(contour)
            #     pass

        cv2.drawContours(self._original, contour_list, -2, (255, 0, 0))
        
    def histogram(self):
        #print('size',self._img.shape)

        data_f = self._img.ravel()
        data_f = np.delete(data_f, np.where(data_f == 0))
        sns.set(style="darkgrid")
        f, (ax_box, ax_hist) = plt.subplots(2,figsize=(12,3), sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data_f, ax=ax_box, orient='h')
        sns.histplot(data=data_f, binwidth=1, ax=ax_hist)
        ax_box.set(xlabel='')
        #ax_box.set_title("Distribución longitud de las palabras",fontsize=15)
        #plt.ylabel('Numero de palabras', fontsize=13)
        #plt.xlabel('Longitud de las palabras', fontsize=13)
        plt.show()

    def information(self):
        print('information')
        
        data = self._img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        mean = np.average(data_f)
        #mode = statistics.mode(data_f)
        p5,p10,q1,q2,q3,p90= np.percentile(data_f, [5,10,25,50,75,90])
        iqr = q3 - q1
        threshold = q1-1.5*iqr
        max_value_words = max(data_f)
        min_value_words = min(data_f)
        var_value = np.var(data_f)
        std_value = np.std(data_f)
        print("mean:", mean)
        print("median:", q2)
        #print("mode:", mode)
        #print("q1:", q1)
        #print("q3:", q3)
        #print("p90:", p90)
        print("p5:", p5)
        print("p10:", p10)
        #print("iqr:", iqr)
        print("threshold atypical values",threshold)
        #print("max:", max_value_words)
        #print("min:", min_value_words)
        #print("range:", max_value_words-min_value_words)
        #print("var:", var_value)
        #print("std:",std_value)

    def draw_circules(self):
        radius_pupil = self._pupil[2]
        radius_iris = self._iris[2]
        diff = radius_iris - radius_pupil
        percentages = [0.1333 , 0.3222 , 0.4888 , 0.6888 , 0.8666,0.9555]
        raddi = [int(diff*x+radius_pupil) for x in percentages]
        #print(raddi)
        #center = (self._pupil[0], self._pupil[1])
        center = (self._iris[0], self._iris[1])
        for radius in raddi:
            cv2.circle(self._original, center, radius, (0, 255, 0), thickness = 1)
    
    def start_detection(self):
        if(self.load_image()):
            self.convert_to_gray_scale()
            self.detect_pupil()
            self.detect_iris()
            self.detect_anomalies()
            self.information()
            self.histogram()
            self.draw_circules()
            #imS = cv2.resize(self._img, (1280,720))   
            #cv2.imshow("Result",imS)

            cv2.imshow("Result",self._original)
            #cv2.imshow("Result",self._img)
            #cv2.imshow("Result",self._imgbaw)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')
        #cv2.destroyAllWindows()
        return self._img
        
            
#f_types = [('Jpg Files', '*.jpg')]
#f_types = [('Jpg Files', '*.jpg','Png Files', '*.png')]
# f_types = [('Png Files', '*.png')]
# filename = filedialog.askopenfilename(filetypes=f_types)
# id = Iris_detection(filename) 

#id = Iris_detection('004R_3.png')
id = Iris_detection('013L_1.png')

id.start_detection()