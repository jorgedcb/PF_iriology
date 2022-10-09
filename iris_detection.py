import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog
import math


#from scipy import stats as st
#from tkinter import ttk
#from tkinter.messagebox import showinfo
#import tkinter as tk                    
#from tkinter.filedialog import askopenfile
#from PIL import Image, ImageTk

class circle():
    def __init__(self, name, center, radious):
        self.name = name
        self.radius = radious

class IrisDetection():
    def __init__(self, image_path):
        self._img = None
        self._img_path = image_path
        self._pupil = None
        self._iris = None
        self._img_segmatation = None
        self._original = None
        self._baw_detection = None
        self._color_detection = None
        self._circles = []
        


    def load_image(self):

        self._img = cv2.imread(self._img_path)
        #self._img = cv2.imread(self._img_path,1) #read in gray scale
        self._original = self._img.copy()
        self._color_detection = self._img.copy()
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale_no_blur(self):
        self._img =  cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        
    def convert_to_gray_scale(self):
        imagen = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        self._img_segmatation = imagen.copy()
        self._img  = cv2.GaussianBlur(imagen,(9,9), cv2.BORDER_DEFAULT) #estudiar como funciona esto

        #cv2.imwrite(r'images\original.jpg', imagen)
        #cv2.imwrite(r'images\opencv_th_tz.jpg', t)

        #cv2.imshow("Result",imagen)
        #cv2.waitKey(0)

    def detect_pupil(self):

        _, thresh = cv2.threshold(self._img, 70, 255, cv2.THRESH_BINARY) #70
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite(r'images\iris_tresh.jpg', thresh)
        img_with_contours = np.copy(self._img)
        
        #cv2.drawContours(self._original, contours, -2, (0, 255, 0))
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        #cv2.imwrite(r'images\iris_tresh.jpg', img_with_contours)
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, 400, maxRadius=100) #80
        len_c = len(c)
        for l in c:
            if len_c !=1 | len(l) !=1:
                print('More than 1 circle found for the pupil')
            for circle in l:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2]) 
                cv2.circle(self._color_detection, center, radius, (0, 0, 255), thickness = 1) # solo dibujar circulo
                cv2.circle(self._color_detection, center, 3, (255, 0, 0), thickness = -1)
                cv2.circle(self._original, center, radius, (0, 0, 0), thickness = -1)
                cv2.circle(self._original, center, 3, (255, 0, 0), thickness = -1)
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
                cv2.circle(self._img_segmatation, center, radius, (0, 0, 0), thickness=-1)
                self._pupil = (center[0], center[1], radius)

    def detect_iris(self):
        _, t = cv2.threshold(self._img, 125, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = np.copy(self._img)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        cv2.imwrite(r'images\opencv_th_tz.jpg', img_with_contours)
        
    
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 3, maxRadius=280, minRadius=200)
        len_c = len(c)
        for l in c:
            if len_c !=1 | len(l) !=1:
                print('More than 1 circle found for the iris')
            for circle in l:
                
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                
                cv2.circle(self._color_detection, center, radius, (0, 0, 255), thickness = 1)
                cv2.circle(self._color_detection, center, 3, (0, 0, 255), thickness = -1) 
                cv2.circle(self._original, center, 3, (0, 0, 255), thickness = -1)
                
                self._iris = (center[0], center[1], radius)
                #print((self._img.shape[0], self._img.shape[1], 1))
                #print((self._original.shape[0], self._original.shape[1], 1))
                mask = np.zeros((self._img.shape[0], self._img.shape[1], 1), np.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
                self._original = cv2.bitwise_and(self._original,self._original, mask=mask)
                self._img = cv2.bitwise_and(self._img, mask)
                self._img_segmatation = cv2.bitwise_and(self._img_segmatation, mask)
                
                

    def detect_anomalies(self):
        
        _, thresh = cv2.threshold(self._img,self._p5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite(r'images\anomalies.jpg', thresh)
        contour_list = []
        
        
        for contour in contours:
            # print("_____________________________________")
            # print(cv2.mean(contour))
            # print("contour")
            # print(contour)
           
            #x = input()
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) #this is to find circles, dont need it right know
            area = cv2.contourArea(contour)
            #mean_val = cv2.mean(im,mask = mask)
            #print('area',len(approx),area)
            if  area < 300: #avoid find countours to big maybe y have to increase this value
                #print('area',len(approx),area)
                contour_list.append(contour)
            # else:
            #     #contour_list.append(contour)
            #     pass
        #print("contour_list",contour_list[0])
        self._countour_list = contour_list
        self._baw_detection = self._img_segmatation.copy()
        cv2.drawContours(self._baw_detection, contour_list, -2, (255, 0, 0))
        cv2.drawContours(self._original, contour_list, -2, (255, 0, 0))
        cv2.drawContours(self._color_detection, contour_list, -2, (255, 0, 0))
        #print("_baw_detection",self._baw_detection[220, 245])
        #print("_color_detection",self._color_detection[220, 245])
        cv2.circle(self._original, (220, 245), 3, (0, 0, 255), thickness = -1)

    def fusifier(self):
        contour_list = self._countour_list
        #print(contour_list)
        #print(self._circles)
        # Initialize empty list
        lst_intensities = []
        img = self._img_segmatation.copy()
        jorge = img.copy()

        #cv2.imwrite(r'images\anomalies.jpg', img)
        #x = input()
        # For each list of contour points...
        for i in range(len(contour_list)):
            cimg = np.zeros_like(img)
            cv2.drawContours(cimg, contour_list, i, color=255, thickness=-1)
            cv2.drawContours(jorge, contour_list, i, color=255, thickness=-1)
            cv2.imwrite(r'images\anomalies.jpg', jorge)
            pts = np.where(cimg == 255)
            intensity = 100 - ((img[pts[0], pts[1]].mean())*(100/255))
            names = []
            countour = contour_list[i]
            #print(countour)
            min_distance = 0
            print(len(self._circles))
            for circle in self._circles:
                distances = [math.dist(x[0], circle[0]) for x in countour]
                print(distances)
                if any((x >= min_distance and x < circle[1]) for x in distances):
                    names.append(circle[2])
                min_distance = circle[1]
            
            if 0 <= intensity <= 25:
                organ_name = "Agudo"
            elif 25 <= intensity <= 50:
                organ_name = "Sub-Agudo "
            elif 50 <= intensity <= 75:
                organ_name = "Cronico"
            elif 75 <= intensity <= 100:
                organ_name = "Degenerativo"

            anomalie =  [names,intensity,organ_name]
            lst_intensities.append(anomalie)
            print("--------------------------")
            #print("contour_list",contour_list[i])
            print(anomalie)
            #x = input()
            
            # x = input()
        #print("--------------------------")
        #print(lst_intensities)


    def draw_circules(self):
        radius_pupil = self._pupil[2]
        radius_iris = self._iris[2]
        diff = radius_iris - radius_pupil
        percentages = [0.1333 , 0.3222 , 0.4888 , 0.6888 , 0.8666,0.9555]
        names  = ["Estomago" , "Intestino" , "Corazon, bronquios y páncreas" , "Esqueleto, útero y Prostata" ,"Cerebro, pulmones, higado, bazo, riñones","Músculos, sistema nervioso, linfático y circulatorio", "Piel y nervios sensoriales"]
        raddi = [int(diff*x+radius_pupil) for x in percentages]
        center_pupil = (self._pupil[0], self._pupil[1])
        center_iris = (self._iris[0], self._iris[1])
        differe = tuple(map(lambda i, j: i - j, center_iris, center_pupil))
        # print("center_pupil",center_pupil)
        # print("center_iris",center_iris)
        # print("different",differe)
        #self._baw_detection = self._img_segmatation.copy()
        number_circles = len(raddi)
        
        for i in range(number_circles):

            diff = tuple((ti/number_circles)*(i+1) for ti in differe)
            #print("different",diff)
            center = tuple(map(lambda i, j: int(i + j), center_pupil, diff))
            self._circles.append([center,raddi[i],names[i]])
            #print("center",center)
            cv2.circle(self._color_detection, center, raddi[i], (0, 255, 0), thickness = 1)
            cv2.circle(self._baw_detection, center, raddi[i], (0, 255, 0), thickness = 1)
            cv2.circle(self._original, center, raddi[i], (0, 255, 0), thickness = 1)
        
    def histogram(self):
        
        #data_f = self._img.ravel()
        data_f = self._img_segmatation.ravel() # revisar diferencia entre _img y _img_segmatation
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
        
        data = self._img_segmatation.ravel() 
        data_f = np.delete(data, np.where(data == 0))
        mean = np.average(data_f)
        p5,p10,q1,q2,q3= np.percentile(data_f, [5,10,25,50,75])
        self._p5 = p5
        iqr = q3 - q1
        threshold = q1-1.5*iqr
        
        print("mean:", mean)
        print("median:", q2)
        print("p5:", p5)
        print("p10:", p10)
        print("threshold atypical values",threshold)
        #print("iqr:", iqr)

        # max_value_words = max(data_f)
        # min_value_words = min(data_f)
        # var_value = np.var(data_f)
        # std_value = np.std(data_f)
        #print("max:", max_value_words)
        #print("min:", min_value_words)
        #print("range:", max_value_words-min_value_words)
        #print("var:", var_value)
        #print("std:",std_value)

    
    def start_detection(self):
        if(self.load_image()):
            self.convert_to_gray_scale()
            self.detect_pupil()
            self.detect_iris()
            self.information()
            self.detect_anomalies()
            
           
            #self.histogram()
            self.draw_circules()
            self.fusifier()
            #imS = cv2.resize(self._img, (1280,720))   
            #cv2.imshow("Result",imS)

            #cv2.imshow("Result",self._original) 
            #cv2.imshow("Result",self._img)
            #cv2.imshow("Result",self._img_segmatation)
            #cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')
        #cv2.destroyAllWindows()
        #return self._img
        return self._original, self._img_segmatation, self._baw_detection, self._color_detection
        
if __name__ == "__main__":           
    #f_types = [('Jpg Files', '*.jpg')]
    #f_types = [('Jpg Files', '*.jpg','Png Files', '*.png')]
    # f_types = [('Png Files', '*.png')]
    # filename = filedialog.askopenfilename(filetypes=f_types)
    # id = IrisDetection(filename) 

    id = IrisDetection(r'images\004R_3.png')

    #id = IrisDetection(r'images\013L_1.png')
    original_image, segmetation_image, baw_detection, color_detection = id.start_detection()
    cv2.imshow("Result",original_image) 
    cv2.waitKey(0)