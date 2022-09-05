import numpy as np
import cv2

from tkinter import ttk
from tkinter.messagebox import showinfo
import tkinter as tk                    
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import statistics

class Iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_path = image_path
        self._pupil = None
        self._iris_radius = None

    def load_image(self):

        self._img = cv2.imread(self._img_path)
        self._original = self._img
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def detect_pupil(self):
        _, thresh = cv2.threshold(self._img, 90, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imwrite('opencv_th_tz.jpg', thresh)
        print(len(contours))
        img_with_contours = np.copy(self._img)
        #cv2.drawContours(self._original, contours, -2, (0, 255, 0))
        cv2.drawContours(img_with_contours, contours, -2, (0, 255, 0))
        # # cv2.imshow("Result", self._original)
        # # cv2.waitKey(0)
        # #print(self._img.shape[0] / 2)
        # contour_list = []
        # for contour in contours:
        #     approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        #     area = cv2.contourArea(contour)
        #     if ((len(approx) > 8) & (area > 4000)  & (area < 6000) ):
        #         print('lol',len(approx),area)
        #         contour_list.append(contour)
        # cv2.drawContours(self._original, contour_list, -1, (0, 255, 0))
        # cv2.imshow("jorge", self._original)
        # cv2.waitKey(0)
        # # print('wtf',len(contour_list))
        # cv2.drawContours(img_with_contours, contour_list, -1, (0, 255, 0))
        # #print(self._img.shape)
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, 200, maxRadius=60)

        for l in c:
            for circle in l:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                print('wtf',radius)
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
                #cv2.circle(self._original, center, radius, (0, 255, 0), thickness=1) #activar
                # cv2.imshow("pupil",self._img)
                # cv2.waitKey(0)
                self._pupil = (center[0], center[1], radius)

    def detect_iris(self):
        _, t = cv2.threshold(self._img, 195, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite('opencv_th_tz.jpg', t)
        img_with_contours = np.copy(self._img)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        
    
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, maxRadius=170, minRadius=80)
        print('len1',len(c))
        for l in c:
            print('len',len(l))
            for circle in l:
                print('circle',circle)
                center = (self._pupil[0], self._pupil[1])
                radius = int(circle[2])
                self._iris_radius = radius
                mask = np.zeros((self._img.shape[0], self._img.shape[1], 1), np.uint8)
                print('lol',radius)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
                #cv2.circle(self._original, center, radius, (255, 255, 255), thickness = 1)
                self._img = cv2.bitwise_and(self._img, mask)
                break

    def detect_anomalies(self):

        _, thresh = cv2.threshold(self._img, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite('opencv_th_tz.jpg', thresh)

        contour_list = []
        #print('lenjorge',len(contours))
        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            #print('lol1',len(approx),area)
            if ((len(approx) < 11) & (area < 100) ):
                #print('lol2',len(approx),area)
                contour_list.append(contour)

        #cv2.imwrite('opencv_th_tz.jpg', thresh)
        #print(len(contours))
        img_with_contours = np.copy(self._img)
        #cv2.drawContours(self._original, contours, -2, (0, 255, 0))
        cv2.drawContours(self._original, contour_list, -2, (255, 0, 0))
        # _, t = cv2.threshold(self._img, 0, 10, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # img_with_contours = np.copy(self._img)
        # cv2.drawContours(self._original, contours, -1, (0, 255, 0))
        
    
        # # c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, maxRadius=170, minRadius=80)
        # # for l in c:
        # #     for circle in l:
        # #         center = (self._pupil[0], self._pupil[1])
        # #         radius = int(circle[2])
        # #         mask = np.zeros((self._img.shape[0], self._img.shape[1], 1), np.uint8)
        # #         cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
        # #         self._img = cv2.bitwise_and(self._img, mask)
    
    def histogram(self):
        # print(type(self._img))
        print('size',self._img.shape)

        imgpath = "test.tiff"
        img = cv2.imread(imgpath,0)
        data_f = self._img.ravel()
        data_f = np.delete(data_f, np.where(data_f == 0))
        plt.subplot(1,2,1)
        #plt.imshow(img,cmap='gray')
        plt.imshow(self._img)
        plt.title('image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1,2,2)
        plt.hist(data_f,256,[0,255])
        plt.title('histogram')

        plt.show()

    def information(self):
        print('information')
        
        data = self._img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        mean = statistics.mean(data_f)
        averague = np.average(data_f)
        median = statistics.median(data_f)
        mode = statistics.mode(data_f)
        p5,p10,q1,q2,q3,p90= np.percentile(data_f, [5,10,25,50,75,90])
        iqr = q3 - q1
        threshold = q1-1.5*iqr
        max_value_words = max(data_f)
        min_value_words = min(data_f)
        var_value = np.var(data_f)
        std_value = np.std(data_f)
        print("mean:", mean)
        print("averague:", averague)
        print("median:", median)
        print("q2:", q2)
        print("mode:", mode)
        print("q1:", q1)
        print("q3:", q3)
        print("p90:", p90)
        print("p5:", p5)
        print("p10:", p10)
        print("iqr:", iqr)
        print("threshold atypical values",threshold)
        print("max:", max_value_words)
        print("min:", min_value_words)
        print("range:", max_value_words-min_value_words)
        print("var:", var_value)
        print("std:",std_value)

    def draw_circules(self):
        radius_pupil = self._pupil[2]
        radius_iris = self._iris_radius
        print('pupil ',radius_pupil )
        print('iris ',radius_iris )
        diff = radius_iris - radius_pupil
        percentages = [0.01333 , 0.3222 , 0.4888 , 0.6888 , 0.8666,0.9555,1]
        raddi = [int(diff*x+radius_pupil) for x in percentages]
        
        print(raddi)
        center = (self._pupil[0], self._pupil[1])
        for radius in raddi:
            cv2.circle(self._original, center, radius, (0, 255, 0), thickness = 1)
    def start_detection(self):
  
        if(self.load_image()):
            self.convert_to_gray_scale()
            self.detect_pupil()
            self.detect_iris()
            self.detect_anomalies()
            self.information()
            #self.histogram()
            self.draw_circules()
            #imS = cv2.resize(self._img, (1280,720))   
            #cv2.imshow("Result",imS)
            cv2.imshow("Result",self._original)
            #cv2.imshow("Result",self._img)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')
        #cv2.destroyAllWindows()
        return self._img
        
            
# f_types = [('Jpg Files', '*.jpg')]
# filename = filedialog.askopenfilename(filetypes=f_types)
# id = Iris_detection(filename) 

id = Iris_detection('S1202R03.jpg')

id.start_detection()