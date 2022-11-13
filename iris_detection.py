import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import math
from models import Circle, Zone

class IrisDetection():
    def __init__(self, image_path, percentil:int):
        if  100 < percentil < 0: 
            raise ValueError("invalid argument!") 
        self._img_path = image_path
        self.pupil = None
        self.iris = None

        self._percentil = percentil 
        self.original_image = None
        self.work_image = None
        self._anomalies_thresh = None
        self._countours_anomalies = []
        self._found_diseases = []
        self._circles = []
        self._zones = []
        self._gray_scale_segmentate_image = None
        self._gray_scale_diagnostic_image = None
        self._color_diagnostic_image = None
        self._color_image = None


    @staticmethod 
    def write_image(filename, image):
        cv2.imwrite(r'images\{}.png'.format(filename), image)

    def load_image(self):
        self.original_image = cv2.imread(self._img_path)
        if type(self.original_image) is type(None):
            return False
        else:
            return True

    @staticmethod
    def _convert_to_gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _blur_image(image):
        return cv2.GaussianBlur(image,(9,9), cv2.BORDER_DEFAULT)

    @staticmethod
    def _pupil_tresh(img):
        data = img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        std = np.std(data_f)
        mean = np.mean(data_f)
        return mean - std
    
    @staticmethod
    def _pupil_tresh_alternative(img):
        data = img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        return np.percentile(data_f, 12)

    def _detect_pupil(self):
        img = self.work_image.copy()
        thresh = self._pupil_tresh_alternative(img)
        _, t = cv2.threshold(self.work_image, thresh, 255, cv2.THRESH_BINARY) 
        self.write_image("4.1 binarización para encontrar contornos de la pupila",t)
        contours, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = np.copy(self.work_image)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        self.write_image("4.2 a la imagen del paso 3 se le dibujan los contornos para detectar el circulo de la pupila",img_with_contours)
        circles_founds = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, 400, maxRadius=100)
        if(circles_founds.shape != (1,1,3)):
            raise ValueError("More than 1 circle found for the pupil")
        pupil = circles_founds[0][0]
        center = (int(pupil[0]), int(pupil[1]))
        radius = int(pupil[2])
        self.pupil = Circle(radius, center)

    @staticmethod
    def _iris_tresh(img):
        data = img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        mean = np.mean(data_f)
        std = np.std(data_f)
        return mean + (0.4*std)

    @staticmethod
    def _iris_tresh_alternative(img):
        data = img.ravel()
        data_f = np.delete(data, np.where(data == 0))
        return np.percentile(data_f, 70)
   
    def _detect_iris(self):
        img = self.work_image.copy()
        thresh = self._iris_tresh_alternative(img)
        _, t = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        self.write_image("5.1 binarización para encontrar contornos del iris",t)
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = np.copy(self.work_image)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        self.write_image("5.2 a la imagen del paso 3 se le dibujan los contornos para detectar el circulo del iris",img_with_contours)
        circles_founds = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self.pupil.radius * 3, maxRadius=280, minRadius=200)
        if(circles_founds.shape != (1,1,3)):
            raise ValueError("More than 1 circle found for the iris")
        iris = circles_founds[0][0]
        center = (int(iris[0]), int(iris[1]))
        radius = int(iris[2])
        self.iris = Circle(radius, center)

    def _find_anomalies_thresh(self):
        data = self.work_image.ravel()
        data_f = np.delete(data, np.where(data == 0))
        self._anomalies_thresh = np.percentile(data_f, self._percentil)

    def _detect_anomalies(self):
        self._find_anomalies_thresh()
        _, t = cv2.threshold(self.work_image, self._anomalies_thresh, 255, cv2.THRESH_BINARY)
        self.write_image("6.1 binarización para hallar anomalias" ,t)
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if  area < 5000: 
                contour_list.append(contour)
        self._countours_anomalies = contour_list

    
    def _find_circles(self):
        percentages = [0.1333 , 0.3222 , 0.4888 , 0.6888 , 0.8666,0.9555]
        radii_differences = self.iris.radius - self.pupil.radius
        raddi = [int(radii_differences*x+self.pupil.radius) for x in percentages]
        centers_differences = tuple(map(lambda i, j: i - j, self.iris.center, self.pupil.center))
        number_circles = len(raddi)
        for i in range(number_circles):
            diff = tuple((coordenate/number_circles)*(i+1) for coordenate in centers_differences)
            center = tuple(map(lambda i, j: int(i + j), self.pupil.center, diff))
            self._circles.append(Circle(raddi[i],center))
        

    def _find_zones(self):
        zones_names  = ["Estomago" , "Intestino" , "Páncreas, Riñones y Corazón" , "Órganos respiratorios" ,"Cerebro y órganos reproductivos","Hígado, Bazo, Tiroides, Linfáticos y Glándulas Pequeñas", "Piel, Músculos, Nervios motores y sensoriales"]
        circles_zones = [self.pupil] + self._circles + [self.iris]
        for i in range(len(zones_names)):
            self._zones.append(Zone(circles_zones[i],circles_zones[i+1],zones_names[i]))

    def _fusifier(self):
        anomalies = self._countours_anomalies
        img = self._convert_to_gray_scale(self.original_image)
        for i in range(len(anomalies)):
            cimg = np.zeros_like(img)
            cv2.drawContours(cimg, anomalies, i, color=255, thickness=-1)
            pts = np.where(cimg == 255)
            intensity = 100 - ((img[pts[0], pts[1]].mean())*(100/255))
            affected_areas = []          
            for zone in self._zones:
                distances_center_inner_circle = [math.dist(x[0], zone.center_inner_circle) for x in anomalies[i]]
                distances_center_outter_circle = [math.dist(x[0], zone.center_outter_circle) for x in anomalies[i]]
                if any((inner_distance >= zone.inner_radius and outter_distance <= zone._outter_radius) for inner_distance, outter_distance in zip(distances_center_inner_circle,distances_center_outter_circle)):
                    affected_areas.append(zone.name)
                    
            if (not affected_areas):
                raise ValueError("Error founding affected_areas")
            self._found_diseases.append([affected_areas,intensity])

    @staticmethod
    def _get_disease_severity(intensity,mean,std):

        if intensity <= mean-std:
            disease_severity = "Agudo"
        elif mean-std <= intensity <= mean:
            disease_severity = "Sub-Agudo "
        elif mean <= intensity <= mean+std:
            disease_severity = "Cronico"
        elif mean+std <= intensity:
            disease_severity = "Degenerativo"
        else:
            disease_severity = "ERROR"
        return disease_severity


    def results(self):
       
        dict_results = {}
        for disease in self._found_diseases:
            for zone in disease[0]:
                if zone in dict_results:
                    dict_results[zone].append(disease[1])
                else:
                    dict_results[zone] = [disease[1]]
        
        
        list_percentages= ([max(dict_results[key]) for key in dict_results])
        std = np.std(list_percentages, ddof=1)
        mean = np.mean(list_percentages)
        return [(key, self._get_disease_severity(max(dict_results[key]),mean,std)) for key in dict_results]

    def _draw_all_circles(self, image):
        for circle in self._circles:
            cv2.circle(image, circle.center , circle.radius, (0, 0, 255), thickness = 1)

    @staticmethod
    def _draw_countours(image, contour_list):
        cv2.drawContours(image, contour_list, -1, (255, 0, 0))
        

    @staticmethod
    def _draw_circle_perimeter(image, circle):
        cv2.circle(image, circle.center , circle.radius, (0, 0, 255), thickness = 1)
    
    @staticmethod
    def _segmentate_circle_inside(image, circle):
        cv2.circle(image, circle.center , circle.radius, (0, 0, 0), thickness = -1)

    @staticmethod
    def _segmentate_circle_outside(image, circle):
        mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv2.circle(mask, circle.center, circle.radius, (255, 255, 255), thickness = -1)
        return cv2.bitwise_and(image, mask)

    @staticmethod
    def _segmentate_circle_outside_color(image, circle):
        mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv2.circle(mask, circle.center, circle.radius, (255, 255, 255), thickness = -1)
        return cv2.bitwise_and(image,image, mask=mask)   

    @staticmethod
    def _draw_point(image, circle):
        cv2.circle(image, circle.center , 2, (0, 0, 255), thickness = 1)
    
    def run_diagnostic(self):

        if self.load_image():
            self.write_image("1. original",self.original_image)
            self.work_image = self._convert_to_gray_scale(self.original_image)
            self.write_image("2. blanco y negro",self.work_image)
            self.work_image = self._blur_image(self.work_image)
            self.write_image("3. difuminada",self.work_image)
            self._detect_pupil()
            self._segmentate_circle_inside(self.work_image, self.pupil)
            self.write_image("4.3 imagen con pupila detectada" ,self.work_image)
            self._detect_iris()
            self.work_image = self._segmentate_circle_outside(self.work_image,self.iris)
            self.write_image("5.3 imagen con iris detectado" ,self.work_image)
            self._detect_anomalies()
            self._draw_countours(self.work_image, self._countours_anomalies)
            self.write_image("6.2 a la imagen 5.3 le dubijo las anomalias encontradas" ,self.work_image)
            self._find_circles()
            self._draw_all_circles(self.work_image)
            self.write_image("7. dibujo los circulos" ,self.work_image)
            self._find_zones()
            self._fusifier()
            self._set_gray_scale_diagnostic_image()
            self._set_gray_scale_segmentate_image()
            self._set_color_diagnostic_image()
            self._set_color_image()
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')
            
    
    def _set_gray_scale_segmentate_image(self):
        img = self._convert_to_gray_scale(self.original_image)
        img = self._blur_image(img)
        self._segmentate_circle_inside(img, self.pupil)
        img = self._segmentate_circle_outside(img,self.iris)
        self._gray_scale_segmentate_image = img
    
    @property
    def gray_scale_segmentate_image(self):
        return self._gray_scale_segmentate_image

    def _set_gray_scale_diagnostic_image(self):
        self._gray_scale_diagnostic_image = self.work_image

    @property
    def gray_scale_diagnostic_image(self):
        return self._gray_scale_diagnostic_image

    def _set_color_diagnostic_image(self):
        img = self.original_image.copy()
        self._segmentate_circle_inside(img, self.pupil)
        img = self._segmentate_circle_outside_color(img,self.iris)
        self._draw_countours(img,self._countours_anomalies)
        self._draw_all_circles(img)
        self._color_diagnostic_image = img

    @property
    def color_diagnostic_image(self):
        return self._color_diagnostic_image

    def _set_color_image(self):
        img = self.original_image.copy()
        self._draw_circle_perimeter(img, self.pupil)
        self._draw_circle_perimeter(img,self.iris)
        self._draw_countours(img,self._countours_anomalies)
        self._draw_all_circles(img)
        self._color_image = img
    
    @property
    def color_image(self):
        return self._color_image
    
    @staticmethod
    def histogram(image):
        data_f = image.ravel() 
        data_f = np.delete(data_f, np.where(data_f == 0))
        sns.set(style="darkgrid")
        f, (ax_box, ax_hist) = plt.subplots(2,figsize=(12,3), sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data_f, ax=ax_box, orient='h')
        sns.histplot(data=data_f, binwidth=1, ax=ax_hist)
        ax_box.set(xlabel='')
        plt.show()

    @staticmethod
    def information(image):
        data = image.ravel() 
        data_f = np.delete(data, np.where(data == 0))
        mean = np.average(data_f)
        p5,p10,q1,q2,q3= np.percentile(data_f, [5,10,25,50,75])
        iqr = q3 - q1
        threshold = q1-1.5*iqr
        print("mean:", mean)
        print("median:", q2)
        print("p5:", p5)
        print("p10:", p10)
        print("threshold atypical values",threshold)

        
if __name__ == "__main__":           
    id = IrisDetection(r'Experimentos\019L_3.png',5)
    id.run_diagnostic()
    results = id.results()
    print(results)
    cv2.imshow("result",id.color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()