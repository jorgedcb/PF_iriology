import numpy as np
import cv2


class Iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_path = image_path
        self._pupil = None

    def load_image(self):

        self._img = cv2.imread(self._img_path)
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def detect_pupil(self):
        _, thresh = cv2.threshold(self._img, 100, 255, cv2.THRESH_BINARY)
        img_with_contours = np.copy(self._img)
        print(self._img.shape[0] / 2)
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._img.shape[0] / 2, maxRadius=60)
        for l in c:
            for circle in l:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
                self._pupil = (center[0], center[1], radius)

    def detect_iris(self):
        _, t = cv2.threshold(self._img, 195, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_with_contours = np.copy(self._img)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0))
        c = cv2.HoughCircles(img_with_contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, maxRadius=170, minRadius=80)
        for l in c:
            for circle in l:
                center = (self._pupil[0], self._pupil[1])
                radius = int(circle[2])
                mask = np.zeros((self._img.shape[0], self._img.shape[1], 1), np.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
                self._img = cv2.bitwise_and(self._img, mask)

    def start_detection(self):
  
        if(self.load_image()):
            self.convert_to_gray_scale()
            self.detect_pupil()
            self.detect_iris()
            #cv2.imshow("Result", self._img)
            #cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')
        return self._img
            
        
#id = Iris_detection('058_2_1.jpg')
#id.start_detection()