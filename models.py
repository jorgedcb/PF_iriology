#validate if type is int of float
from unicodedata import name


class Circle:
    def __init__(self, radius : int , center : tuple) -> None:
        self._radius = radius
        self._center = center
        self._center_x_coordinate = center[0]
        self._center_y_coordinate = center[1]

    @property
    def radius(self):
        return self._radius
        
    @property
    def center(self):
        return self._center

    @property
    def center_x_coordinate(self):
        return self.center_x_coordinate

    @property
    def center_y_coordinate(self):
        return self._center_y_coordinate


class Zone:
    def __init__(self, inner_circle : Circle, outter_circle : Circle, name : str) -> None:
        self.different_between_radius = outter_circle.radius - inner_circle.radius #deberia dar error
        self._inner_radius = inner_circle.radius
        self._outter_radius = outter_circle.radius
        self._center_inner_circle = inner_circle.center
        self._center_inner_circle_x_coordinate = inner_circle.center[0]
        self._center_inner_circle_y_coordinate = inner_circle.center[1]
        self._center_outter_circle = outter_circle.center
        self._center_outter_circle_x_coordinate = outter_circle.center[0]
        self._center_outter_circle_y_coordinate = outter_circle.center[1]
        self._name = name

    @property
    def different_between_radius(self):
        return self._different_between_radius
         
    @different_between_radius.setter
    def different_between_radius(self, value):
        if(value < 0):
            raise ValueError("outter_radius must be greater than inner_radius")
        self._different_between_radius = value

    @property
    def inner_radius(self):
        return self._inner_radius
    
    @property
    def outter_radius(self):
        return self._outter_radius
    
    @property
    def center_inner_circle(self):
        return self._center_inner_circle
    
    @property
    def center_inner_circle_x_coordinate(self):
        return self._center_inner_circle_x_coordinate
    
    @property
    def center_inner_circle_y_coordinate(self):
        return self._center_inner_circle_y_coordinate

    @property
    def center_outter_circle(self):
        return self._center_outter_circle
    
    @property
    def center_outter_circle_x_coordinate(self):
        return self._center_outter_circle_x_coordinate

    @property
    def center_outter_circle_y_coordinate(self):
        return self._center_outter_circle_y_coordinate

    @property
    def name(self):
        return self._name