import random

def to_degree(radian):
    return radian / 3.14159 * 180

def euler_simplify(angle):
    angle %= 360  
    if (angle > 180):
        return angle - 360
    else:
        return angle

def generate_random_color(normalized=False):
    a = random.randint(100,255)/(255 if normalized else 1)
    b = random.randint(100,255)/(255 if normalized else 1)
    c = random.randint(100,255)/(255 if normalized else 1)
    return (a, b, c)

class Vector3D:
    def __init__(self, x = 0, y = 0, z = 0, vector = [0,0,0]) -> None:
        if (vector != [0,0,0]):
            self.x = -vector[2]
            self.y = vector[0]
            self.z = vector[1]
        elif x != 0 and y != 0 and z != 0:
            self.x = -z
            self.y = x
            self.z = y
    
    def rev_x(self):
        self.x = -self.x
        return self
    
    def vector(self):
        return [self.x, self.y, self.z]
    
    def scale(self, scale):
        self.x *= scale
        self.y *= scale
        self.z *= scale
        return self
    
    def is_zero(self):
        return True if self.x == 0 and self.y == 0 and self.z == 0 else False