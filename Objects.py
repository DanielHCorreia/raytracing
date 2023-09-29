import numpy as np
from funções import *
from math import inf, sqrt

# Classe Esfera com centro e raio
class Sphere:
    def __init__(self, center, radius, ambient_coeff, diffuse_coeff, specular_coeff, specular_exp, reflection_coeff, transmission_coeff, refraction_index):
        self.center = np.array(center)
        self.radius = radius
        # Coeficientes de iluminação
        self.ambient_coeff = ambient_coeff #ka
        self.diffuse_coeff = diffuse_coeff #kd
        self.specular_coeff = specular_coeff #ks
        self.specular_exp = specular_exp #ks
        self.reflection_coeff = reflection_coeff 
        self.transmission_coeff = transmission_coeff #kt
        self.refraction_index = refraction_index #n

    def set_color(self, RGB_color):
        self.color = np.array(RGB_color)

    def get_color(self):
        return self.color


    def translation(self, vector):
        self.center = [v1 + v2 for v1, v2  in zip(self.center, vector)]

    def rotate(self, x=0, y=0, z=0, point = (0,0,0)):
        self.center = rotate_x(x, point,self.center)
        self.center = rotate_y(y, point, self.center)
        self.center = rotate_z(z, point, self.center)

    # Calcula a normal na superfície da esfera
    def get_normal(self, point):
        normal_vector = np.array(point - self.center)
        normalized_normal = normal_vector / np.linalg.norm(normal_vector)
        return normalized_normal

    # Calcula a interseção do raio com a esfera
    @staticmethod
    def raysphere_intersect(center, radius, ray_origin, ray_direction):
        offset = center - ray_origin
        closest_approach = np.dot(offset, ray_direction)
        distance_squared = np.dot(offset, offset) - (closest_approach ** 2)
        
        if distance_squared <= (radius ** 2):
            delta = sqrt(radius ** 2 - distance_squared)
            intersection1 = closest_approach - delta
            intersection2 = closest_approach + delta
            
            if intersection1 < 0:
                if intersection2 < 0:
                    return inf
                else:
                    return intersection2
            else:
                return intersection1
        else:
            return inf

    def __str__(self):
        return "Sphere"

# Classe Plano com vetor normal e um ponto no plano
class Plane:
    def __init__(self, normal_vector, point, ambient_coeff, diffuse_coeff, specular_coeff, specular_exp, reflection_coeff, transmission_coeff, refraction_index):
        self.normal_vector = np.array(normal_vector)
        self.point = np.array(point)
        # Coeficientes de iluminação
        self.ambient_coeff = ambient_coeff
        self.diffuse_coeff = diffuse_coeff
        self.specular_coeff = specular_coeff
        self.specular_exp = specular_exp
        self.reflection_coeff = reflection_coeff
        self.transmission_coeff = transmission_coeff
        self.refraction_index = refraction_index



    def set_color(self, RGB_color):
        self.color = np.array(RGB_color)

    def get_color(self):
        return self.color

    def translation(self, vector):
        self.point = [v1 + v2 for v1, v2 in zip(self.point, vector)]

    def rotate(self, x=0, y=0, z=0, point = (0,0,0)):
        self.point = rotate_x(x, point, self.point)
        self.point = rotate_y(y, point, self.point)
        self.point = rotate_z(z, point, self.point)
        self.normal = rotate_x(x, point, self.normal)
        self.normal = rotate_y(y, point, self.normal)
        self.normal = rotate_z(z, point, self.normal)

    # Retorna o vetor normal do plano
    def get_normal(self, _):
        return self.normal_vector

    # Calcula a interseção do raio com o plano
    @staticmethod
    def rayplane_intersect(normal_vector, point, ray_origin, ray_direction, epsilon=1e-6):
        dot_product = np.dot(ray_direction, normal_vector)
        
        if abs(dot_product) > epsilon:
            w = point - ray_origin
            t = np.dot(w, normal_vector) / dot_product
            if t < 0:
                return inf
            return t
        else:
            return inf

    def __str__(self):
        return "Plane"

# Classe Triângulo com três pontos
class Triangle:
    def __init__(self, pointA, pointB, pointC, ambient_coeff, diffuse_coeff, specular_coeff, specular_exp, reflection_coeff, transmission_coeff, refraction_index):
        self.pointA = np.array(pointA)
        self.pointB = np.array(pointB)
        self.pointC = np.array(pointC)
        # Coeficientes de iluminação
        self.ambient_coeff = ambient_coeff
        self.diffuse_coeff = diffuse_coeff
        self.specular_coeff = specular_coeff
        self.specular_exp = specular_exp
        self.reflection_coeff = reflection_coeff
        self.transmission_coeff = transmission_coeff
        self.refraction_index = refraction_index

    def set_color(self, RGB_color):
        self.color = np.array(RGB_color)

    def get_color(self):
        return self.color

    def translation(self, vector):
        self.a = [v1 + v2 for v1, v2 in zip(self.a, vector)]
        self.b = [v1 + v2 for v1, v2 in zip(self.b, vector)]
        self.c = [v1 + v2 for v1, v2 in zip(self.c, vector)]

    # Calcula a normal na superfície do triângulo
    def get_normal(self, _):
        vectorAB = self.pointB - self.pointA
        vectorAC = self.pointC - self.pointA
        normal_vector = np.cross(vectorAB, vectorAC)
        normalized_normal = normal_vector / np.linalg.norm(normal_vector)
        return normalized_normal

    # Calcula a interseção do raio com o triângulo
    @staticmethod
    def raytriangle_intersect(pointA, pointB, pointC, ray_origin, ray_direction):
        # Pré-processamento
        vectorU = pointB - pointA
        vectorV = pointC - pointA
        normal = np.cross(vectorU, vectorV)
        normalized_normal = normal / np.linalg.norm(normal)

        t = Plane.rayplane_intersect(normalized_normal, pointA, ray_origin, ray_direction)

        if t < inf:
            intersection_point = ray_origin + t * ray_direction
            vectorAP = intersection_point - pointA

            beta = np.dot(vectorAP, vectorU)
            gamma = np.dot(vectorAP, vectorV)
            alpha = 1 - (beta + gamma)

            if alpha < 0 or beta < 0 or gamma < 0:
                return inf
        return t

    def __str__(self):
        return "Triangle"


class TriangleMesh:
    def __init__(self, faces, vertices, ambient_coeff, diffuse_coeff, specular_coeff, specular_exp, reflection_coeff, transmission_coeff, RGB_color):
        self.ambient_coeff = ambient_coeff
        self.diffuse_coeff = diffuse_coeff
        self.specular_coeff = specular_coeff
        self.specular_exp = specular_exp
        self.reflection_coeff = reflection_coeff
        self.transmission_coeff = transmission_coeff
        self.vertices = vertices
        self.faces = faces
    
        self.color = np.array(RGB_color)

    

    def generate_triangles(self):
        triangles = []
        for face in self.faces:
            triangle = Triangle(self.vertices[int(face[0])-1], self.vertices[int(face[1])-1], self.vertices[int(face[2])-1],  self.ambient_coeff, self.diffuse_coeff, self.specular_coeff, self.specular_exp, self.transmission_coeff, self.color)
            triangles.append(triangle)
        return triangles
    
    def print_triangles(self):
        triangles = []
        for face in self.faces:
            triangle = Triangle(self.vertices[int(face[0])-1], self.vertices[int(face[1])-1], self.vertices[int(face[2])-1],  self.ambient_coeff, self.diffuse_coeff, self.specular_coeff, self.specular_exp, self.transmission_coeff, self.color)
            triangles.append(triangle.getTriangle())
        return triangles

    def intersect(self, ray_origin, ray_direction):
        intersect = []
        for triangle in self.generate_triangles():
            if triangle.intersect(ray_origin, ray_direction) != None:
                intersect.append(triangle.intersect(ray_origin, ray_direction))
        
        if len(intersect) == 0:
            return None
        
        else:
            return min(intersect)
        
    def translation(self, vector):
        for i in range(len(self.vertices)):
            self.vertices[i] = [v1 + v2 for v1, v2 in zip(self.vertices[i], vector)]

    def rotate(self, x = 0, y = 0, z = 0, point = (0, 0, 0)):
        for i in range(len(self.vertices)):
            self.vertices[i] = rotate_x(self.vertices[i], x, point)
            self.vertices[i] = rotate_y(self.vertices[i], y, point)
            self.vertices[i] = rotate_z(self.vertices[i], z, point)
