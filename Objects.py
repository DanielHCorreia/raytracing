import numpy as np
from math import inf, sqrt

# Classe Esfera com centro e raio
class Sphere:
    def __init__(self, center, radius, ambient_coeff, diffuse_coeff, specular_coeff, specular_exp, reflection_coeff, transmission_coeff, refraction_index):
        self.center = np.array(center)
        self.radius = radius
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
