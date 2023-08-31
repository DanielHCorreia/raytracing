'teste usando a camera e duas esferas simples' 
import numpy as np
import matplotlib.pyplot as plt


# Função para normalizar um vetor
def normalize(vector):
    return vector / np.linalg.norm(vector)


# Função que define a câmera e a tela de projeção
def camera(E_point, L_point, up_vector, d, height, width):
    # Vetores da base da câmera
    w_vector = normalize(E_point - L_point)  # Direção oposta para onde a câmera está olhando
    u_vector = normalize(np.cross(up_vector, w_vector))  # Vetor à direita da câmera
    v_vector = normalize(np.cross(w_vector, u_vector))  # Vetor acima da câmera

    # Calcula o centro da tela de projeção
    screenCenter = E_point - d * w_vector
    # Inicializa a tela com cores pretas
    screen = np.zeros((height, width, 3))

    return screen, u_vector, v_vector, w_vector


# Definição da classe Esfera
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    # Função que verifica se um raio interage com a esfera
    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant > 0:
            return True
        return False


# Definindo a posição e direção da câmera
E = np.array([30, 30, 30])
L = np.array([2, 4, 5])
up = np.array([15, 15, 15])
d = 0.5

# Inicializando a tela e obtendo os vetores da base da câmera
screen, u_vector, v_vector, w_vector = camera(E, L, up, d, 800, 800)
# Lista de objetos na cena
objects = [
    Sphere(np.array([0, 0, -5]), 2, np.array([255, 255, 0])),
    Sphere(np.array([0, 0, -2]), 0.2, np.array([255, 0, 0]))
]


# Dimensões da tela
height, width, _ = screen.shape
aspect_ratio = width / height
fov = 1
scale = np.tan(fov * 0.5)

# Para cada pixel da tela
for i in range(height):
    for j in range(width):
        # Calcula a posição do pixel no espaço da cena
        x = (2 * (j + 0.5) / width - 1) * aspect_ratio * scale
        y = (1 - 2 * (i + 0.5) / height) * scale
        Q = E + d * w_vector + x * u_vector + y * v_vector
        # Direção do raio que passa por esse pixel
        ray_direction = normalize(Q - E)

        # Verifica se o raio interage com algum objeto
        for obj in objects:
            if obj.intersect(E, ray_direction):
                # Se interagir, define a cor do pixel para a cor do objeto
                screen[i, j] = obj.color

# Mostra a imagem gerada
plt.imshow(screen / 255)
plt.show()
