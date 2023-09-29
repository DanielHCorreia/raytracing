import numpy as np
import matplotlib.pyplot as plt

def rotate_x(angle, point,center):
    angle = np.radians(angle)
    aux = center - point
    aux = np.array([aux[0], aux[1] * np.cos(angle) - aux[2] * np.sin(angle),
                       aux[1] * np.sin(angle) + aux[2] * np.cos(angle)])
    center = aux + point
    return center


def rotate_y(angle,point, center):
    angle = np.radians(angle)
    aux = center - point
    aux = np.array([aux[0] * np.cos(angle) + aux[2] * np.sin(angle), aux[1],
                       -aux[0] * np.sin(angle) + aux[2] * np.cos(angle)])
    center = aux + point
    return center

def rotate_z(angle,point, center):
    angle = np.radians(angle)
    aux = center - point
    aux = np.array([aux[0] * np.cos(angle) - aux[1] * np.sin(angle),
                       aux[0] * np.sin(angle) + aux[1] * np.cos(angle), aux[2]])
    center = aux + point
    return center



'função que normaliza um vetor'
'o norma é a função que vai calcular a norma do vetor'
def normalize (vector):
    
    return vector / np.linalg.norm(vector) 

'''E_point - centro da camera
   L_point - ponto para onde a câmera aponta
   d - distancia entre a camera e a tela
   height e widht - dimensões da tela em pixels'''

def camera (E_point, L_point, up_vector, d, height, widht, s):
    
    w_vector = normalize (E_point - L_point)
    v_vector = normalize (np.cross(w_vector, up_vector))
    u_vector = normalize(np.cross(w_vector, v_vector))
    Vres = height
    Hres = widht 
    distance = d
    pixel_side = s

    'aqui é calculado o centro da tela'
    screenCenter= np.array(E_point - (distance*w_vector))

    'Definição da nossa tela como uma matriz de triplas de 0'
    screen = np.zeros((Vres, Hres, 3))
    image = np.zeros((Vres, Hres, 3))
    c = np.zeros((Vres, Hres, 3))

    screen[0,0] = np.array((screenCenter + 1/2 * pixel_side * (Vres - 1) * v_vector) - 
    (1/2 * pixel_side * (Hres - 1) * u_vector))


    for i in range(Vres):
        for j in range(Hres):
            #Cálculo do Qij
            screen[i,j] = screen[0,0] + (pixel_side * j * u_vector) - (pixel_side * i * v_vector)
            ray_direction = normalize(screen[i, j] - E_point)
            #aux = cast(objects, E_point, ray_direction, BC_RGB, Ca, lights, max_depth)

            #Tratando o Overflow
            aux = aux/max(*aux,1)
            image[i , j] = aux 
            
    return image

    #'essas duas linhas fazem aparecer a imagem'
    #plt.imshow(screen)
    #plt.show() 

'só teste'
E = np.array([30,30,30])
L = np.array([2,4,5])
up = np.array([15,15,15])

camera(E,L, up, 10,20,20)



