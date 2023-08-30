import numpy as np
import matplotlib.pyplot as plt

'função que normaliza um vetor'
'o norma é a função que vai calcular a norma do vetor'
def normalize (vector):
    
    return vector / np.linalg.norm(vector) 

'''E_point - centro da camera
   L_point - ponto para onde a câmera aponta
   d - distancia entre a camera e a tela
   height e widht - dimensões da tela em pixels'''

def camera (E_point, L_point, up_vector, d, height, widht):
    
    w_vector = normalize (E_point - L_point)
    v_vector = normalize (np.cross(w_vector, up_vector))
    u_vector = normalize(np.cross(w_vector, v_vector))
    Vres = height
    Hres = widht 
    distance = d

    'aqui é calculado o centro da tela'
    screenCenter= np.array(E_point - (distance*w_vector))

    'Definição da nossa tela como uma matriz de triplas de 0'
    screen = np.zeros((Vres, Hres, 3))


    'essas duas linhas fazem aparecer a imagem'
    plt.imshow(screen)
    plt.show() 

'só teste'
E = np.array([30,30,30])
L = np.array([2,4,5])
up = np.array([15,15,15])

camera(E,L, up, 10,20,20)



