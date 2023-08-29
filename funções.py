import numpy as np
import matplotlib.pyplot as plt

'função que normaliza um vetor'
def normalize (vector):
    print("a norma do vetor", np.linalg.norm(vector) )
    return vector / np.linalg.norm(vector) 

def camera (E_point, L_point, up_vector, d, height, widht):

    
    w_vector = normalize (E_point - L_point)
    v_vector = normalize (np.cross(w_vector, up_vector))
    u_vector = normalize(np.cross(w_vector, v_vector))
    Vres = height
    Hres = widht 
    distance = d


    screenCenter= np.array(E_point - (distance*w_vector))

    screen = np.zeros((Vres, Hres, 3))

    plt.imshow(screen)
    plt.show() 

E = np.array([30,30,30])
L = np.array([2,4,5])
up = np.array([15,15,15])

camera(E,L, up, 10,20,20)



