import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.FisherFaceRecognizer_create()
lbph = cv2.LBPHFaceRecognizer_create()

def getImageComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImage in caminhos:
        imageFace = cv2.cvtColor(cv2.imread(caminhoImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImage)[-1] ('.') [1])
        #print(id)
        ids.append(id)
        faces.append(imageFace)
        #cv2.imshow("Face", imageFace)
        #cv2.waitKey(10)
        return np.array(ids), face

ids, faces = getImageComId()

print("Treinando...")
eigenface.train(faces, id)
eigenface.write('classifierEigen.yml')

fisherface.train(faces, id)
fisherface.write('classifierFisher.yml')

lbph.train(face, id)
lbph.write('classifierLBPH.yml')

print("Treinamento realizado")