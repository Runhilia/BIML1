# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch

if __name__ == '__main__':
    batch_size = 5 # nombre de données lues à chaque fois
    nb_epochs = 10 # nombre de fois que la base de données sera lue
    eta = 0.00001 # taux d'apprentissage

    # on lit les données
    ((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
    print("size of data_train: ", data_train.size())
    print("size of label_train: ", label_train.size())
    print("size of data_test: ", data_test.size())
    print("size of label_test: ", label_test.size())

    # on initialise le modèle et ses poids
    w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
    b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
    torch.nn.init.uniform_(w,-0.001,0.001)
    torch.nn.init.uniform_(b,-0.001,0.001)

    done = False
    nb_data_train = data_train.shape[0]
    nb_data_test = data_test.shape[0]
    indices = numpy.arange(nb_data_train)
    for n in range(nb_epochs):
        # on mélange les (indices des) données
        numpy.random.shuffle(indices)
        # on lit toutes les (indices des) données d'apprentissage
        for i in range(0,nb_data_train,batch_size):
            # on récupère les entrées
            x = data_train[indices[i:i+batch_size]]
            
            # on calcule la sortie du modèle
            y = torch.mm(x,w)+b     # on définit le nombre de neuronnes ici
            
            # on regarde les vrais labels
            t = label_train[indices[i:i+batch_size]]
            
            # on met à jour les poids
            grad = (t-y)
            w += eta * torch.mm(x.T,grad)
            b += eta * grad.sum(axis=0)
            
            if not done : 
                print("size of x: ", x.size())
                print("size of y: ", y.size())
                print("size of t: ", t.size())
                print("size of grad: ", grad.size())
                print("size of w: ", w.size())
                print("size of b: ", b.size())
                done = True

            # test du modèle (on évalue la progression pendant l'apprentissage)
            acc = 0.
            
        # on lit toutes les donnéees de test
        for i in range(nb_data_test):
            # on récupère l'entrée
            x = data_test[i:i+1]
            # on calcule la sortie du modèle
            y = torch.mm(x,w)+b
            # on regarde le vrai label
            t = label_test[i:i+1]
            # on regarde si la sortie est correcte
            acc += torch.argmax(y,1) == torch.argmax(t,1)
            
        # on affiche le pourcentage de bonnes réponses
        print(acc/nb_data_test)
