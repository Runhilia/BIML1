import gzip,numpy,torch
import torch.nn.functional as F
import torch.utils
import sklearn.model_selection as sklearn
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepNetwork(torch.nn.Module):
    
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(DeepNetwork, self).__init__()
        layers= []
        hidden1 = torch.nn.Linear(input_layer, hidden_layer[0])
        
        layers.append(hidden1)
        
        for i in range(1, len(hidden_layer)):
            hidden2 = torch.nn.Linear(hidden_layer[i-1], hidden_layer[i])
            layers.append(hidden2)
        
        hidden3 = torch.nn.Linear(hidden_layer[-1], output_layer)
        layers.append(hidden3)
        
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    
def train_and_eval_model(hidden_layers, train_loader, valid_loader, eta = 0.05, nbEpochs = 10):
    acc = 0.
    x= None
    t = None
    y = None
    loss = None

    model = None
    
        
    model = DeepNetwork(data_train.shape[1], hidden_layers, label_train.shape[1])
    
    model.to(device)
    loss_func = torch.nn.MSELoss(reduction='mean')   
    optim = torch.optim.SGD(model.parameters(), lr=eta)
    
    for n in range(nbEpochs):
        # on lit toutes les données d'apprentissage
        for x,t in train_loader:
            x= x.to(device)
            t= t.to(device)
            
            # on calcule la sortie du modèle
            y = model(x)
            # on met à jour les poids
            loss = loss_func(t,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
        # validation du modèle (on évalue la progression pendant l'apprentissage)
        acc = 0.
        # on lit toutes les donnéees de validation
        for x,t in valid_loader:
            x = x.to(device)
            t = t.to(device)
            # on calcule la sortie du modèle
            y = model(x)
            # on regarde si la sortie est correcte
            acc += torch.argmax(y,1) == torch.argmax(t,1)
        # on affiche le pourcentage de bonnes réponses
        print("Données validation : ", n, acc/data_valid.shape[0])
        
        acc /= data_valid.shape[0]
        
        with open("epoch.txt", "a") as f:
            f.write("Donnees validation : " + str(n) + " : " + str(acc) + "\n")
            
    acc = 0.
    for x,t in test_loader:
        x = x.to(device)
        t = t.to(device)
        # on calcule la sortie du modèle
        y = model(x)
        # on regarde si la sortie est correcte
        acc += torch.argmax(y,1) == torch.argmax(t,1)
    print("Données test : ", acc/data_test.shape[0])
    
    acc /= data_test.shape[0]
    
    return acc


if __name__ == '__main__':
    batch_size = 1 # nombre de données lues à chaque fois
    
    best_accuracy = 0
    best_config = None
    best_eta = None

    # on lit les données
    ((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    size_data = (int)(data_train.shape[0] * 0.2)
    data_train, data_valid = data_train[size_data:] , data_train[:size_data]
    label_train, label_valid = label_train[size_data:] , label_train[:size_data]
    
    for batch in [1]:
        train_dataset = TensorDataset(data_train,label_train)
        test_dataset = TensorDataset(data_test,label_test)    
        
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # on crée les lecteurs de données de validation
        valid_dataset = TensorDataset(data_valid,label_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        
        for n in [10]: #, 15
            for etaa in [0.1]: #0.01, 0.05, 0.1
                with open("epoch.txt", "a") as f:
                    f.write("test "+ str(n) +" epoch, " +str(etaa)+ " eta, "+ str(batch)+ " batch " + "\n")
                for nb_layers in [3]: #2,3
                        for neurones_layer1 in [650]: #100, 700, 200
                            for neurones_layer2 in [350]:
                                if(neurones_layer2>neurones_layer1):
                                    continue
                                hidden_layers = [neurones_layer1, neurones_layer2]
                                if nb_layers == 3:
                                    for neurones_layer3 in [350]:
                                        if(neurones_layer3>neurones_layer2):
                                            continue
                                        hidden_layers = [neurones_layer1, neurones_layer2, neurones_layer3]
                                        
                                        print("Hidden layers :", hidden_layers)
                                        accuracy = train_and_eval_model(hidden_layers, train_loader, valid_loader, eta = etaa, nbEpochs = n)
                                        print("Accuracy :", accuracy)
                                        
                                        with open("epoch.txt", "a") as f:
                                            f.write("Hidden layers : " + str(hidden_layers) + " : " + str(accuracy) + "\n\n")
                                        
                                        if accuracy > best_accuracy:
                                            best_accuracy = accuracy
                                            best_config = batch
                                
            
        
        with open("epoch.txt", "a") as f:
            f.write("Meilleur resultat :" + str(best_accuracy) + " avec " + str(best_config) )
                                
        print("Meilleur résultat :", best_accuracy, " avec ", best_config)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """ for batch in range (1, 32, 7):
        train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
        test_dataset = torch.utils.data.TensorDataset(data_test,label_test)    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # on crée les lecteurs de données de validation
        valid_dataset = torch.utils.data.TensorDataset(data_valid,label_valid)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
        for etaa in [0.1]: #0.01, 0.05,
            with open("batchSize" +".txt", "a") as f:
                f.write("batch : " + str(batch) + "\n")                   
            for nb_layers in [2]:
                for neurones_layer1 in [600]:
                    for neurones_layer2 in [300]:
                        if(neurones_layer2>neurones_layer1):
                            continue
                        hidden_layers = [neurones_layer1, neurones_layer2]
                        if nb_layers == 3:
                            for neurones_layer3 in range(100, 700, 100):
                                if(neurones_layer3>neurones_layer2):
                                    continue
                                hidden_layers = [neurones_layer1, neurones_layer2, neurones_layer3]
                                
                                print("Hidden layers :", hidden_layers)
                                accuracy = train_and_eval_model(hidden_layers, train_loader, valid_loader, eta = etaa)
                                print("Accuracy :", accuracy)
                                
                                with open("nbCouche"+ str(etaa) +".txt", "a") as f:
                                    f.write("Hidden layers : " + str(hidden_layers) + " : " + str(accuracy) + "\n\n")
                                
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_config = batch
                                    #best_eta = etaa
                        else:
                            print("Hidden layers :", hidden_layers)
                            accuracy = train_and_eval_model(hidden_layers, train_loader, valid_loader, eta= etaa)
                            print("Accuracy :", accuracy)
                            
                            with open("batchSize" +".txt", "a") as f:
                                f.write("Accuracy : " + str(accuracy) + "\n\n")
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_config = batch
                                #best_eta = etaa
                                
            #with open("batchSize" +".txt", "a") as f:
            #    f.write("Meilleur résultat :" + str(best_accuracy) + "avec" + str(best_config) )
         """
    
    
    
    