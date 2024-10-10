import gzip,numpy,torch
import torch.nn.functional as F
import torch.utils
import sklearn.model_selection as sklearn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShallowN(torch.nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(ShallowN, self).__init__()
        self.hidden = torch.nn.Linear(input_layer, hidden_layer)
        self.output = torch.nn.Linear(hidden_layer, output_layer)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    
    
if __name__ == '__main__':
    batch_size = 8 # nombre de données lues à chaque fois
    nbEpochs = 10 # nombre de fois que la base de données sera lue
    eta = 0.5 # taux d'apprentissage
    nbNeurones = 410 # nombre de neurones mini dans la couche cachée
    mapRes = {} # dictionnaire pour stocker les résultats

    # on lit les données
    ((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('./mnist.pkl.gz'))
    
    size_data = (int)(data_train.shape[0] * 0.2)
    data_train, data_valid = data_train[size_data:] , data_train[:size_data]
    label_train, label_valid = label_train[size_data:] , label_train[:size_data]
    
    train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
    test_dataset = torch.utils.data.TensorDataset(data_test,label_test)    
    
    for nbEpochs in [10, 15, 20]:
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # on crée les lecteurs de données de validation
        valid_dataset = torch.utils.data.TensorDataset(data_valid,label_valid)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)


    
        with open("nbEpochs.txt", "a") as f:
                f.write("nbEpochs  : " + str(nbEpochs) + "\n")
    
        acc = 0.
        x= None
        t = None
        y = None
        loss = None
        
        # Pour stocker l'accuracy à chaque époque
        accuracies = []
        
        model = None
        # on initialise le modèle
        model = ShallowN(data_train.shape[1], nbNeurones, label_train.shape[1])

        loss_func = torch.nn.MSELoss(reduction='mean')   
        optim = torch.optim.SGD(model.parameters(), lr=eta)
        
        model.to(device)
        
        for n in range(nbEpochs):
            # on lit toutes les données d'apprentissage
            for x,t in train_loader:
                x, t = x.to(device), t.to(device)
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
                x, t = x.to(device), t.to(device)
                # on calcule la sortie du modèle
                y = model(x)
                # on regarde si la sortie est correcte
                acc += torch.argmax(y,1) == torch.argmax(t,1)
            # on affiche le pourcentage de bonnes réponses
            print("Données validation : ", n, acc/data_valid.shape[0])
            
            acc /= data_valid.shape[0]
            
            accuracies.append(acc.item())
            
            with open("nbEpochs.txt", "a") as f:
                f.write("Donnees validation : " + str(n) + " : " + str(acc) + "\n")
        
        #Test
        acc = 0.
        for x,t in test_loader:
            x, t = x.to(device), t.to(device)
            # on calcule la sortie du modèle
            y = model(x)
            # on regarde si la sortie est correcte
            acc += torch.argmax(y,1) == torch.argmax(t,1)
        print("Données test : ", acc/data_test.shape[0])
        
        with open("nbEpochs.txt", "a") as f:
            f.write("Donnees test : " + str(acc/data_test.shape[0]) + "\n\n")
            
        final_acc = acc / data_test.shape[0]
        mapRes[nbNeurones] = final_acc.item()
        
    for key, value in mapRes.items():
        print(key, " : ", value)
    print("Meilleur résultat : ", max(mapRes.values()), " avec ", max(mapRes, key=mapRes.get) )