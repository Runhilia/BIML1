import gzip,numpy,torch
import torch.nn.functional as F
import torch.utils
import sklearn.model_selection as sklearn
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def expandecanal(image):
    return image.repeat(1, 3, 1, 1)


((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

# Redimensionner et étendre les canaux
data_train = data_train.view(-1, 1, 28, 28)  # Redimensionner en (N, 1, 28, 28)
data_test = data_test.view(-1, 1, 28, 28)

transforms = transforms.Compose([
    transforms.Lambda(lambda x: expandecanal(x)),
])

data_train = transforms(data_train)
data_test = transforms(data_test)

#data_train = [expandecanal(img) for img in data_train]  # Étendre les canaux à 3
#data_test = [expandecanal(img) for img in data_test]    # Étendre les canaux à 3

# Convertir les données et les labels en tenseurs
dataset = [(data, label) for data, label in zip(data_train, label_train.argmax(dim=1))]
test_dataset = [(data, label) for data, label in zip(data_test, label_test.argmax(dim=1))]

# Division des données en 3 ensembles : train (80%), validation (20%)
train_size = int(0.8 * len(dataset))
val_size = int(0.2 * len(dataset))

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


def train(model, optimizer, loss_func, nbEpochs = 10):
    
    for n in range(nbEpochs):
        model.train()
        local_acc, local_loss = 0,0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            loss = loss_func(outputs, labels)
            local_loss += loss.item()
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
        local_acc, local_loss = 0., 0.
        model.eval()
        for images, labels in val_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_func(outputs, labels)
            local_loss += loss.item()
            
            local_acc += (preds == labels.data).float().sum()
        
        local_loss /= len(val_loader)
        local_acc /= len(val_loader.dataset)
            
        print(f"Epoch {n+1}, accuracy: {local_acc:.4f}, loss: {local_loss:.4f}")
        
        with open("ResNetEpoch" +".txt", "a") as f:
            f.write(f"Epoch {n+1}, accuracy: {local_acc:.4f}, loss: {local_loss:.4f}" + "\n")
        
    local_acc, local_loss = 0., 0.
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = loss_func(outputs, labels)
        local_loss += loss.item()
        
        local_acc += (preds == labels.data).float().sum()
    
    local_loss /= len(test_loader)
    local_acc  /= len(test_loader.dataset)
        
    print(f"Test accuracy: {local_acc:.4f}")
    with open("ResNetEpoch" +".txt", "a") as f:
            f.write(f"Test accuracy: {local_acc:.4f}, loss: {local_loss:.4f}" + "\n\n")
    

if __name__ == '__main__':
    
    for eta in [0.0001]: # 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05
        for batch in [32]: # 8, 16, 32, 64, 128, 256
            for epoch in [10]: #10, 15, 20, 30
                train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

                modelR = models.resnet18()
                
                modelR.fc = nn.Linear(modelR.fc.in_features, 10)

                for param in modelR.fc.parameters():
                    param.requires_grad = True

                optim = torch.optim.Adam(modelR.parameters(), lr=eta)
                loss_func = nn.CrossEntropyLoss()
                modelR = modelR.to(device)
                
                with open("ResNetEpoch" +".txt", "a") as f:
                    f.write("test "+ str(epoch) +" epoch, " +str(eta)+ " eta, "+ str(batch)+" batch, Adam optim\n")
                
                train(modelR, optim, loss_func, epoch)

            
        
    