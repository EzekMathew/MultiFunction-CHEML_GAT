#imports specific to environment

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
print(tf.__version__)
!python --version
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
!pip install rdkit-pypi -qqq
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from IPython.display import SVG
#from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
import torch
!pip install torch_geometric
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
from rdkit.Chem.Draw import DrawingOptions, rdMolDraw2D
from matplotlib import cm
import io
from PIL import Image as PILImage

from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image as PILImage
import io
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.nn.functional import mse_loss

!pip install openpyxl
from google.colab import drive
drive.mount('/content/drive') #if using colab 




hc = 256



#First is Task1; this is a classification task, as it seeks to predict ligand selectivity
"""
print("Training set size (SMILES): ", smile s_train.shape)
print("Validation set size (SMILES): ", smiles_val.shape)
print("Training set size (y): ", y_train.shape)
print("Validation set size (y): ", y_val.shape)
print("Training set size (y2): ", y2_train.shape)
print("Validation set size (y2): ", y3_val.shape)
print("Training set size (name): ", Name_train.shape)
print("Validation set size (name): ", Name_val.shape)


invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

print(f'Found {len(invalid_smiles)} invalid SMILES strings.')
# Get the first SMILES string in the training set
first_smiles_train = smiles_train.iloc[0]
# Get the adjacency matrix, edge features, and node features for the first SMILES string in the training set
adjacency_matrix, edge_features, node_features = smiles_to_graph(first_smiles_train)
# Print the adjacency matrix, edge features, and node features
print("Node Features shape:", np.array(node_features).shape)
print("First SMILES String in Training Set:", first_smiles_train)
print("Adjacency Matrix:")
print(adjacency_matrix)
print("\nEdge Features:")
print(edge_features)
print("\nNode Features:")
print(node_features)





invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

max_nodes = 0
for i, smiles in enumerate(smiles_train):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

for i, smiles in enumerate(smiles_test):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

print("\nmax_nodes:")
print(max_nodes)

"""

permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Mg', "I"]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(permitted_list_of_atoms).reshape(-1, 1)) # reshape for 1 feature
def atom_features(atom):
    ComputeGasteigerCharges(atom.GetOwningMol())  # Compute Gasteiger charges for the molecule
    # Get the one-hot encoded representation of the atom symbol
    symbol_one_hot = encoder.transform(np.array(atom.GetSymbol()).reshape(-1, 1))
    # Decompose the one-hot encoded symbol into multiple 1D arrays
    symbol_C, symbol_N, symbol_O, symbol_S, symbol_F, symbol_P, symbol_Cl, symbol_Br, symbol_Mg, symbol_I= symbol_one_hot.T
    degree = np.eye(6)[atom.GetDegree()] if atom.GetDegree() < 6 else np.array([0, 0, 0, 0, 0, 1])
    total_num_hs = np.eye(5)[atom.GetTotalNumHs()] if atom.GetTotalNumHs() < 5 else np.array([0, 0, 0, 0, 1])
    implicit_valence = np.eye(6)[atom.GetImplicitValence()] if atom.GetImplicitValence() < 6 else np.array([0, 0, 0, 0, 0, 1])
    is_aromatic = np.array([atom.GetIsAromatic()], dtype=np.float32)
    gasteiger_charge = np.array([float(atom.GetProp('_GasteigerCharge'))], dtype=np.float32) #Feature id is 28
    hybridization = np.eye(7)[atom.GetHybridization().real] if atom.GetHybridization().real < 7 else np.array([0, 0, 0, 0, 0, 0, 1])
    explicit_valence = np.eye(10)[atom.GetExplicitValence()] if atom.GetExplicitValence() < 10 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    formal_charge = np.eye(5)[atom.GetFormalCharge()+2] if -2<=atom.GetFormalCharge()<=2 else np.array([0, 0, 0, 0, 0, 1])
    atomic_mass_scaled = [float((atom.GetMass())/35)] #Feature id is 51
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())-1.5)/(1.7-1.5))] #Feature id is 52
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())-0.64)/(0.68-0.64))] #Feature id is 53
    return np.concatenate([symbol_C.ravel(), symbol_N.ravel(), symbol_O.ravel(), symbol_S.ravel(), symbol_F.ravel(), symbol_P.ravel(), symbol_Cl.ravel(), symbol_Br.ravel(), symbol_Mg.ravel(), symbol_I.ravel(), degree, total_num_hs, implicit_valence, is_aromatic, gasteiger_charge, hybridization, explicit_valence, formal_charge, atomic_mass_scaled, vdw_radius_scaled, covalent_radius_scaled])

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))  # use atom_features function here
    # Get adjacency matrix and edge features (bond types)
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Uncomment these lines to fill the adjacency matrix
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        edge_features.append((i, j))
    return adjacency_matrix, edge_features, node_features


data= pd.read_excel("/content/.....xlsx")


train = data[data["fold"]!=foldnumber]
test = data[data["fold"]==foldnumber]

Name_train = train['Original Label']
Name_test = test['Original Label']

x = data.pop('SMILES')
smiles = x
smiles_train = train['SMILES']
print("smiles_train.shape", smiles_train.shape)
smiles_test = test['SMILES']
print("smiles_test.shape", smiles_test.shape)

yreserve = data['ReceptorBinarized'].copy()
y = data.pop('ReceptorBinarized')
y_train = train['ReceptorBinarized']
y_test = test['ReceptorBinarized']

def one_hot_encode_y(y):
    # Map y values to their one-hot encodings
    mapping = {
        0: [1, 0, 0],
        1: [0, 1, 0],
        2: [0, 0, 1]
    }
    return torch.tensor(mapping[y], dtype=torch.float)

smiles_train, smiles_val, y_train, y_val = train_test_split(smiles_train, y_train, test_size=0.05, random_state=902203)
max_nodes = 40 # with mGlu2&3 dataset, max nodes is actually 40
num_node_features = 54


test_data = []
test_labels = []  #for 'Original Label'
for i, smiles in enumerate(smiles_test):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_test.iloc[i]))
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    test_data.append(data)
    test_labels.append(Name_test.iloc[i])

val_data = []
for i, smiles in enumerate(smiles_val):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_val.iloc[i]))  # Convert to integer and then pass to one_hot_encode_y()
    #print(y.shape)
    #print(x.shape)
    #old data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    val_data.append(data)

train_data = []
for i, smiles in enumerate(smiles_train):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_train.iloc[i]))
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    train_data.append(data)



train_loader = DataLoader(train_data, batch_size=250, shuffle=True) #feel free to try various batch sizes, depending on your own data volume
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)





class mGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(mGAT, self).__init__()
        self.att_CL1 = GATConv(num_node_features, hidden_channels)
        self.att_CL2 = GATConv(hidden_channels, hidden_channels)
        self.att_CL3 = GATConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, 3)  # Classification
        self.fc2 = Linear(hidden_channels, 1)  # Regression. Try 2 and 1 class. problem with 2 class is that for correct calculation, ytrue needs to be known, thus giving algo an unfair advantage
        self.fc3 = Linear(hidden_channels, 2)  # 2 class IC50, low vs high. no unfait advantage this way.


    def forward(self, data):
        #what to do about shared layers
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.att_CL1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.att_CL2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.att_CL3(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = global_mean_pool(x1, batch)
        x1 = self.fc1(x1)
        return x1


def test(model, loader, test_labels):
    model.eval()
    correct = 0
    y_preds = []
    y_trues = []
    original_labels = []
    for i, data in enumerate(loader):
        out1 = model(data)
        _, pred = torch.max(out1, 1)  # Get the index of the max logi which schould be indicative of predicted class
        correct += (pred == torch.argmax(data.y, dim=1)).sum().item()
        pred_labels = pred
        y_preds.extend(pred_labels.tolist())
        y_trues.extend(data.y.tolist())
        original_labels.append(test_labels[i])

    accuracy = correct / (len(loader.dataset)) #nomultiplier needed as argmax takes care of the cass number issue
    return accuracy, original_labels, y_preds, y_trues

def train(model, loader, optimizer, criterion1):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out1 = model(data)
        #print("Shape of data.y:", data.y.shape)
        #print("Shape of out1:", out1.shape)
        loss1 = criterion1(out1, data.y.float())
        total_loss += loss1.item() * data.num_graphs
        loss1.backward()
        optimizer.step()
    return total_loss / len(loader.dataset)
def validate(model, loader, criterion1):
    model.eval()
    total_loss1 = 0
    correct = 0
    for data in loader:
        out1 = model(data)
        loss1 = criterion1(out1, data.y.float())
        total_loss1 += loss1.item() * data.num_graphs
        _, pred_probs  = torch.max(out1, 1)  # Get the index of the max logi which schould be indicative of predicted class
        correct += (pred_probs  == torch.argmax(data.y, dim=1)).sum().item()
        pred_labels = pred_probs
    accuracy = correct / (len(loader.dataset))
    return total_loss1 / len(loader.dataset), accuracy

#get prepped to save stuff
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
criterion1 = BCEWithLogitsLoss()
model = mGAT(num_node_features=54, hidden_channels=hc)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(250):
    train_loss = train(model, train_loader, optimizer, criterion1)
    train_losses.append(train_loss)
    train_accuracy, _, _, _ = test(model, train_loader, test_labels)
    train_accuracies.append(train_accuracy)

    val_loss1, val_accuracy = validate(model, val_loader, criterion1)
    val_losses.append(val_loss1)
    val_accuracies.append(val_accuracy)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss1:.4f}, Val Acc: {val_accuracy:.4f}')


accuracy, original_labels, y_preds, y_trues = test(model, test_loader, test_labels)
print(f'Test Acc after classification: {accuracy:.4f}')

# Save the model, need to specify freeze later
torch.save(model.state_dict(), 'model_task1.pth')


# plot training and validation losses for classification
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train and Validation Loss over epochs (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#dataframe for save
# Create a dictionary to hold the values
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)





print(df)


def save_to_excel(df, foldnumber):
    if 1 <= foldnumber <= 11:  # Only allow fold numbers from 1 to 6
        filename = f"fold{foldnumber}_S.xlsx"  # Generate filename to save selectivity data
        df.to_excel(filename, index=False)  # Save DataFrame to an Excel file
        !cp {filename} "drive/My Drive/" #neeed to specify your own directory
    else:
        print("Invalid fold number. Please provide a fold number between 1 and 6.")

# To call the function, you can use something like this:
save_to_excel(df, foldnumber)  # Replace 'foldnumber' with the relevant fold number





#next, onto Task2; this is a regression task, as it seeks to predict IC50 values

"""
print("Training set size (SMILES): ", smiles_train.shape)
print("Validation set size (SMILES): ", smiles_val.shape)
print("Training set size (y): ", y_train.shape)
print("Validation set size (y): ", y_val.shape)
print("Training set size (y2): ", y2_train.shape)
print("Validation set size (y2): ", y3_val.shape)
print("Training set size (name): ", Name_train.shape)
print("Validation set size (name): ", Name_val.shape)


invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

print(f'Found {len(invalid_smiles)} invalid SMILES strings.')
# Get the first SMILES string in the training set
first_smiles_train = smiles_train.iloc[0]
# Get the adjacency matrix, edge features, and node features for the first SMILES string in the training set
adjacency_matrix, edge_features, node_features = smiles_to_graph(first_smiles_train)
# Print the adjacency matrix, edge features, and node features
print("Node Features shape:", np.array(node_features).shape)
print("First SMILES String in Training Set:", first_smiles_train)
print("Adjacency Matrix:")
print(adjacency_matrix)
print("\nEdge Features:")
print(edge_features)
print("\nNode Features:")
print(node_features)





invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

max_nodes = 0
for i, smiles in enumerate(smiles_train):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

for i, smiles in enumerate(smiles_test):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

print("\nmax_nodes:")
print(max_nodes)

"""

permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Mg', "I"]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(permitted_list_of_atoms).reshape(-1, 1)) # reshape for 1 feature
def atom_features(atom):
    ComputeGasteigerCharges(atom.GetOwningMol())  # Compute Gasteiger charges for the molecule
    # Get the one-hot encoded representation of the atom symbol
    symbol_one_hot = encoder.transform(np.array(atom.GetSymbol()).reshape(-1, 1))
    # Decompose the one-hot encoded symbol into multiple 1D arrays
    symbol_C, symbol_N, symbol_O, symbol_S, symbol_F, symbol_P, symbol_Cl, symbol_Br, symbol_Mg, symbol_I= symbol_one_hot.T
    degree = np.eye(6)[atom.GetDegree()] if atom.GetDegree() < 6 else np.array([0, 0, 0, 0, 0, 1])
    total_num_hs = np.eye(5)[atom.GetTotalNumHs()] if atom.GetTotalNumHs() < 5 else np.array([0, 0, 0, 0, 1])
    implicit_valence = np.eye(6)[atom.GetImplicitValence()] if atom.GetImplicitValence() < 6 else np.array([0, 0, 0, 0, 0, 1])
    is_aromatic = np.array([atom.GetIsAromatic()], dtype=np.float32)
    gasteiger_charge = np.array([float(atom.GetProp('_GasteigerCharge'))], dtype=np.float32) #Feature id is 28
    hybridization = np.eye(7)[atom.GetHybridization().real] if atom.GetHybridization().real < 7 else np.array([0, 0, 0, 0, 0, 0, 1])
    explicit_valence = np.eye(10)[atom.GetExplicitValence()] if atom.GetExplicitValence() < 10 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    formal_charge = np.eye(5)[atom.GetFormalCharge()+2] if -2<=atom.GetFormalCharge()<=2 else np.array([0, 0, 0, 0, 0, 1])
    atomic_mass_scaled = [float((atom.GetMass())/35)] #Feature id is 51
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())-1.5)/(1.7-1.5))] #Feature id is 52
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())-0.64)/(0.68-0.64))] #Feature id is 53
    return np.concatenate([symbol_C.ravel(), symbol_N.ravel(), symbol_O.ravel(), symbol_S.ravel(), symbol_F.ravel(), symbol_P.ravel(), symbol_Cl.ravel(), symbol_Br.ravel(), symbol_Mg.ravel(), symbol_I.ravel(), degree, total_num_hs, implicit_valence, is_aromatic, gasteiger_charge, hybridization, explicit_valence, formal_charge, atomic_mass_scaled, vdw_radius_scaled, covalent_radius_scaled])

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))  # use atom_features function here
    # Get adjacency matrix and edge features (bond types)
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Uncomment these lines to fill the adjacency matrix
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        edge_features.append((i, j))
    return adjacency_matrix, edge_features, node_features



data= pd.read_excel("/content/.....xlsx")


#foldnumber = 1
#just use the one specified before
train = data[data["fold"]!=foldnumber]
test = data[data["fold"]==foldnumber]

Name_train = train['Original Label']
Name_test = test['Original Label']

x = data.pop('SMILES')
smiles = x
smiles_train = train['SMILES']
print("smiles_train.shape", smiles_train.shape)
smiles_test = test['SMILES']
print("smiles_test.shape", smiles_test.shape)







y = data.pop('mGluaffinity')
y = np.log10(y)
min_mGluaffinity = y.min()
max_mGluaffinity = y.max()
print(max_mGluaffinity)
print(min_mGluaffinity)
y_train = train['mGluaffinity']
y_test = test['mGluaffinity']
y_train = np.log10(y_train)
y_test = np.log10(y_test)
#y_train = 2 * ((y_train - min_mGluaffinity) / (max_mGluaffinity - min_mGluaffinity)) - 1
#y_test = 2 * ((y_test - min_mGluaffinity) / (max_mGluaffinity - min_mGluaffinity)) - 1
y_train = ((y_train - min_mGluaffinity) / (max_mGluaffinity - min_mGluaffinity))
y_test = ((y_test - min_mGluaffinity) / (max_mGluaffinity - min_mGluaffinity))

min_mGluaffinity = y_train.min()
max_mGluaffinity = y_train.max()

print(max_mGluaffinity)
print(min_mGluaffinity)








smiles_train, smiles_val, y_train, y_val = train_test_split(smiles_train, y_train, test_size=0.053, random_state=902203)
max_nodes = 40 # with mGlu2&3 dataset, max nodes is actually 40
num_node_features = 54

val_data = []
for i, smiles in enumerate(smiles_val):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y_val.iloc[i], dtype=torch.float)
    #print(y.shape)
    #print(x.shape)
    #old data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    val_data.append(data)

train_data = []
for i, smiles in enumerate(smiles_train):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y_train.iloc[i], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    train_data.append(data)

test_data = []
test_labels = []  #for 'Original Label storage'
for i, smiles in enumerate(smiles_test):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y_test.iloc[i], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    test_data.append(data)
    test_labels.append(Name_test.iloc[i])

train_loader = DataLoader(train_data, batch_size=250, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)











###>below is for regression


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def custommse(self, yhat, y):
        #return torch.mean(((yhat - y)**2)/(1))
        return torch.mean(((yhat - y)**2)/((y+0.05)))

    def forward(self, yhat, y):
        return torch.sqrt(self.custommse(yhat, y))



class mGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(mGAT, self).__init__()
        self.att_CL1 = GATConv(num_node_features, hidden_channels)
        self.att_CL2 = GATConv(hidden_channels, hidden_channels)
        self.att_CL1share = GATConv(num_node_features, hidden_channels)
        self.att_CL2share = GATConv(hidden_channels, hidden_channels)
        self.att_CL3 = GATConv(hidden_channels, hidden_channels)
        self.att_CL4 = GATConv(hidden_channels, hidden_channels)
        self.att_CL5 = GATConv(hidden_channels, hidden_channels)
        self.att_CL6 = GATConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, 3)  # Classification
        self.fc2 = Linear(hidden_channels, 1)  # Regression. Try 2 and 1 class. problem with 2 class is that for correct calculation, ytrue needs to be known, thus giving algo an unfair advantage
        self.fc3 = Linear(hidden_channels, 2)  # 2 class IC50, low vs high. no unfait advantage this way.

    def forward(self, data):
        #what to do about shared layers
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.att_CL1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.15, training=self.training)
        x1 = self.att_CL2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.15, training=self.training)
        x1 = self.att_CL3(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.15, training=self.training)
        x1 = self.att_CL4(x1, edge_index)
        x1 = F.relu(x1)
        x1 = global_mean_pool(x1, batch)
        x1 = self.fc2(x1)
        return x1.squeeze(-1)

def test(model, loader):
    model.eval()
    mse_total = 0  # Initialize the mean squared error sum
    n = 0  # Initialize the count of samples
    original_labels = []
    y_preds = []
    y_trues = []
    for i, data in enumerate(loader):
        out = model(data)
        loss = mse_loss(out, data.y.float())
        mse_total += loss.item() * data.num_graphs
        n += data.num_graphs
        y_preds.extend(out.detach().cpu().numpy().tolist())
        y_trues.extend(data.y.cpu().numpy().tolist())
        original_labels.append(test_labels[i])  # Assuming test_labels is globally accessible or passed to the function

    mse_avg = mse_total / n  # Calculate average MSE
    return mse_avg, original_labels, y_preds, y_trues

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        #print(data.y)
        #print(out)
        #print(data.y.shape)
        #print(out.shape)
        loss = criterion(out, data.y.float())
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        out = model(data)
        loss = criterion(out, data.y.float())
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


criterion = RMSELoss()
model = mGAT(num_node_features=54, hidden_channels=hc)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




model.load_state_dict(torch.load('model_task1.pth'), strict=False)


train_losses = []
val_losses = []














for epoch in range(200):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)

    val_loss = validate(model, val_loader, criterion)
    val_losses.append(val_loss)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')



test_mse, original_labels, y_preds, y_trues = test(model, test_loader)
print(f'Test MSE: {test_mse:.4f}')
# Save the model, need to specify freeze later
torch.save(model.state_dict(), 'model_task2.pth')


# plot training and validation losses for classification
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train and Validation Loss over epochs (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#dataframe for save
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)


#dataframe for save
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

_, _, y_preds, y_trues = test(model, test_loader)

test_predictions = np.array(y_preds)
test_targets = np.array(y_trues)


# Plot the test predictions
plt.figure(figsize=(8, 6))
plt.scatter(test_targets, test_predictions, label='Test', alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Test Data')
plt.legend()
plt.show()

Unnormed_test_predictions = 10**((((test_predictions))*(4+0.744727494896694))-0.744727494896694)
Unnormed_test_targets = 10**((((test_targets))*(4+0.744727494896694))-0.744727494896694)

test_percent_error = np.mean(np.abs(100 * (Unnormed_test_targets.flatten() - Unnormed_test_predictions.flatten()) / Unnormed_test_targets.flatten()))

print('Mean Absolute Percent Error for Testing Data:', test_percent_error)


df['Unnormed_test_predictions'] = Unnormed_test_predictions
df['Unnormed_test_targets'] = Unnormed_test_targets













for epoch in range(200):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)

    val_loss = validate(model, val_loader, criterion)
    val_losses.append(val_loss)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')



test_mse, original_labels, y_preds, y_trues = test(model, test_loader)
print(f'Test MSE: {test_mse:.4f}')
# Save the model, need to specify freeze later
torch.save(model.state_dict(), 'model_task2.pth')


# plot training and validation losses for classification
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train and Validation Loss over epochs (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#dataframe for save
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)


#dataframe for save
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

_, _, y_preds, y_trues = test(model, test_loader)

test_predictions = np.array(y_preds)
test_targets = np.array(y_trues)


# Plot the test predictions
plt.figure(figsize=(8, 6))
plt.scatter(test_targets, test_predictions, label='Test', alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Test Data')
plt.legend()
plt.show()

Unnormed_test_predictions = 10**((((test_predictions))*(4+0.744727494896694))-0.744727494896694)
Unnormed_test_targets = 10**((((test_targets))*(4+0.744727494896694))-0.744727494896694)

test_percent_error = np.mean(np.abs(100 * (Unnormed_test_targets.flatten() - Unnormed_test_predictions.flatten()) / Unnormed_test_targets.flatten()))

print('Mean Absolute Percent Error for Testing Data:', test_percent_error)


df['Unnormed_test_predictions'] = Unnormed_test_predictions
df['Unnormed_test_targets'] = Unnormed_test_targets


print(df)





def save_to_excel(df, foldnumber):
    if 1 <= foldnumber <= 11:  # Only allow fold numbers from 1 to 6
        filename = f"fold{foldnumber}SR.xlsx"  # Generate filename
        df.to_excel(filename, index=False)  # Save DataFrame to an Excel file
        # Assuming a Google Colab environment; use the command to copy the file
        !cp {filename} "drive/My Drive/"
    else:
        print("Invalid fold number. Please provide a fold number between 1 and 6.")

# To call the function, you can use something like this:
save_to_excel(df, foldnumber)  # Replace 'foldnumber' with the relevant fold number









#next, onto Task3; this is a classification task, as it seeks to predict potency classification (hi/lo). You can add more classes if you would like. Tailor to your own dataset





"""
print("Training set size (SMILES): ", smiles_train.shape)
print("Validation set size (SMILES): ", smiles_val.shape)
print("Training set size (y): ", y_train.shape)
print("Validation set size (y): ", y_val.shape)
print("Training set size (y2): ", y2_train.shape)
print("Validation set size (y2): ", y3_val.shape)
print("Training set size (name): ", Name_train.shape)
print("Validation set size (name): ", Name_val.shape)


invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

print(f'Found {len(invalid_smiles)} invalid SMILES strings.')
# Get the first SMILES string in the training set
first_smiles_train = smiles_train.iloc[0]
# Get the adjacency matrix, edge features, and node features for the first SMILES string in the training set
adjacency_matrix, edge_features, node_features = smiles_to_graph(first_smiles_train)
# Print the adjacency matrix, edge features, and node features
print("Node Features shape:", np.array(node_features).shape)
print("First SMILES String in Training Set:", first_smiles_train)
print("Adjacency Matrix:")
print(adjacency_matrix)
print("\nEdge Features:")
print(edge_features)
print("\nNode Features:")
print(node_features)





invalid_smiles = []

for smiles in x:  # Assuming x is your list of SMILES strings
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    if adjacency_matrix is None or edge_features is None or node_features is None:
        invalid_smiles.append(smiles)

max_nodes = 0
for i, smiles in enumerate(smiles_train):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

for i, smiles in enumerate(smiles_test):
    _, _, node_features = smiles_to_graph(smiles)
    max_nodes = max(max_nodes, len(node_features))

print("\nmax_nodes:")
print(max_nodes)

"""

permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Mg', "I"]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(permitted_list_of_atoms).reshape(-1, 1)) # reshape for 1 feature
def atom_features(atom):
    ComputeGasteigerCharges(atom.GetOwningMol())  # Compute Gasteiger charges for the molecule
    # Get the one-hot encoded representation of the atom symbol
    symbol_one_hot = encoder.transform(np.array(atom.GetSymbol()).reshape(-1, 1))
    # Decompose the one-hot encoded symbol into multiple 1D arrays
    symbol_C, symbol_N, symbol_O, symbol_S, symbol_F, symbol_P, symbol_Cl, symbol_Br, symbol_Mg, symbol_I= symbol_one_hot.T
    degree = np.eye(6)[atom.GetDegree()] if atom.GetDegree() < 6 else np.array([0, 0, 0, 0, 0, 1])
    total_num_hs = np.eye(5)[atom.GetTotalNumHs()] if atom.GetTotalNumHs() < 5 else np.array([0, 0, 0, 0, 1])
    implicit_valence = np.eye(6)[atom.GetImplicitValence()] if atom.GetImplicitValence() < 6 else np.array([0, 0, 0, 0, 0, 1])
    is_aromatic = np.array([atom.GetIsAromatic()], dtype=np.float32)
    gasteiger_charge = np.array([float(atom.GetProp('_GasteigerCharge'))], dtype=np.float32) #Feature id is 28
    hybridization = np.eye(7)[atom.GetHybridization().real] if atom.GetHybridization().real < 7 else np.array([0, 0, 0, 0, 0, 0, 1])
    explicit_valence = np.eye(10)[atom.GetExplicitValence()] if atom.GetExplicitValence() < 10 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    formal_charge = np.eye(5)[atom.GetFormalCharge()+2] if -2<=atom.GetFormalCharge()<=2 else np.array([0, 0, 0, 0, 0, 1])
    atomic_mass_scaled = [float((atom.GetMass())/35)] #Feature id is 51
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())-1.5)/(1.7-1.5))] #Feature id is 52
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())-0.64)/(0.68-0.64))] #Feature id is 53
    return np.concatenate([symbol_C.ravel(), symbol_N.ravel(), symbol_O.ravel(), symbol_S.ravel(), symbol_F.ravel(), symbol_P.ravel(), symbol_Cl.ravel(), symbol_Br.ravel(), symbol_Mg.ravel(), symbol_I.ravel(), degree, total_num_hs, implicit_valence, is_aromatic, gasteiger_charge, hybridization, explicit_valence, formal_charge, atomic_mass_scaled, vdw_radius_scaled, covalent_radius_scaled])

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))  # use atom_features function here
    # Get adjacency matrix and edge features (bond types)
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Uncomment these lines to fill the adjacency matrix
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        edge_features.append((i, j))
    return adjacency_matrix, edge_features, node_features

data= pd.read_excel("/content/.....xlsx")



#foldnumber = 1
#specified above
train = data[data["fold"]!=foldnumber]
test = data[data["fold"]==foldnumber]

Name_train = train['Original Label']
Name_test = test['Original Label']

x = data.pop('SMILES')
smiles = x
smiles_train = train['SMILES']
print("smiles_train.shape", smiles_train.shape)
smiles_test = test['SMILES']
print("smiles_test.shape", smiles_test.shape)

y = data.pop('twoclasssplitstatic')
y_train = train['twoclasssplitstatic']
y_test = test['twoclasssplitstatic']

"""
y = data.pop('fourclasssplitstatic')
y_train = train['fourclasssplitstatic']
y_test = test['fourclasssplitstatic']
"""

def one_hot_encode_y(y):
    # Map y values to their one-hot encodings
    mapping = {
        0: [1, 0],
        1: [0, 1]
    }
    return torch.tensor(mapping[y], dtype=torch.float)

smiles_train, smiles_val, y_train, y_val = train_test_split(smiles_train, y_train, test_size=0.053, random_state=902203)
max_nodes = 40 # with mGlu2&3 dataset, max nodes is actually 40
num_node_features = 54


test_data = []
test_labels = []  #for 'Original Label'
for i, smiles in enumerate(smiles_test):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_test.iloc[i]))
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    test_data.append(data)
    test_labels.append(Name_test.iloc[i])

val_data = []
for i, smiles in enumerate(smiles_val):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_val.iloc[i]))  # Convert to integer and then pass to one_hot_encode_y()
    #print(y.shape)
    #print(x.shape)
    #old data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    val_data.append(data)

train_data = []
for i, smiles in enumerate(smiles_train):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(smiles)
    node_features = np.array(node_features)  # Convert to numpy array
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = one_hot_encode_y(int(y_train.iloc[i]))
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y.unsqueeze(0))
    train_data.append(data)



train_loader = DataLoader(train_data, batch_size=250, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)












#argmax frozen
#S>R>C

class mGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(mGAT, self).__init__()
        self.att_CL1c = GATConv(num_node_features, hidden_channels)
        self.att_CL2c = GATConv(hidden_channels, hidden_channels)
        self.att_CL3c = GATConv(hidden_channels, hidden_channels)
        self.att_CL1 = GATConv(num_node_features, hidden_channels)
        self.att_CL2 = GATConv(hidden_channels, hidden_channels)
        self.att_CL3 = GATConv(hidden_channels, hidden_channels)
        self.att_CL4 = GATConv(hidden_channels, hidden_channels)
        #self.att_CL5 = GATConv(2*hidden_channels, hidden_channels)
        self.mlpdropout = nn.Dropout(p=0.1)
        self.mlp_layer1 = nn.Linear(hidden_channels*2, 32)  # Assume in_features is the size of x1c
        #self.mlp_layer2 = nn.Linear(128, 32)
        self.fc1 = Linear(hidden_channels, 3)  # Classification
        self.fc2 = Linear(hidden_channels, 1)  # Regression. Try 2 and 1 class. problem with 2 class is that for correct calculation, ytrue needs to be known, thus giving algo an unfair advantage
        self.fc4 = Linear(32, 2)  # 2 class IC50, low vs high. no unfair advantage this way. Can try 4 class too


    def forward(self, data):
        #what to do about shared layers
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.att_CL1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.att_CL2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.att_CL3(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = self.att_CL4(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.05, training=self.training)
        x1 = global_mean_pool(x1, batch)

        x2 = self.att_CL1c(x, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.15, training=self.training)
        x2 = self.att_CL2c(x2, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.15, training=self.training)
        x2 = self.att_CL3c(x2, edge_index)
        x2 = F.relu(x2)
        x2 = global_mean_pool(x2, batch)

        x1c = torch.cat([x2, x1], dim=-1)
        x1c = self.mlpdropout(x1c)
        x1c = F.relu(self.mlp_layer1(x1c))
        #x1c = self.mlpdropout(x1c)
        #x1c = F.relu(self.mlp_layer2(x1c))
        x1c = self.fc4(x1c)
        return x1c


def test(model, loader, test_labels):
    model.eval()
    correct = 0
    y_preds = []
    y_trues = []
    original_labels = []
    for i, data in enumerate(loader):
        out1 = model(data)
        _, pred = torch.max(out1, 1)  # Get the index of the max logi which schould be indicative of predicted class
        correct += (pred == torch.argmax(data.y, dim=1)).sum().item()
        pred_labels = pred
        y_preds.extend(pred_labels.tolist())
        y_trues.extend(data.y.tolist())
        original_labels.append(test_labels[i])

    accuracy = correct / (len(loader.dataset)) #nomultiplier needed as argmax takes care of the cass number issue
    return accuracy, original_labels, y_preds, y_trues

def train(model, loader, optimizer, criterion1):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out1 = model(data)
        #print("Shape of data.y:", data.y.shape)
        #print("Shape of out1:", out1.shape)
        loss1 = criterion1(out1, data.y.float())
        total_loss += loss1.item() * data.num_graphs
        loss1.backward()
        optimizer.step()
    return total_loss / len(loader.dataset)
def validate(model, loader, criterion1):
    model.eval()
    total_loss1 = 0
    correct = 0
    for data in loader:
        out1 = model(data)
        loss1 = criterion1(out1, data.y.float())
        total_loss1 += loss1.item() * data.num_graphs
        _, pred_probs  = torch.max(out1, 1)  # Get the index of the max logi which schould be indicative of predicted class
        correct += (pred_probs  == torch.argmax(data.y, dim=1)).sum().item()
        pred_labels = pred_probs
    accuracy = correct / (len(loader.dataset))
    return total_loss1 / len(loader.dataset), accuracy







model.load_state_dict(torch.load('model_task2.pth'), strict=False)


from collections import OrderedDict
def remap_name(old_name):
    old_name = old_name.replace('att_CL1c', 'att_CL1')
    old_name = old_name.replace('att_CL2c', 'att_CL2')
    return old_name.replace('att_CL3c', 'att_CL3')
saved_state_dict = torch.load('model_task1.pth')
new_state_dict = OrderedDict()
for old_name, param in saved_state_dict.items():
    new_name = remap_name(old_name)  # Remap the name of each parameter.
    new_state_dict[new_name] = param  # Add the parameter to the new state dictionary.
# Load the new state dictionary into the model.
model.load_state_dict(new_state_dict, strict=False)


#get prepped to save stuff
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
criterion1 = BCEWithLogitsLoss()
model = mGAT(num_node_features=54, hidden_channels=hc)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
















for epoch in range(400):

    if epoch == 0:
        for name, param in model.named_parameters():
            if remap_name(name) in new_state_dict:
                param.requires_grad = False  # Freeze

    if epoch == 100:
        for name, param in model.named_parameters():
            param.requires_grad = True  # Unfreeze
    train_loss = train(model, train_loader, optimizer, criterion1)
    train_losses.append(train_loss)
    train_accuracy, _, _, _ = test(model, train_loader, test_labels)
    train_accuracies.append(train_accuracy)

    val_loss1, val_accuracy = validate(model, val_loader, criterion1)
    val_losses.append(val_loss1)
    val_accuracies.append(val_accuracy)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss1:.4f}, Val Acc: {val_accuracy:.4f}')

    accuracy, original_labels, y_preds, y_trues = test(model, test_loader, test_labels)
    print(f'Test Acc after classification: {accuracy:.4f}')





# Save the model, need to specify freeze later
torch.save(model.state_dict(), 'model_task3.pth')


# plot training and validation losses for classification
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train and Validation Loss over epochs (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#dataframe for save
# Create a dictionary to hold the values
data_dict = {
    'Original_Labels': original_labels,
    'y_pred': y_preds,
    'y_true': y_trues
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

#print(df)


def save_to_excel(df, foldnumber):
    if 1 <= foldnumber <= 11:  # Only allow fold numbers from 1 to 6
        filename = f"fold{foldnumber}SRCt.xlsx"  # Generate filename
        df.to_excel(filename, index=False)  # Save DataFrame to an Excel file
        # Assuming a Google Colab environment; use the command to copy the file
        !cp {filename} "drive/My Drive/"
    else:
        print("Invalid fold number. Please provide a fold number between 1 and 6.")

# To call the function, you can use something like this:
save_to_excel(df, foldnumber)  # Replace 'foldnumber' with the relevant fold number
