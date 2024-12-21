###> For interpretability, first set 100% of data as training. Define the folds that will be associated with training.
###> Now have your test subjects (new ligands) be assigned a fold number that is outside of training range. This way, they will be excluded from the training set
###> Note that the ligands that were used to confirm the validity of intepretable gradients were included in training, and then re-evaluated.

###> Load the model, but save the model state

# Load your dataset
data= pd.read_excel("/content/......xlsx")
foldnumber = 11 #list fold number outside range; this will effectively bypass "test"
train = data[data["fold"] != foldnumber]

# Extract SMILES and Labels from the training set
smiles_train = train['SMILES']
Name_train = train['Original Label']

#load pth
path1 = "/content/.....pth" #wherever you saved model path
model.load_state_dict(torch.load(path1))
model.eval()






###> BELOW IS INTERPRETABILITY for Task1



def smiles_to_sel(smiles_string):
    return smiles_string
def sel_to_sample_data(sel, max_nodes):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(sel)
    node_features = np.array(node_features)
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    return data

sel = smiles_to_sel("c1cc(F)cc(Cl)c1-c(cc(n2)C(=O)N)c(c23)ccc(c3)CN4C(=O)CCC4=O")
sample_data = sel_to_sample_data(sel, max_nodes)


def print_grad(module, grad_input, grad_output):
    print('Inside ' + module.__class__.__name__ + ' backward')
    print('Inside class:' + module.__class__.__name__)
    print('grad_input: ', grad_input)
    print('grad_output: ', grad_output)

#hook1 = model.att4.register_backward_hook(print_grad)
#register_full_backward_hook maybe later
sample_data.x.requires_grad_()

output = model(sample_data)
print(output.shape)
print(output)
#shape is torch.Size([1, 2])
output.backward(torch.tensor([[1.0, 0.0, 0.0]]))
###> [1.0 ,0.0, 0.0] looks at the first node
###> [+, -, -] means mGlu2
###>  However, make sure id is correctly mapped to your own use case
node_gradients = sample_data.x.grad
print(node_gradients.shape)


# Compute the average gradient for each feature across all nodes
#each entry corresponds to the average gradient of a particular feature across all nodes (atoms) in one sample molecule
#This helps understand which features, on average, have the most influence on the model's decision for this particular sample.
average53_gradients = node_gradients.mean(axis=0).detach().numpy()
print(average53_gradients.shape)
plt.figure(figsize=(12, 6))
plt.bar(range(54), average53_gradients)
plt.xlabel('Feature Index')
plt.ylabel('Average Gradient')
plt.title('Average Gradient across Features')
plt.show()


#This tells you which nodes (atoms) are, on average, most influential in the model's decision for this sample.
average40_gradients = node_gradients.mean(axis=1).detach().numpy()
print(average40_gradients.shape)
plt.figure(figsize=(12, 6))
plt.bar(range(40), average40_gradients)
plt.xlabel('Feature Index')
plt.ylabel('Average Gradient')
plt.title('Average Gradient per Feature across Nodes')
plt.show()


#select_feature_gradients = node_gradients[:, 27].detach().numpy()
select_feature_gradients = average40_gradients
# Plotting
plt.figure(figsize=(12, 6))
plt.bar(range(40), select_feature_gradients)
plt.xlabel('Node Index')
plt.ylabel('Gradient')
plt.title('Gradient of the Select Feature across Nodes')
plt.show()


def visualize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    drawer.drawOptions().includeAtomNumbers = True
    drawer.drawOptions().dotsPerAngstrom = 400
    drawer.drawOptions().bondLineWidth = 1.5
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    plt.axis('off')
    plt.show()

    return img_array  # 

img = visualize_molecule(sel)




def normalize_gradient(gradient):
    min_val = np.min(gradient)
    max_val = np.max(gradient)
    return (gradient - min_val) / (max_val - min_val)

def visualize_gradients_on_molecule_with_colorbar(smiles, select_feature_gradients):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)
    norm_gradients = normalize_gradient(select_feature_gradients)
    colormap = cm.get_cmap('bwr')
    colors = [colormap(x) for x in norm_gradients]
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber', str(i))

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().includeAtomNumbers = True
    drawer.drawOptions().dotsPerAngstrom = 200
    drawer.drawOptions().bondLineWidth = 1.5

    atom_highlights = {i: colors[i] for i in range(len(colors))}
    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightAtomColors=atom_highlights)
    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(img_array, cmap='bwr', vmin=min_value, vmax=max_value)
    cbar = fig.colorbar(cax, orientation='vertical', pad=0.05)
    cbar.set_label('Gradient magnitude', rotation=270, labelpad=15)
    plt.axis('off')
    plt.show()


    return img_data
img_data = visualize_gradients_on_molecule_with_colorbar(sel, select_feature_gradients)


def visualize_gradients_on_molecule_with_colorbar(smiles, select_feature_gradients):
    mol = AllChem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)

    norm_gradients = normalize_gradient(select_feature_gradients)
    red_white_cmap = LinearSegmentedColormap.from_list('red_white', [(1, 1, 1), (1, 0, 0)], N=256)

    colors = [red_white_cmap(x) for x in norm_gradients]
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()
    opts.includeAtomNumbers = False
    opts.atomLabelFontSize = 50
    opts.dotsPerAngstrom = 100
    opts.bondLineWidth = 1.5

    # have gradient values becomeatom labels
    for i in range(len(select_feature_gradients)):
        opts.atomLabels[i] = "{:.2f}".format(select_feature_gradients[i])

    #coloring
    atom_highlights = {i: colors[i] for i in range(len(colors))}
    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightAtomColors=atom_highlights)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(img_array, cmap=red_white_cmap, vmin=min_value, vmax=max_value)
    cbar = fig.colorbar(cax, orientation='vertical', pad=0.05)
    cbar.set_label('Gradient magnitude', rotation=270, labelpad=15)
    plt.axis('off')
    plt.show()

    return img_data

img_data = visualize_gradients_on_molecule_with_colorbar(sel, select_feature_gradients)



def visualize_gradients_on_molecule(smiles, select_feature_gradients):
    mol = AllChem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)

    norm_gradients = normalize_gradient(select_feature_gradients)


    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    opts = drawer.drawOptions()
    opts.includeAtomNumbers = False
    opts.atomLabelFontSize = 50
    opts.dotsPerAngstrom = 400
    opts.bondLineWidth = 1.5

    # have gradient values becomeatom labels
    for i in range(len(select_feature_gradients)):
        opts.atomLabels[i] = "{:.2f}".format(select_feature_gradients[i])

    #coloring
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    plt.axis('off')
    plt.show()

    return img_data

img_data = visualize_gradients_on_molecule(sel, select_feature_gradients)


















#load pth for Task 3 classification; similar to above
path2 = "/content/....pth"
model.load_state_dict(torch.load(path2), strict=False)
model.eval()



###ABOVE IS MODEL LOAD; BELOW IS INTERPRETABILITY


def smiles_to_sel(smiles_string):
    return smiles_string
def sel_to_sample_data(sel, max_nodes):
    adjacency_matrix, edge_features, node_features = smiles_to_graph(sel)
    node_features = np.array(node_features)
    node_features = np.pad(node_features, ((0, max_nodes - node_features.shape[0]), (0, 0)), 'constant')
    edge_index = torch.tensor(edge_features, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    return data

sel = smiles_to_sel("c1ncncc1CCc(cc2)cc(c23)nc(C#N)cc3-c4ccc(F)cc4")
sample_data = sel_to_sample_data(sel, max_nodes)


def print_grad(module, grad_input, grad_output):
    print('Inside ' + module.__class__.__name__ + ' backward')
    print('Inside class:' + module.__class__.__name__)
    print('grad_input: ', grad_input)
    print('grad_output: ', grad_output)

#hook1 = model.att4.register_backward_hook(print_grad)
#register_full_backward_hook maybe later
sample_data.x.requires_grad_()

output = model(sample_data)
print(output.shape)
print(output)
#shape is torch.Size([1, 2])
output.backward(torch.tensor([[1.0, 0.0]]))
###> [1.0, 0.0] looks at the first node
###> [+, -] means mGlu2
node_gradients = sample_data.x.grad
print(node_gradients.shape)


# Compute the average gradient for each feature across all nodes
#each entry corresponds to the average gradient of a particular feature across all nodes (atoms) in one sample molecule
#This helps understand which features, on average, have the most influence on the model's decision for this particular sample.
average53_gradients = node_gradients.mean(axis=0).detach().numpy()
print(average53_gradients.shape)
plt.figure(figsize=(12, 6))
plt.bar(range(54), average53_gradients)
plt.xlabel('Feature Index')
plt.ylabel('Average Gradient')
plt.title('Average Gradient across Features')
plt.show()


#This tells you which nodes (atoms) are, on average, most influential in the model's decision for this sample.
average40_gradients = node_gradients.mean(axis=1).detach().numpy()
print(average40_gradients.shape)
plt.figure(figsize=(12, 6))
plt.bar(range(40), average40_gradients)
plt.xlabel('Feature Index')
plt.ylabel('Average Gradient')
plt.title('Average Gradient per Feature across Nodes')
plt.show()


#select_feature_gradients = node_gradients[:, 27].detach().numpy()
select_feature_gradients = average40_gradients
# Plotting
plt.figure(figsize=(12, 6))
plt.bar(range(40), select_feature_gradients)
plt.xlabel('Node Index')
plt.ylabel('Gradient')
plt.title('Gradient of the Select Feature across Nodes')
plt.show()


def visualize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)


    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    drawer.drawOptions().includeAtomNumbers = True
    drawer.drawOptions().dotsPerAngstrom = 400
    drawer.drawOptions().bondLineWidth = 1.5

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    plt.axis('off')
    plt.show()

    return img_array 

img = visualize_molecule(sel)




def normalize_gradient(gradient):
    min_val = np.min(gradient)
    max_val = np.max(gradient)
    return (gradient - min_val) / (max_val - min_val)

def visualize_gradients_on_molecule_with_colorbar(smiles, select_feature_gradients):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)
    norm_gradients = normalize_gradient(select_feature_gradients)
    colormap = cm.get_cmap('bwr')
    colors = [colormap(x) for x in norm_gradients]
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('molAtomMapNumber', str(i))

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().includeAtomNumbers = True
    drawer.drawOptions().dotsPerAngstrom = 200
    drawer.drawOptions().bondLineWidth = 1.5

    atom_highlights = {i: colors[i] for i in range(len(colors))}
    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightAtomColors=atom_highlights)
    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(img_array, cmap='bwr', vmin=min_value, vmax=max_value)
    cbar = fig.colorbar(cax, orientation='vertical', pad=0.05)
    cbar.set_label('Gradient magnitude', rotation=270, labelpad=15)
    plt.axis('off')
    plt.show()


    return img_data
img_data = visualize_gradients_on_molecule_with_colorbar(sel, select_feature_gradients)


def visualize_gradients_on_molecule_with_colorbar(smiles, select_feature_gradients):
    mol = AllChem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)

    norm_gradients = normalize_gradient(select_feature_gradients)

    red_white_cmap = LinearSegmentedColormap.from_list('red_white', [(1, 1, 1), (1, 0, 0)], N=256)

    colors = [red_white_cmap(x) for x in norm_gradients]
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()
    opts.includeAtomNumbers = False
    opts.atomLabelFontSize = 50
    opts.dotsPerAngstrom = 100
    opts.bondLineWidth = 1.5
    for i in range(len(select_feature_gradients)):
        opts.atomLabels[i] = "{:.2f}".format(select_feature_gradients[i])
    atom_highlights = {i: colors[i] for i in range(len(colors))}
    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightAtomColors=atom_highlights)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(img_array, cmap=red_white_cmap, vmin=min_value, vmax=max_value)
    cbar = fig.colorbar(cax, orientation='vertical', pad=0.05)
    cbar.set_label('Gradient magnitude', rotation=270, labelpad=15)
    plt.axis('off')
    plt.show()

    return img_data

img_data = visualize_gradients_on_molecule_with_colorbar(sel, select_feature_gradients)



def visualize_gradients_on_molecule(smiles, select_feature_gradients):
    mol = AllChem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    max_value = np.max(select_feature_gradients)
    min_value = np.min(select_feature_gradients)

    norm_gradients = normalize_gradient(select_feature_gradients)

    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    opts = drawer.drawOptions()
    opts.includeAtomNumbers = False
    opts.atomLabelFontSize = 50
    opts.dotsPerAngstrom = 400
    opts.bondLineWidth = 1.5

    for i in range(len(select_feature_gradients)):
        opts.atomLabels[i] = "{:.2f}".format(select_feature_gradients[i])

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    image = PILImage.open(io.BytesIO(img_data))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    plt.axis('off')
    plt.show()

    return img_data

img_data = visualize_gradients_on_molecule(sel, select_feature_gradients)

