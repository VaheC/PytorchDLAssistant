
resnet18_model = resnet18(weights=ResNet18_Weights.DEFAULT)

def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

freeze_model(resnet18_model)

resnet18_model.fc = nn.Identity()

# resnet18_model.fc = nn.Linear(2048, 42)
# resnet18_model

def extract_model_features(model, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = None
    features = None

    for i, (X, y) in enumerate(data_loader):
        model.eval()
        output = model(X.to(device))
        if i == 0:
            labels = y.cpu()
            features = output.detach().cpu()
        else:
            labels = torch.cat([labels, y.cpu()])
            features = torch.cat([features, output.detach.cpu()])

    dataset = TensorDataset(features, labels)

    return dataset

TRAIN_PATH = Path(r"C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\portfolio/\PytorchDLAssistant/\data/\train")
VALID_PATH = Path(r"C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\portfolio/\PytorchDLAssistant/\data/\valid")

normalizer = Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]
)

data_transform = Compose([Resize((28, 28)), ToTensor(), normalizer])

train_dataset = ImageFolder(root=TRAIN_PATH, transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

valid_dataset = ImageFolder(root=VALID_PATH, transform=data_transform)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=16)

train_feat_dataset = extract_model_features(resnet18_model, train_loader)
train_loader = DataLoader(dataset=train_feat_dataset, batch_size=16, shuffle=True)

valid_feat_dataset = extract_model_features(resnet18_model, valid_loader)
valid_loader = DataLoader(dataset=valid_feat_dataset, batch_size=16)
