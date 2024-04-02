
np.random.seed(42)
X = np.random.randn(1000, 5)

a = np.array([0.2, 0.1, 0.4, -0.7, 0.03]).reshape(-1, 1)
b = -0.9
y = b + X @ a

X_tensor = torch.as_tensor(X).float()
y_tensor = torch.as_tensor(y).float()

dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(X.shape[0] * 0.8)
valid_size = X.shape[0] - train_size

train_data, valid_data = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True
)

valid_loader = DataLoader(
    dataset=valid_data,
    batch_size=32
)
