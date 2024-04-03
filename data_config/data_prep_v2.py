
class CustomDataset(Dataset):

    def __init__(self, X, y, X_transform=None):
        self.X = X
        self.y = y 
        self.X_transform = X_transform

    def __getitem__(self, index):
        X = self.X[index]

        if self.X_transform is not None:
            X = self.X_transform(X)

        return X, self.y[index]

    def __len__(self):
        return len(self.X)

def index_splitter(data_size, splits, seed=13):
    idxs = torch.arange(data_size)
    splits_tensor = torch.as_tensor(splits)
    multiplier = data_size / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    diff = data_size - splits_tensor.sum()
    splits_tensor[0] += diff
    torch.manual_seed(seed)
    return random_split(idxs, splits_tensor)

def create_balanced_sampler(y):

    classes, counts = y.unique(return_counts=True)
    weights = 1 / counts.float()
    sample_weights = weights[y.squeeze().long()]

    generator = torch.Generator()

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        number_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )

    return weighted_sampler
