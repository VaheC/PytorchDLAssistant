
def create_train_step_fn(model, loss_fn, optimizer):
    def output_train_step_loss(X, y):
        model.train()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return output_train_step_loss

def create_valid_step_fn(model, loss_fn):
    def output_valid_step_loss(X, y):
        model.eval()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        return loss.item()
    return output_valid_step_loss

def get_mini_batch_loss(device, data_loader, step_fn):
    mini_batch_losses = []
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        batch_loss = step_fn(X_batch, y_batch)
        mini_batch_losses.append(batch_loss)
    
    return np.mean(mini_batch_losses)

lr = 0.01

torch.manual_seed(13)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(nn.Linear(5, 1)).to(device)

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)

train_step_fn = create_train_step_fn(model, loss_fn, optimizer)
valid_step_fn = create_valid_step_fn(model, loss_fn)

tensorboard_writer = SummaryWriter('summary/simple_linear_reg')
