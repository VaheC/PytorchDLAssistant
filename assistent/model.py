class DLAssistant(object):
    
    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = None
        self.valid_loader = None
        self.tensorboard_writer = None

        self.train_losses = []
        self.valid_losses = []
        self.total_epochs = 0

        self.train_step_fn = self._create_train_step_fn()
        self.valid_step_fn = self._create_valid_step_fn()

        self.layer_output_dict = {}
        self.hook_handlers_dict = {}

        self.scheduler = None
        self.is_batch_lr_scheduler = False
        self.learning_rates = []

    def to(self, device):

        try:
            self.model = self.model.to(device)
            self.device = device
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            print(f"{device} device is not accessable!!!\n{self.device} is used!!!")

    def set_loaders(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def set_tensorboard(self, name, log_dir='summary'):
        
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        name_suffix = dt.now().strftime('%Y%m%d_%H%M%S')
        full_name = f"{log_dir}/{name}_{name_suffix}"
        self.tensorboard_writer = SummaryWriter(full_name)

    def _create_train_step_fn(self):

        def get_train_loss(X, y):
            self.model.train()
            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        
        return get_train_loss
    
    def _create_valid_step_fn(self):
        
        def get_valid_loss(X, y):
            self.model.eval()
            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, y)
            return loss.item()
        
        return get_valid_loss
    
    def _get_mini_batch_loss(self, validation=False):

        if validation:
            data_loader = self.valid_loader
            step_fn = self.valid_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []

        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_loss = step_fn(X_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

            if not validation:
                self._step_mini_batch_schedulers(i / n_batches)

        return np.mean(mini_batch_losses)
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass
    
    def train(self, n_epochs, seed=42):

        self.total_epochs += 1

        self.set_seed(seed)

        for epoch in tqdm(range(n_epochs)):

            train_loss = self._get_mini_batch_loss()
            self.train_losses.append(train_loss)

            with torch.no_grad():
                valid_loss = self._get_mini_batch_loss(validation=True)
                self.valid_losses.append(valid_loss)

            self._step_epoch_schedulers(valid_loss)

            if self.tensorboard_writer is not None:
                scalar_dict = {'train': train_loss}
                if self.valid_loader is not None:
                    scalar_dict.update({'validation': valid_loss})
                self.tensorboard_writer.add_scalars(
                    main_tag = 'losses',
                    tag_scalar_dict = scalar_dict,
                    global_step = epoch
                )

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()

    def show_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='train', c='blue')
        if self.valid_loader is not None:
            plt.plot(self.valid_losses, label='valid', c='orange')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        return fig
    
    def save_results(self, filepath):
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_epochs': self.total_epochs,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses
        }
        torch.save(checkpoint_dict, filepath)

    def load_states(self, filepath):
        checkpoint_dict = torch.load(filepath)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.total_epochs = checkpoint_dict['total_epochs']
        self.train_losses = checkpoint_dict['train_losses']
        self.valid_losses = checkpoint_dict['valid_losses']

        self.model.train()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.as_tensor(X).float()
        X_tensor = X_tensor.to(self.device)
        pred_y = self.model(X_tensor).detach().cpu().numpy()
        self.model.train()
        return pred_y

    def count_parameters(self):
        parameter_number_list = [p.numel() for p in self.model.parameters() if p.requires_grad]
        return sum(parameter_number_list)

    def attach_hook(self, layers_to_hook_list, hook_fn=None):

        modules_list = list(self.model.named_modules())[1:]
        layer_names_dict = {elem[1]: elem[0] for elem in modules_list}

        self.layer_output_dict = {}

        if hook_fn is None:

            def hook_fn(layer, inputs, outputs):
                layer_name = layer_names_dict[layer]
                output_values = outputs.detach().cpu().numpy()
                if self.layer_output_dict[layer_name] is None:
                    self.layer_output_dict[layer_name] = output_values
                else:
                    self.layer_output_dict[layer_name] = np.concatenate(
                        [self.layer_output_dict[layer_name], output_values]
                    )

        for layer_name, layer in modules_list:
            if layer_name in layers_to_hook_list:
                self.layer_output_dict[layer_name] = None
                self.hook_handlers_dict[layer_name] = layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        for handle in self.hook_handlers_dict.values():
            handle.remove()
        self.hook_handlers_dict = {}

    def show_metric(self, X, y, metric, is_classification=False, threshold=0.5):

        self.model.eval()
        X = torch.as_tensor(X).float()
        y_hat = self.model(X.to(self.device))
        self.model.train()

        if is_classification:
            
            if y_hat.size()[1] > 1:
                _, predicted_class = torch.max(y_hat, 1).detach().cpu().numpy()
            else:
                predicted_class = (torch.sigmoid(y_hat) > threshold).long().cpu().numpy()

            return metric(y, predicted_class)

        else:

            return metric(y, y_hat.detach().cpu().numpy())
    
    @staticmethod
    def apply_fn_over_loader(loader, func, reduce='sum'):

        results = [func(X, y) for X, y in loader]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.mean(axis=0)

        return results

    @staticmethod
    def get_stats_per_channel(images, labels):
        n_samples, n_channels, height, width = images.size()
        flattened_images = images.reshape(n_samples, n_channels, -1)
        means = flattened_images.mean(axis=2)
        stds = flattened_images.std(axis=2)
        sum_means = means.sum(axis=0)
        sum_stds = stds.sum(axis=0)
        n_samples = torch.tensor([n_samples]*n_channels).float()
        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    @staticmethod
    def create_normalizer(loader):
        total_samples, total_means, total_stds = DLAssistant.apply_fn_over_loader(
            loader, DLAssistant.get_stats_per_channel
        )
        norm_mean = total_means / total_samples
        norm_stds = total_stds / total_samples
        return Normalize(norm_mean, norm_stds)

    @staticmethod
    def create_lr_fn(start_lr, end_lr, n_iter, lr_mode='exp'):
        if lr_mode == 'linear':
            lr_factor = (end_lr/start_lr - 1) / n_iter
            def lr_fn(iteration):
                return 1 + lr_factor * iteration
        if lr_mode == 'exp':
            lr_factor = np.log(end_lr/start_lr) / n_iter
            def lr_fn(iteration):
                return np.exp(lr_factor) ** iteration
        return lr_fn

    def lr_range_test(self, data_loader, end_lr, n_iter=100, lr_mode='exp', ewma_param=0.5):

        initial_state_dict = {
            'model': deepcopy(self.model.state_dict()),
            'optimizer': deepcopy(self.optimizer.state_dict())
        }

        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        lr_fn = DLAssistant.create_lr_fn(start_lr, end_lr, n_iter, lr_mode)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        tracking_dict = {'loss': [], 'lr': []}

        current_iteration = 0

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_hat = self.model(X_batch)
            loss = self.loss_fn(y_hat, y_batch)

            tracking_dict['lr'].append(scheduler.get_last_lr()[0])

            if len(tracking_dict['loss']) == 0:
                tracking_dict['loss'].append(loss.item())
            else:
                tracking_dict['loss'].append((1-ewma_param)*tracking_dict['loss'][-1] + ewma_param*loss.item())

            loss.backward()
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()

            current_iteration += 1

            if current_iteration == n_iter:
                break

        self.model.load_state_dict(initial_state_dict['model'])
        self.optimizer.load_state_dict(initial_state_dict['optimizer'])
            
        fig = plt.figure(figsize=(10, 6))
        plt.plot(tracking_dict['lr'], tracking_dict['loss'])
        if lr_mode == 'exp':
            plt.xscale('log')
        plt.xlabel('learning rate')
        plt.ylabel('loss')

        return tracking_dict, fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, scheduler):
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or\
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or\
                isinstance(scheduler, optim.lr_scheduler.Cosine)):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def _step_epoch_schedulers(self, valid_loss):
        if self.scheduler is not None:
            if not self.is_batch_lr_sheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda x: x['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def _step_mini_batch_schedulers(self, frac_epoch):
        if self.scheduler is not None:
            if self.is_batch_lr_sheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda x: x['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)
