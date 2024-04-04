
lr = 0.01

model = nn.Sequential()
model.add_module('cnn1', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)) # 16@26X26
model.add_module('activation1', nn.ReLU())
model.add_module('pool1', nn.AvgPool2d(kernel_size=2)) # 16@13X13
model.add_module('cnn2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)) # 32@11X11
model.add_module('activation2', nn.ReLU())
model.add_module('pool2', nn.AvgPool2d(kernel_size=2)) # 32@5X5
# model.add_module('cnn3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)) # 64@3X3
# model.add_module('activation3', nn.ReLU())
# model.add_module('pool3', nn.AvgPool2d(kernel_size=2)) # 64@1X1
model.add_module('flatten', nn.Flatten())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('fc', nn.Linear(32*5*5, 42))

loss_fn = nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)
