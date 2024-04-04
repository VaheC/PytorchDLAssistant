
TRAIN_PATH = Path(r"C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\portfolio/\PytorchDLAssistant/\data/\train")
VALID_PATH = Path(r"C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\portfolio/\PytorchDLAssistant/\data/\valid")

# if not os.path.exists(VALID_PATH):
#     os.mkdir(VALID_PATH)

# cat_list = os.listdir(TRAIN_PATH)

# for cat in tqdm(cat_list):

#     if not os.path.exists(os.path.join(VALID_PATH, cat)):
#         os.mkdir(os.path.join(VALID_PATH, cat))

#     files_list = os.listdir(os.path.join(TRAIN_PATH, cat))

#     total_files = len(files_list)
#     valid_files = int(total_files*0.2)
#     valid_idxs_list = list(np.random.choice(np.arange(total_files), valid_files, replace=False))

#     for i in valid_idxs_list:
#         shutil.move(
#             os.path.join(TRAIN_PATH, cat, files_list[i]),
#             os.path.join(VALID_PATH, cat, files_list[i])
#         )

temp_data_transform = Compose([Resize((28, 28)), ToTensor()])
temp_dataset = ImageFolder(root=TRAIN_PATH, transform=temp_data_transform)
temp_loader = DataLoader(dataset=temp_dataset, batch_size=16)

normalizer = DLAssistant.create_normalizer(temp_loader)

data_transform = Compose([Resize((28, 28)), ToTensor(), normalizer])

train_dataset = ImageFolder(root=TRAIN_PATH, transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

valid_dataset = ImageFolder(root=VALID_PATH, transform=data_transform)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=16)
