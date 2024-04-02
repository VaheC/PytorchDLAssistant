
n_epochs = 100

train_losses = []
valid_losses = []

for epoch in range(n_epochs):
    train_loss = get_mini_batch_loss(device=device, data_loader=train_loader, step_fn=train_step_fn)
    train_losses.append(train_loss)

    with torch.no_grad():
        valid_loss = get_mini_batch_loss(device=device, data_loader=valid_loader, step_fn=valid_step_fn)
        valid_losses.append(valid_loss)

    tensorboard_writer.add_scalars(
        main_tag = 'losses',
        tag_scalar_dict = {
            'train': train_loss,
            'validation': valid_loss
        },
        global_step = epoch
    )

tensorboard_writer.close()
