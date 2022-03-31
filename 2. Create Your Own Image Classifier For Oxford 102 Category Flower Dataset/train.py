from datafunc import process_data
from input_arg import train_input_args
import model_func
import torch


def start_training(epochs, use_gpu, dataloaders):

    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    elif use_gpu:
        print('CUDA not available')
    model.to(device)
    epochs = epochs

    for e in range(epochs):

        train_loss = 0
        valid_loss = 0
        accuracy = 0

        model.train()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_out = model(images)
            loss = criterion(log_out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            model.eval()
            for images, labels in dataloaders['valid']:
                with torch.no_grad():
                    images, labels = images.to(device), labels.to(device)

                    log_out = model(images)
                    loss = criterion(log_out, labels)

                    out = torch.exp(log_out)
                    top_prop, top_class = out.topk(1, dim=1)
                    correct_out = top_class == labels.view(*top_class.shape)

                    valid_loss += loss.item()
                    accuracy += torch.mean(correct_out.float()).item()

            print(f'Epoch: {e+1:2}/{epochs}.. '
                  f'Train loss: {train_loss/len(dataloaders["train"]):.3f}.. '
                  f'Validation loss: {valid_loss/len(dataloaders["valid"]):.3f}.. '
                  f'Validation accuracy: {accuracy/len(dataloaders["valid"])*100:.2f}%')


inputs = train_input_args()

dataloaders, class_to_index = process_data(inputs.data_dir)

model, optimizer, criterion = model_func.build(
    inputs.arch, inputs.hidden_units, inputs.learning_rate)

print(f'\n\nStarted training a/an {inputs.arch} model.\n')
start_training(inputs.epochs, inputs.use_gpu, dataloaders)

checkpoint = {'epochs': inputs.epochs,
              'arch': inputs.arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'mapping': class_to_index}

torch.save(checkpoint, inputs.save_dir)
print(f'\nFinshed training\nA checkpoint is saved to {inputs.save_dir}')
