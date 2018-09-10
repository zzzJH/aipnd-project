import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import utils


def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    print(args)

    # cuda is avaliable
    cuda = utils.isCudaAvaliable(gpu)

    # datasets and dataloader
    datasets = utils.getDatasets(data_dir)
    dataloaders = utils.getDataLoaders(datasets)

    # build network
    model = utils.getModelsByArch(arch)

    # custom model args and classifier
    model = utils.ininModelArgsAndClassifier(model, hidden_units)

    # define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # train model
    steps = 0
    running_loss = 0
    print_every = 20

    if cuda:
        model.cuda()
    else:
        model.cpu()

    print('train start')
    for e in range(epochs):
        model.train()
        for images, labels in iter(dataloaders['train']):
            steps += 1

            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            if cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                test_loss = 0
                for ii, (images, labels) in enumerate(dataloaders['test']):
                    images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)

                    if cuda:
                        images, labels = images.cuda(), labels.cuda()

                    output = model.forward(images)
                    test_loss += criterion(output, labels).data[0]

                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(
                          test_loss/len(dataloaders['test'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))

            running_loss = 0
            model.train()
    print('train finsh')

    # set checkpoint
    if save_dir:
        checkpoint = {
            'epochs': epochs,
            'arch': arch,
            'learning_rate': learning_rate,
            'hidden_units': hidden_units,
            'gpu': gpu,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_to_idx': datasets['train'].class_to_idx
        }
        torch.save(checkpoint, save_dir)
        print('checkpoint save success')


if __name__ == "__main__":
    args = utils.initTrainParse()
    main(args)
