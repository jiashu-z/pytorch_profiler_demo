import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T


def train(model, data, optimizer, criterion, device):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    transform = T.Compose([T.Resize(224),
                           T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    device = torch.device("cuda:0")
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
    criterion = torch.nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    epochs = 500

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True)
    prof.start()

    for step, batch_data in enumerate(train_loader):
        prof.step()
        if step >= epochs:
            break
        train(model, batch_data, optimizer, criterion, device)
    prof.stop()


if __name__ == '__main__':
    main()
