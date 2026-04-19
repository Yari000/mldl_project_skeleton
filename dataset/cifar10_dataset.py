# quick and dirty just for this notebook
image_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=image_transforms
    ),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=False,
        transform=image_transforms
    ),
    batch_size=1000,
    shuffle=True,
)

# Retrieve the image size and the number of color channels
x, yy = next(iter(train_loader))

n_channels = x.shape[1]
input_size_w = x.shape[2]
input_size_h = x.shape[3]
input_size = input_size_w * input_size_h

# Specify the number of classes in CIFAR10
output_size = yy.max().item() + 1  # there are 10 classes
output_classes = ('plane', 'car', 'bird', 'cat', 'deer',


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, data_path="./data"):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.CIFAR10(data_path, train=False, transform=transform),
        batch_size=1000,
        shuffle=False,
    )

    return train_loader, test_loader
                  'dog', 'frog', 'horse', 'ship', 'truck')
