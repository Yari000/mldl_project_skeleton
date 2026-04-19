import torch

from datasets.cifar10_dataset import get_dataloaders
from models.cnn import CNN
from models.fnn import FC2Layer
from utils.train_utils import fit, get_model_optimizer
from utils.misc import count_parameters

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders()

    x, y = next(iter(train_loader))
    input_size = x.shape[2] * x.shape[3]
    n_channels = x.shape[1]
    output_size = 10

    model = CNN(input_size, n_channels, n_feature=9, output_size=output_size)
    model.to(device)

    optimizer = get_model_optimizer(model)

    print("Parameters:", count_parameters(model))

    fit(
        epochs=10,
        train_dl=train_loader,
        test_dl=test_loader,
        model=model,
        opt=optimizer,
        tag="cnn",
        device=device
    )

if __name__ == "__main__":
    main()
