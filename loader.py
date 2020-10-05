
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

def get_mnist_data(batch_size: int=8) -> DataLoader:
    '''Creates dataloaders for MNIST data'''
    image_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=image_transforms)
    return DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)

