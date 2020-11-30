from datetime import datetime
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch import Tensor


class ModelTrainer:
    """
    Base trainer class containing methods to train modules, plot image grids, loss histories, etc.
    """
    def __init__(self):
        pass
    
    def _make_match_labels(self, batch_size: int) -> Tensor:
        """
        Produces a tensor of consecutive, sequential integers,
        e.g. [0, 1, 2, 3, 4, 5, 6, 7] if batch_size is equal to 8
        """
        return torch.LongTensor(range(batch_size)).cuda()
    
    def _count_parameters(self, model: Module):
        """Counts number of parameters in Module"""
        name = model.__class__.__name__
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model {name} has {num_params} parameters")

    def _make_noise(self, batch_size: int, z_dim: int) -> Tensor:
        """Produces a 2-D Noise matrix with shape [batch_size, z_dim]"""
        return torch.FloatTensor(batch_size, z_dim).normal_(mean=0, std=1).cuda()
    
    def _denormalise_single(self, tensor: Tensor) -> Tensor:
        mean = std = 0.5
        tensor = (tensor * std) + mean
        return tensor

    def _denormalise_multiple(self, tensors: List[Tensor]) -> List[Tensor]:
        return [self._denormalise_single(tensor) for tensor in tensors]

    def _display_image(self, image: Tensor) -> None:
        image = image.detach().cpu()
        return plt.imshow(image.permute(1, 2, 0))

    def _plot_history(self, history: List[float], epoch: int = None, window_size=100, name='loss_hist', folder='generated_images') -> None:
        """
        Plot a list of float values - save it to a folder for display
        If passed a list of lists, will unpack and plot all on the chart
        """
        # Place history inside another list if it is a standalone list
        all_histories = [history] if not isinstance(history[0], list) else history
        for history in all_histories:
            i = 0
            moving_averages = []
            while i < (len(history) - window_size + 1):
                window = history[i : i + window_size]
                window_average = sum(window) / window_size
                moving_averages.append(window_average)
                i += 1
            plt.plot(moving_averages)
        plt.savefig(f"{folder}/epoch_{epoch}-{name}")
        plt.close()

    def _plot_image_grid(self, fake_images: List[Tensor], epoch: int=None, folder='generated_images') -> None:
        '''
        Produces an [n_images, n_images] image grid for evaluation purposes, saves to an image folder
        Params:
            fake_images: list containing 3 tensors, each one at an increasing resolution (64, 128, 256)
            epoch: epoch number
            folder: name of directory to place the generated images into
        '''
        # Find largest square number
        num_images = len(fake_images[0])
        square = next(i for i in range(num_images, 1, -1) if math.sqrt(i) == int(math.sqrt(i)))
        sqrt = int(math.sqrt(square))
        for images in fake_images:
            res = images.shape[-1]
            # plot images
            images = images[:square].detach().cpu()
            (f, axarr) = plt.subplots(sqrt, sqrt)
            counter = 0
            for i in range(sqrt):
                for j in range(sqrt):
                    # define subplot
                    image = images[counter]
                    image = image.permute(1, 2, 0)
                    axarr[i,j].axis('off')
                    axarr[i,j].imshow(image)
                    counter += 1
            # save plot to file
            timenow: str = str(datetime.now()).split('.')[0].replace(':', '-')
            fname = f'{folder}/epoch_{epoch}-{res}x{res}.png' if epoch else f'{folder}/_{res}x{res}-{timenow}.png'
            plt.savefig(fname)
            plt.close()

    def _plot_single_image(self, fake_images: Tensor, epoch: int, folder='generated_images') -> None:
        image = fake_images[0].detach().cpu()
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        res = fake_images.shape[-1]
        fname = f'{folder}/epoch_{epoch}-{res}x{res}_singleImage.png'
        plt.savefig(fname)
        plt.close()

    def _save_weights(self, modules: List[Module], root_folder='saved_weights') -> None:
        """Saves the weights of each module in modules list to a file"""
        for module in modules:
            name = module.__class__.__name__
            path = f"{root_folder}/{name}.pkl"
            torch.save(module.state_dict(), path)
            print(f'Module {name} weights saved to {path}')

    def _load_weights(self, modules: List[Module], root_folder='saved_weights') -> None:
        """Loads weights for all passed modules, sets each module to eval mode"""
        for module in modules:
            name = module.__class__.__name__
            path = f"{root_folder}/{name}.pkl"
            try:
                module.load_state_dict(torch.load(path))
                module.eval()
                print(f'Module {name} weights loaded from {path}')
            except FileNotFoundError:
                print(f'FAILED: Module {name}... weights at path {path} were not found')