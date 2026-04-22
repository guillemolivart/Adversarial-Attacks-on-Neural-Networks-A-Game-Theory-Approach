import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random


def set_seed(
    seed=42
) -> None:
    
    """
    Fix random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.
    
    Returns:
        None (This function sets seeds in-place and does not return anything).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cifar10_data(
    batch_size=64,
    data_dir='./data'
) -> tuple:
    
    """
    Load and return CIFAR-10 train and test dataloaders.
    
    Args:
        batch_size (int): Number of samples per batch for training and testing. 
        data_dir (str): Directory where the dataset will be stored or loaded from.

    Returns:
        A tuple containing the trainloader and testloader for CIFAR-10 dataset.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    # Utilitzem data_dir en lloc de './data'
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


def evaluate_accuracy(
    classifier: nn.Module, 
    generator: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    epsilon: float, 
    device: torch.device
) -> tuple:
    
    """
    Evaluate clean and adversarial accuracy of the classifier.

    Args:
        classifier (nn.Module): The Defender's model to evaluate.
        generator (nn.Module): The Attacker's model used to generate adversarial examples. If None, only clean accuracy is evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on (e.g., testloader).
        epsilon (float): The maximum perturbation allowed for adversarial examples.
        device (torch.device): The computation device ('cpu' or 'cuda').

    Returns:
        A tuple containing the clean accuracy and adversarial accuracy as percentages.
    """

    classifier.eval()
    if generator is not None:
        generator.eval()
        
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Clean accuracy
            outputs_clean = classifier(images)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            correct_clean += (predicted_clean == labels).sum().item()
            
            # Adversarial accuracy
            if generator is not None:
                x_adv, _ = generator(images, epsilon=epsilon)
                outputs_adv = classifier(x_adv)
                _, predicted_adv = torch.max(outputs_adv.data, 1)
                correct_adv += (predicted_adv == labels).sum().item()
                
            total += labels.size(0)
            
    acc_clean = 100.0 * correct_clean / total
    acc_adv = 100.0 * correct_adv / total if generator is not None else 0.0
    return acc_clean, acc_adv


def initialize_defenders() -> list:

    """
    Initializes the Defender's strategies (CNNs with different width and depth).

    Returns:
        A list of dictionaries, each containing the name and parameters for a Defender strategy.
    """

    defender_strategies = [
        {'name': 'D1', 'num_blocks': 1, 'base_channels': 16},
        {'name': 'D2', 'num_blocks': 1, 'base_channels': 32},
        {'name': 'D3', 'num_blocks': 1, 'base_channels': 64},
        {'name': 'D4', 'num_blocks': 2, 'base_channels': 16},
        {'name': 'D5', 'num_blocks': 2, 'base_channels': 32}, # Standard One
        {'name': 'D6', 'num_blocks': 2, 'base_channels': 64},
        {'name': 'D7', 'num_blocks': 3, 'base_channels': 32},
        {'name': 'D8', 'num_blocks': 3, 'base_channels': 64},
        {'name': 'D9', 'num_blocks': 4, 'base_channels': 32},
        {'name': 'D10', 'num_blocks': 4, 'base_channels': 64}
    ]

    return defender_strategies


def initialize_attackers() -> list:

    """
    Initializes the Attacker's strategies (Generators with different depth and capacity).

    Returns:
        A list of dictionaries, each containing the name and parameters for an Attacker strategy.
    """
    
    attacker_strategies = [
        {'name': 'A1', 'num_blocks': 1, 'base_channels': 8},
        {'name': 'A2', 'num_blocks': 1, 'base_channels': 16}, # Equivalent to BasicGenerator
        {'name': 'A3', 'num_blocks': 1, 'base_channels': 32},
        {'name': 'A4', 'num_blocks': 2, 'base_channels': 8},
        {'name': 'A5', 'num_blocks': 2, 'base_channels': 16}, # Equivalent to the Advanced original
        {'name': 'A6', 'num_blocks': 2, 'base_channels': 32},
        {'name': 'A7', 'num_blocks': 3, 'base_channels': 16},
        {'name': 'A8', 'num_blocks': 3, 'base_channels': 32},
        {'name': 'A9', 'num_blocks': 3, 'base_channels': 64}, 
        {'name': 'A10', 'num_blocks': 4, 'base_channels': 32} # Bottleneck very thin
    ]
        
    return attacker_strategies

def get_defender_ablations(num_blocks: int) -> list:

    """
    Generates the ablation matrix (Norm, DropBlock, Both) for Defenders.

    Args:
        num_blocks (int): Number of convolutional blocks. Determines base channels.

    Returns:
        list: Dictionaries containing hyperparameters ('name', 'num_blocks', 
              'base_channels', 'reg_type', 'dropout_rate', 'norm_type').
    """

    if num_blocks == 1:
        channel_configs = [8, 16]
    elif num_blocks == 2:
        channel_configs = [16, 32]
    elif num_blocks == 3:
        channel_configs = [32, 64]
    elif num_blocks >= 4:
        # Mantenim el 128 per poder demostrar empíricament l'overfitting!
        channel_configs = [32, 64, 128] 
    else:
        channel_configs = [32]

    strategies = []
    
    for base_channels in channel_configs:
        prefix = f"D_b{num_blocks}_c{base_channels}"
        
        strategies.extend([
            # 1. Baseline (without normalization or regularization)
            {'name': f'{prefix}_Vanilla', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'none'},
            
            # 2. Only Normalization (BatchNorm i InstanceNorm)
            {'name': f'{prefix}_BN', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'instance'},
            
            # 3. Only Regularization (DropBlock at 0.25 and 0.5)
            {'name': f'{prefix}_DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'none'},
            {'name': f'{prefix}_DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'none'},
            
            # 4. Both: Normalization + DropBlock at 0.25
            {'name': f'{prefix}_BN+DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN+DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'instance'},

            # 5. Both: Normalization + DropBlock at 0.5
            {'name': f'{prefix}_BN+DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN+DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'instance'}
        ])

    return strategies


def get_attacker_ablations(num_blocks: int) -> list:

    """
    Generates the ablation matrix (Norm, DropBlock, Both) for Attackers.

    Args:
        num_blocks (int): Number of convolutional blocks. Determines base channels.

    Returns:
        list: Dictionaries containing hyperparameters ('name', 'num_blocks', 
              'base_channels', 'reg_type', 'dropout_rate', 'norm_type').
    """

    if num_blocks == 1:
        channel_configs = [4, 8]
    elif num_blocks == 2:
        channel_configs = [8, 16]
    elif num_blocks == 3:
        channel_configs = [16, 32]
    elif num_blocks >= 4:
        channel_configs = [16, 32, 64] 
    else:
        channel_configs = [16]

    strategies = []
    
    for base_channels in channel_configs:
        prefix = f"A_b{num_blocks}_c{base_channels}"
        
        strategies.extend([
            # 1. Baseline (without normalization or regularization)
            {'name': f'{prefix}_Vanilla', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'none'},
            
            # 2. Only Normalization (BatchNorm and InstanceNorm)
            {'name': f'{prefix}_BN', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'none', 'dropout_rate': 0.0, 'norm_type': 'instance'},
            
            # 3. Only Regularization (DropBlock at 0.25 and 0.5)
            {'name': f'{prefix}_DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'none'},
            {'name': f'{prefix}_DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'none'},
            
            # 4. Both: Normalization + DropBlock at 0.25
            {'name': f'{prefix}_BN+DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN+DropBlock_025', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.25, 'norm_type': 'instance'},

            # 5. Both: Normalization + DropBlock at 0.5
            {'name': f'{prefix}_BN+DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'batch'},
            {'name': f'{prefix}_IN+DropBlock_050', 'num_blocks': num_blocks, 'base_channels': base_channels, 'reg_type': 'dropblock', 'dropout_rate': 0.5, 'norm_type': 'instance'}
        ])

    return strategies