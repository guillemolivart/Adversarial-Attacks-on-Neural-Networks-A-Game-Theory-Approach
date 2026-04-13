import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import time
import copy
import os

from nn import DynamicCNN
from generator import DynamicGenerator
from utils import set_seed, load_cifar10_data, evaluate_accuracy, initialize_defenders, initialize_attackers


def pretrain_classifier(
    classifier: nn.Module, 
    trainloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: Optimizer, 
    epochs: int, 
    device: torch.device
) -> None:
    
    """
    Pre-train the Defender (Classifier) on clean images.

    Args:
        classifier (nn.Module): The CNN model to be pre-trained.
        trainloader (DataLoader): The dataset loader for training images.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): The optimizer for the classifier.
        epochs (int): Number of epochs to pre-train.
        device (torch.device): The computation device ('cpu' or 'cuda').

    Returns:
        None (The model is updated in-place via the optimizer).
    """

    classifier.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def train_1v1_one_epoch(
    classifier: nn.Module, 
    generator: nn.Module, 
    trainloader: DataLoader, 
    criterion: nn.Module, 
    optimizer_C: Optimizer, 
    optimizer_G: Optimizer, 
    epsilon: float, 
    device: torch.device
) -> None:
    
    """
    Executes one epoch of the 1v1 minimax combat between the Classifier and the Generator.
    
    Args:
        classifier (nn.Module): The defender model (CNN) being trained.
        generator (nn.Module): The attacker model generating adversarial perturbations.
        trainloader (DataLoader): The dataset loader for training images.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer_C (Optimizer): The optimizer for the classifier.
        optimizer_G (Optimizer): The optimizer for the generator.
        epsilon (float): The maximum perturbation limit allowed for the attacker.
        device (torch.device): The computation device ('cpu' or 'cuda').
        
    Returns:
        None (The models are updated in-place via their optimizers).
    """

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        # TURN 1: ATTACKER (Generator) 
        classifier.eval() 
        generator.train()

        # Freeze Classifier gradients and unfreeze Generator gradients
        for param in classifier.parameters(): param.requires_grad = False
        for param in generator.parameters(): param.requires_grad = True

        for _ in range(2):
            optimizer_G.zero_grad()
            x_adv, _ = generator(images, epsilon=epsilon)
            outputs_adv = classifier(x_adv)
            
            loss_cnn = criterion(outputs_adv, labels)
            loss_G = -loss_cnn # Maximize the classifier's loss (minimize negative loss)
            
            loss_G.backward()
            optimizer_G.step()
            
        # TURN 2: DEFENDER (Classifier)
        classifier.train() 
        generator.eval()

        # Unfreeze Classifier gradients and freeze Generator gradients
        for param in classifier.parameters(): param.requires_grad = True
        for param in generator.parameters(): param.requires_grad = False

        optimizer_C.zero_grad()
        
        outputs_clean = classifier(images)
        loss_C_clean = criterion(outputs_clean, labels)

        x_adv, _ = generator(images, epsilon=epsilon) 
        x_adv_final = x_adv.detach() 
        outputs_def = classifier(x_adv_final)
        loss_C_adv = criterion(outputs_def, labels)
        
        loss_C = (loss_C_clean + loss_C_adv) / 2.0
        loss_C.backward()
        optimizer_C.step()


def main() -> None:

    """
    Main function to execute the 1v1 tournament between 10 Defenders (Classifiers) and 10 Attackers (Generators).

    Steps:
    1. Set up the environment and fix random seeds for reproducibility.
    2. Define the directory structure for saving models and matrices.
    3. Initialize the Defender and Attacker strategies with varying architectures.
    4. Pre-train each Defender on clean images and save their initial states.
    5. For each Attacker-Defender pair, conduct a 1v1 combat for a specified number of epochs, updating both models in a minimax fashion.
    6. After each epoch, evaluate the clean and adversarial accuracy of the Defender and store the results in 3D matrices.
    7. Save the final models and the 3D matrices for analysis.

    Args:
        None (All configurations are defined within the function).

    Returns:
        None (The function executes the tournament and saves results without returning any value).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Environment ready. Training with: {device}")
    
    set_seed(42)
    
    # DIRECTORY MANAGEMENT     
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    matrices_dir = os.path.join(project_root, 'output/matrices')
    models_dir = os.path.join(project_root, 'output/models')
    
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Hyperparameters
    batch_size = 1024 # 128 before
    epochs_pretrain = 10
    epochs_combat = 20 # 20 matrices
    epsilon = 0.2
    lr = 0.001
    
    trainloader, testloader = load_cifar10_data(batch_size=batch_size, data_dir=os.path.join(project_root, 'data'))
    criterion = nn.CrossEntropyLoss()
    
    def_strategies = initialize_defenders()
    att_strategies = initialize_attackers()
    
    num_def = len(def_strategies)
    num_att = len(att_strategies)
    


    # PHASE 1: PRE-TRAINING AND STORING BASE STATES
    


    print("\n🛡️ PHASE 1: Pre-training the 10 base Defenders...")
    pretrained_defender_states = []
    
    for i, strat in enumerate(def_strategies):
        print(f"   Pre-training {strat['name']} ({i+1}/10)...")

        # Classifier Model (Defender)
        classifier = DynamicCNN(num_blocks=strat['num_blocks'], base_channels=strat['base_channels']).to(device)

        # Optimizer for the Classifier
        opt_C = optim.Adam(classifier.parameters(), lr=lr)
        
        pretrain_classifier(classifier, trainloader, criterion, opt_C, epochs_pretrain, device)
        
        state_dict = copy.deepcopy(classifier.state_dict())
        pretrained_defender_states.append(state_dict)
        
        # Save the pre-trained model 
        torch.save(state_dict, os.path.join(models_dir, f"pretrained_{strat['name']}.pth"))
        


    # PHASE 2: 100 INDIVIDUAL COMBATS WITH MATRICES PER EPOCH



    print(f"\n⚔️ PHASE 2: Starting the {num_att}x{num_def} tournament ({epochs_combat} epochs per combat)...")
    
    # Create 3D Arrays: Dimensions -> (Epoch, Attacker, Defender)
    matrices_clean = np.zeros((epochs_combat, num_att, num_def))
    matrices_adv = np.zeros((epochs_combat, num_att, num_def))
    
    total_games = num_att * num_def
    current_game = 1
    start_time = time.time()
    
    for r, att_strat in enumerate(att_strategies):
        for c, def_strat in enumerate(def_strategies):
            
            print(f"🔄 Game {current_game}/{total_games}: {att_strat['name']} vs {def_strat['name']}...")
            
            # 1. INITIALIZE MODELS FOR THIS COMBAT

            # Generator Model (Attacker)
            generator = DynamicGenerator(num_blocks=att_strat['num_blocks'], base_channels=att_strat['base_channels']).to(device)
            opt_G = optim.Adam(generator.parameters(), lr=lr)
            
            # Classifier Model (Defender) - Start from the pre-trained state
            classifier = DynamicCNN(num_blocks=def_strat['num_blocks'], base_channels=def_strat['base_channels']).to(device)
            classifier.load_state_dict(copy.deepcopy(pretrained_defender_states[c]))
            opt_C = optim.Adam(classifier.parameters(), lr=lr)
            
            # 2. EPOCH LOOP FOR THIS COMBAT

            for epoch in range(epochs_combat):
                # Train only one epoch
                train_1v1_one_epoch(classifier, generator, trainloader, criterion, opt_C, opt_G, epsilon, device)
                
                # Evaluate and save in the matrix "layer" corresponding to this epoch
                acc_clean, acc_adv = evaluate_accuracy(classifier, generator, testloader, epsilon, device)
                
                matrices_clean[epoch, r, c] = acc_clean
                matrices_adv[epoch, r, c] = acc_adv
            
            # 3. SAVE THE FINAL MODELS OF THIS COMBAT
            torch.save(generator.state_dict(), os.path.join(models_dir, f"generator_{att_strat['name']}_vs_{def_strat['name']}.pth"))
            torch.save(classifier.state_dict(), os.path.join(models_dir, f"classifier_{def_strat['name']}_vs_{att_strat['name']}.pth"))
            
            current_game += 1

            del generator, classifier, opt_G, opt_C
            torch.cuda.empty_cache()



    # RESULTS AND SAVING 3D MATRICES



    print(f"\n🏁 Tournament finished in {time.time() - start_time:.2f} seconds.")
    
    # 1. Calculate the delta matrix (Clean - Adv) for each epoch
    matrices_delta = matrices_clean - matrices_adv
    
    # Print only the last epoch to the console as a summary
    print(f"\n📊 FINAL ADVERSARIAL ACCURACY MATRIX (Epoch {epochs_combat}):")
    print(np.round(matrices_adv[-1], 2))

    print(f"\n📉 FINAL DELTA MATRIX (Clean - Adv | Epoch {epochs_combat}):")
    print(np.round(matrices_delta[-1], 2))
    
    # Save the 3D Arrays in the matrices folder
    np.save(os.path.join(matrices_dir, 'payoff_matrices_clean.npy'), matrices_clean)
    np.save(os.path.join(matrices_dir, 'payoff_matrices_adv.npy'), matrices_adv)
    np.save(os.path.join(matrices_dir, 'payoff_matrices_delta.npy'), matrices_delta)
    
    print(f"\n💾 Everything successfully saved in:\n - {matrices_dir}/\n - {models_dir}/")

if __name__ == "__main__":
    main()