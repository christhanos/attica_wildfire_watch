import torch #convers images into PyTorch tensors
import torch.nn as nn
import torch.optim as optim #optimizer
from torch.utils.data import DataLoader #Dataloader grabs a batch of images (e.g. 32 at a time) and drives them into my model
from torchvision import transforms #Tool to convert and resize images

from dataset import SatelliteWildfireDataset
from model import WildfireClassifier

'''It tells Python to only run the code underneath it if you are executing this exact file directly in your terminal (python src/train.py). 
Without this shield,if you ever imported train.py into your future Streamlit web app, 
your computer would instantly start training the model all over again by accident!
'''
if __name__ == "__main__":
    print(f"Ready to build the training loop!")

    #defining the hyperparameters
    num_epochs = 2
    lr = 0.001
    batch_size = 32

    # --- Image Transformations ---
    # ResNet18 requires images to be exactly 224x224, converted to PyTorch Tensors, 
    # and normalized using standard ImageNet color metrics.
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # THE FIX: Pass the data_transforms into the dataset here!
    training_data = SatelliteWildfireDataset(root_dir='data/train', transform=data_transforms)
    
    # Also ensuring batch_size is explicitly named for clarity
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = WildfireClassifier()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")

        for batch_idx, (inputs,targets) in enumerate(train_dataloader):
            # 1. Zero out gradients
            optimizer.zero_grad()

            # 2. Forward pass: compute model predictions
            outputs = model(inputs)

            # 3. Calculate the loss (Use the criterion defined outside the loop!)
            loss = criterion(outputs, targets)

            # 4. Backward pass: compute gradients
            loss.backward()

            # 5. Update weights
            optimizer.step()

            # Print progress every 50 batches so we don't stare at a blank screen
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} complete! Final Loss: {loss.item():.4f}")

    # ---  Save the Model ---
    torch.save(model.state_dict(), "wildfire_model.pth")
    print("\nTraining Complete! Model saved as 'wildfire_model.pth'.")
