import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import our custom dataset and model
from dataset import SatelliteWildfireDataset
from model import WildfireClassifier

# Duct-tape override just in case the test folder also has corrupted images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    print("Initializing Testing Sequence...")

    # 1. Define the exact same transformations we used in training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the TEST dataset (Make sure you have a 'data/test' or 'data/val' folder!)
    # Note: If your Kaggle folder named it 'val', change 'data/test' to 'data/val' below.
    test_data = SatelliteWildfireDataset(root_dir='data/test', transform=data_transforms)
    
    # We don't need to shuffle for testing, and we can use a larger batch size because we aren't calculating gradients
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 3. Instantiate the model and LOAD OUR SAVED BRAIN
    model = WildfireClassifier()
    model.load_state_dict(torch.load("wildfire_model.pth"))
    print("Saved 'wildfire_model.pth' brain successfully loaded!")

    # 4. Turn on Evaluation Mode (locks the weights)
    model.eval()

    # Trackers for our accuracy math
    correct_predictions = 0
    total_images = 0

    print(f"Beginning evaluation on {len(test_data)} unseen images. This will be fast...")

    # 5. The Testing Loop
    # torch.no_grad() turns off the memory-heavy calculus engine
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            
            # Forward pass: get the raw logits
            outputs = model(inputs)
            
            # Apply Sigmoid to squash the raw logits into a percentage (0.0 to 1.0)
            probabilities = torch.sigmoid(outputs)
            
            # If the probability is >= 0.5, we predict 1 (Fire). Otherwise 0 (No Fire).
            # .float() converts True/False into 1.0/0.0
            predictions = (probabilities >= 0.5).float()
            
            # Math: How many predictions matched the target labels?
            correct_predictions += (predictions == targets).sum().item()
            total_images += targets.size(0)

    # 6. Calculate Final Accuracy
    accuracy = (correct_predictions / total_images) * 100
    print("\n" + "="*40)
    print(f"🔥 FINAL  ACCURACY: {accuracy:.2f}% 🔥")
    print(f"Correctly identified {correct_predictions} out of {total_images} images.")
    print("="*40)