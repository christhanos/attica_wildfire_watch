#dataset.py acts like a translator (hard drive-> RAM)  because PyTorch cannot read folders but only mathematical tensors
#=======================================================================================================================
import os #navigate through my folder paths
import torch #convert images into PyTorch tensors
from torchvision import transforms #we will use this to resize the images
from PIL import Image, ImageFile #to be able to open the images
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset #the blueprint we are going to inherit from

class SatelliteWildfireDataset(Dataset):
    def __init__(self, root_dir, transform = None): #set up the class attributes inside the init
        #assign self to rood_dir and transform, so the rest of the class can access them
        self.root_dir = root_dir
        self.transform = transform
        #create a list with my 2 foler names. Careful here 'nowildfire' has an index of 0 because it is first in the list
        self.classes = ['nowildfire', 'wildfire']

        self.image_paths = []
        self.labels = []

        for idx, name  in enumerate(self.classes): # i use the enumerate function in order to get the index and the name of the class
            class_dir = os.path.join(root_dir, name )
            for filename  in os.listdir(class_dir):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    full_path = os.path.join(class_dir,filename)
                    self.image_paths.append(full_path)
                    self.labels.append(float(idx))

    def __len__(self): #PyTorch just needs to know the total size of my datatset
        return len(self.image_paths)

    def __getitem__(self, idx): #pass an index and expect an index and an image back
        img_path = self.image_paths[idx]
        label = self.labels[idx] 
        image = Image.open(img_path).convert('RGB') #open a file from my hard drive and ensures it has RGB channels
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([label], dtype = torch.float32)
        return image, label

if __name__ == "__main__":
    #Instantiate my custom class, pointing it to my train folder
    test_dataset = SatelliteWildfireDataset(root_dir='data/train')
    
    #Test my __len__ method
    print(f"Dataset successfully loaded! Found {len(test_dataset)} total images.")
    
    #Test my __getitem__ method (asking for the very first image at index 0)
    img, label = test_dataset[0]
    print(f"First image properties -> Size: {img.size}, Tensor Label: {label}")