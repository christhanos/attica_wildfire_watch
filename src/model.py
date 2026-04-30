import torch
import torch.nn as nn
import torchvision.models as models



class WildfireClassifier(nn.Module):
    def __init__(self):
        super(WildfireClassifier,self).__init__() #initialize parent class with super() in order for class to know it is a PyTorch module. 
        
        self.base_model = models.resnet18(weights = "DEFAULT")
        #We need to lock the pre-existing weights of the model
        for param in self.base_model.parameters(): #isolates each parameter object, allowing the code inside the loop to successfully freeze the gradients
            param.requires_grad = False

        #get the number of input features from the existing final layer (.fc stands for the final fully connected layer)
        input_features =self.base_model.fc.in_features 

        # Create a linear layer:  input: number of input features from the existing final layer -> output: 1 node (Fire (1) vs No Fire (0)):think of it as a bulb that flashes!
        self.base_model.fc= nn.Linear (in_features= input_features, out_features= 1)

    def forward(self,x):
        # Pass the incoming image 'x' through the entire ResNet18 pipeline
        x = self.base_model(x)
        return x 

