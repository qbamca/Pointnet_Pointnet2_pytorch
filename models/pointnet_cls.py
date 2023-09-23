import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

# Define the main PointNet classification model
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        
        # If normals are included in the input data, set the number of input channels to 6 (3 for XYZ coordinates and 3 for normals). 
        # Otherwise, just use 3 channels for XYZ.
        if normal_channel:
            channel = 6
        else:
            channel = 3
        
        # The encoder extracts features from the point cloud data. 
        # It will return global features, i.e., a single vector representation of the entire point cloud.
        # It may also return transformation matrices if feature transform is enabled.
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        
        # Define fully connected layers to process the global feature vector for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        # Dropout layer for regularization, helps prevent overfitting
        self.dropout = nn.Dropout(p=0.4)
        
        # Batch normalization layers to stabilize and speed up training
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Get the global features and any transformation matrices from the encoder
        x, trans, trans_feat = self.feat(x)
        
        # Pass the global features through the fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        # Apply softmax activation to get class probabilities
        x = F.log_softmax(x, dim=1)
        
        # Return the class probabilities and the feature transformation matrix (for regularization)
        return x, trans_feat

# Define the loss function for PointNet
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        
        # Scaling factor for the matrix difference loss (used for regularization)
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # Compute the standard classification loss (negative log likelihood)
        loss = F.nll_loss(pred, target)
        
        # Compute the matrix difference loss using the feature transform regularizer
        # This loss encourages the learned transformation matrix to be close to an orthogonal matrix
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        # Combine the classification loss and matrix difference loss
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
