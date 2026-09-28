import torch
import torch.nn as nn

class Scratch3DCNN(nn.Module):
    """
    Custom 3D CNN model for Spatio-Temporal Video Anomaly Detection.
    
    The architecture is inspired by C3D but designed from scratch to work well with 
    the specific requirements of video anomaly detection (e.g., handling variable 
    spatial dimensions using global adaptive pooling).
    
    Inputs:
        x: Tensor of shape (B, C, T, H, W) where C=3, T=16. 
           Spatial dimensions (H, W) can be variable (e.g., 112x112 for training, 112x144 for inference).
           
    Outputs:
        forward(x): Raw logits of shape (B, 1). NO sigmoid is applied here.
        forward_with_embeddings(x): Tuple of (logits, embeddings) where embeddings 
                                   are the 512-dim features before the MLP head.
    """
    def __init__(self):
        super(Scratch3DCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            # Block 3a
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            # Block 3b
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            # Block 4a
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            # Block 4b
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Global Adaptive Average Pool 3D -> (B, 512, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # MLP Head
        self.dropout1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(512, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights using Kaiming He initialization for Conv3d and Linear layers.
        Constant initialization for BatchNorm3d layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract 512-dim features.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        # Flatten: (B, 512, 1, 1, 1) -> (B, 512)
        x = x.view(x.size(0), -1)
        return x

    def forward_with_embeddings(self, x: torch.Tensor):
        """
        Returns both logits and the 512-dim embedding before the MLP head.
        """
        embeddings = self.extract_features(x)
        
        out = self.dropout1(embeddings)
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.dropout2(out)
        logits = self.fc2(out)
        
        return logits, embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning only logits.
        """
        logits, _ = self.forward_with_embeddings(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Prints and returns the total and trainable parameters of a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    return trainable_params
