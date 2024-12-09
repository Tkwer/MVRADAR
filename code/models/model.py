import torch.nn as nn
from models.encoder import FeatureEncoder2D, FeatureEncoder3D

from models.decoder import (
    ConcatDecoder,
    AlignmentDecoder,
    SharedSpecificDecoder
)

class MultiViewFeatureFusion(nn.Module):  
    """  
    A multi-view feature fusion network combining RT, DT (2D), and RDT, ERT, ART (3D) features.  
    """  
    def __init__(self, backbone="custom", cnn_output_size=128, hidden_size=128,   
                 rnn_type='lstm', lstm_layers=1, bidirectional=True, fc_size=7,   
                 input_feature_shapes=None, fusion_mode='concatenate', method='attention', 
                 bottleneck_dim=None, selected_features=None):  
        super(MultiViewFeatureFusion, self).__init__()  

        self.fusion_mode = fusion_mode  # Fusion mode (concatenate/alignment/shared_specific)  
        self.method = method  # For SharedSpecificDecoder (e.g., 'basic_shared', etc.)  
        self.bottleneck_dim = bottleneck_dim  # Bottleneck dimension for shared-specific methods  
        self.selected_features = selected_features if selected_features else ['RT', 'DT', 'RDT', 'ERT', 'ART']  

        # Dynamically create encoders for the selected features  
        self.encoders = nn.ModuleDict()  
        for feature in self.selected_features:  
            if feature in ['RT', 'DT']:  # 2D Features  
                self.encoders[feature] = FeatureEncoder2D(  
                    fc_size=cnn_output_size, backbone=backbone, use_fc=True,  
                    input_feature_shape=input_feature_shapes[feature]  
                )  
            elif feature in ['RDT', 'ERT', 'ART']:  # 3D Features  
                self.encoders[feature] = FeatureEncoder3D(  
                    cnn_output_size=cnn_output_size, backbone=backbone, hidden_size=hidden_size,  
                    rnn_type=rnn_type, lstm_layers=lstm_layers, bidirectional=bidirectional,  
                    fc_size=cnn_output_size, feature_use_fc=True, input_feature_shape=input_feature_shapes[feature]  
                )  
            else:  
                raise ValueError(f"Unsupported feature type: {feature}. Choose from 'RT', 'DT', 'RDT', 'ERT', 'ART'.")  

        # Dimension of each feature representation (e.g., `cnn_output_size` after encoding)  
        feature_dim = cnn_output_size  

        # Initialize decoder based on fusion mode  
        if fusion_mode == 'concatenate':  
            # Use ConcatDecoder  
            self.decoder = ConcatDecoder(feature_dim, num_views=len(self.selected_features))  
        elif fusion_mode == 'alignment':  
            # Use AlignmentDecoder with a specific alignment strategy  
            self.decoder = AlignmentDecoder(feature_dim, num_views=len(self.selected_features), method=method)  
        elif fusion_mode == 'shared_specific':  
            # Use SharedSpecificDecoder with a specific shared/specific method  
            self.decoder = SharedSpecificDecoder(  
                feature_dim, num_views=len(self.selected_features), method=method, bottleneck_dim=bottleneck_dim  
            )  
        else:  
            raise ValueError(f"Invalid fusion mode: {fusion_mode}. Choose from 'concatenate', 'alignment', or 'shared_specific'.")  

        # Fully connected layer for post-fusion representation  
        self.fc_fusion = nn.Sequential(  
            nn.Linear(feature_dim, fc_size),  
            nn.ReLU(inplace=True),  
            nn.Dropout()  
        )  
        self.classifier = nn.Sequential()
        
    def forward(self, features):  
        """  
        Forward pass of the MultiViewFeatureFusion module.  

        Args:  
            features (dict): A dictionary where keys are feature names ('RT', 'DT', 'RDT', 'ERT', 'ART')   
                             and values are the corresponding raw features.  

        Returns:  
            torch.Tensor: The fused feature representation.  
        """  
        if not set(features.keys()).issubset(set(self.selected_features)):  
            raise ValueError(f"Input features ({features.keys()}) don't match selected features ({self.selected_features}).")  

        # Extract features using their corresponding encoders  
        encoded_features = []  
        for feature_name in self.selected_features:  
            encoded_feature = self.encoders[feature_name](features[feature_name])  
            encoded_features.append(encoded_feature)  

        # Fuse the extracted features using the decoder  
        fused_features = self.decoder(*encoded_features)  

        # Apply fully connected layer after fusion to obtain final representation  
        final_features = self.fc_fusion(fused_features)  
        ouputs = self.classifier(final_features)
        return ouputs