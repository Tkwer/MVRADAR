import torch
from bin.train_utils import *
import numpy as np
from utils.helpers import AverageMeter, minmaxscaler
from utils.common import accuracy, confusion_matrix_compute

def validate_model(args, val_loader, num_classes, model, criterion):
    """
    Validate the MultiViewFeatureFusion model on validation or test data.

    Args:
        args: Arguments containing configurations and hyperparameters.
        val_loader: DataLoader for validation data.
        num_classes: Number of output classes.
        model: MultiViewFeatureFusion model to validate.
        criterion: Loss function.

    Returns:
        tuple: (top1 accuracy, top5 accuracy, average loss, normalized accuracy, confusion matrix, error list)
    """
    # Initialize metrics
    loss_tracker = AverageMeter()
    accuracy_top1 = AverageMeter()
    accuracy_top5 = AverageMeter()
    list_err = []
    device = args.device
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Unpack data
            # data[0] is a list of feature tensors
            # data[1] is the target labels
            # data[2] is file_paths
            features_list, targets, file_paths, domain_labels = [
                [d.to(args.device) for d in data[:-3]], data[-3],data[-2],data[-1]]
            
            targets = targets.to(device)

            # Create a features dictionary mapping feature names to tensors
            features_dict = {
                feature_name: feature.to(device)
                for feature_name, feature in zip(model.selected_features, features_list)
            }

            # Create a features dictionary mapping feature names to tensors
            features_dict = {
                feature_name: feature
                for feature_name, feature in zip(args.optional_features, features_list)
            }
            # 从 features_dict 中选择名在 args.selected_features 中的特征  
            selected_features_dict = {  
                feature_name: features_dict[feature_name]  
                for feature_name in model.selected_features  
                if feature_name in features_dict  
            }  
            # Normalize features
            selected_features_dict = {k: minmaxscaler(v) for k, v in selected_features_dict.items()}

            # Forward pass
            # Forward pass
            if args.method=='DScombine':
                alphas, alpha_combined, u_a, u_tensor = model(selected_features_dict) 
                weights = 1 - u_tensor
                outputs = alpha_combined - 1
            else:
                outputs, weights, fused_features = model(selected_features_dict)  # Outputs are logits of shape [batch_size, num_classes]
                # Compute loss
            loss = criterion(outputs, targets.squeeze(dim=1))
            loss_tracker.update(loss.item(), targets.size(0))

            # Compute accuracy and confusion matrix
            prec1, prec5 = accuracy(outputs.cpu(), targets.cpu(), topk=(1, 5))
            predictions = torch.argmax(outputs, dim=1)
            confusion_matrix = confusion_matrix_compute(predictions.cpu(), targets.cpu(), confusion_matrix)
            accuracy_top1.update(prec1.item(), targets.size(0))
            accuracy_top5.update(prec5.item(), targets.size(0))

            # Log validation progress
            args.print_queue.put(
                'Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    batch_idx, len(val_loader),
                    loss=loss_tracker, top1=accuracy_top1)
            )

            # Record misclassified samples
            predicted_labels = predictions.cpu().numpy()
            true_labels = targets.cpu().numpy()
            if args.is_test:
                misclassified_indices = np.where(true_labels.squeeze() != predicted_labels)[0]
                if len(misclassified_indices) > 0:
                    misclassified_paths = [file_paths[i] for i in misclassified_indices]
                    list_err.extend(misclassified_paths)
            else:
                list_err = None

    normalized_accuracy = accuracy_top1.avg / 100.0
    return (accuracy_top1.avg, accuracy_top5.avg, loss_tracker.avg,
            normalized_accuracy, confusion_matrix.cpu().numpy(), list_err)
