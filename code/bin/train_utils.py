import torch
import torch.nn.functional as F  

from utils.helpers import AverageMeter, minmaxscaler
from bin.ride_augmentation import ride_augmentation
from utils.common import accuracy, confusion_matrix_compute
from models.methods.DScombine import combined_loss
def train_model(args, train_loader, model, num_classes, criterion, optimizer, epoch):
    """
    Train the MultiViewFeatureFusion model for one epoch.

    Args:
        args: Arguments containing configurations and hyperparameters.
        train_loader: DataLoader for training data.
        model: MultiViewFeatureFusion model to be trained.
        num_classes: Number of output classes.
        criterion: Loss function.
        optimizer: Optimizer for training.
        epoch: Current epoch number.

    Returns:
        train_loss: Average training loss.
        train_accuracy: Average training accuracy.
        conf_matrix: Training confusion matrix as a numpy array.
    """
    # Initialize metrics
    loss_tracker = AverageMeter()
    accuracy_top1 = AverageMeter()

    model.train()
    confusion_matrix = torch.zeros(num_classes, num_classes).to(args.device)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        # Unpack data: data[0] is a list of feature tensors, data[1] is the target labels
        features_list, targets, domain_labels = [
            [d.to(args.device) for d in data[:-3]], data[-3],data[-1]]
        
        targets = targets.to(args.device)
        domain_labels = domain_labels.to(args.device)
        # Apply RIDE augmentation every 20 batches (optional)
        if args.is_augmentation != 0 and batch_idx % args.is_augmentation == 0:  
            features_list = ride_augmentation(*features_list)
            args.print_queue.put("--RIDE Applied--")

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
        if args.method=='DScombine':
            fused_features, alphas, alpha_combined, u_a, u_tensor = model(selected_features_dict) 
            weights = 1 - u_tensor
            outputs = alpha_combined - 1
            loss = combined_loss(targets.squeeze(dim=1), alphas, num_classes, alpha_combined, epoch, args.epochs, device=args.device)
        else:
            outputs, weights, fused_features = model(selected_features_dict)  # Outputs are logits of shape [batch_size, num_classes]
        
            # Compute loss
            loss = criterion(outputs, targets.squeeze(dim=1))

        # 如果加域不变对抗学习 前向传播，设置 alpha 用于 GRL
        if args.is_domain_loss:
            alpha = 0.1 + 0.9 * (epoch / args.epochs)  # 逐渐增大 alpha
            domain_logits = model.domain_discriminator(fused_features, alpha)
            adversarial_loss = criterion(domain_logits, domain_labels.squeeze(dim=1))
            loss = loss + args.is_domain_loss * adversarial_loss

        # 如果加标签损失
        if args.is_weights_loss:
            weights_label = torch.stack([torch.tensor([  
                args.mv_feature_weights[target.item()][feature_name]   
                for feature_name in model.selected_features])   
                    for target in targets]) 
            weights_log = F.log_softmax(weights, dim=-1)  # 确保 weights 是对数概率分布
            weights_label = F.softmax(weights_label, dim=-1)  # 确保 weights_label 是概率分布
            # 1. pred_log_prob需要是log概率  
            # 2. target_prob是概率分布  
            # 3. reduction参数选择:  
            #    - 'batchmean': 平均批次损失  
            #    - 'sum': 总损失  
            #    - 'none': 逐元素损失
            kl_loss = F.kl_div(weights_log, weights_label.to(args.device), reduction='sum') 
            loss = loss + args.is_weights_loss*kl_loss



        loss_tracker.update(loss.item(), targets.size(0))

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Compute accuracy and confusion matrix
        prec1 = accuracy(outputs.cpu(), targets.cpu(), topk=(1,))[0]
        predictions = torch.argmax(outputs, dim=1)
        confusion_matrix = confusion_matrix_compute(predictions.cpu(), targets.cpu(), confusion_matrix)
        accuracy_top1.update(prec1.item(), targets.size(0))

        # Log training progress
        args.print_queue.put(
            f"Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] "
            f"LR: {optimizer.param_groups[-1]['lr']:.5f} "
            f"Loss: {loss_tracker.val:.4f} ({loss_tracker.avg:.4f}) "
            f"Top1: {accuracy_top1.val:.3f} ({accuracy_top1.avg:.3f})"
        )

    return loss_tracker.avg, accuracy_top1.avg / 100, confusion_matrix.cpu().numpy()
