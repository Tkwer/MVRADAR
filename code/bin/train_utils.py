import torch
from torch.autograd import Variable
import numpy as np
from utils.helpers import AverageMeter, minmaxscaler
from bin.ride_augmentation import ride_augmentation
from utils.common import accuracy, confusion_matrix_compute

def train_model(args, train_loader, model, num_classes, criterion, optimizer, epoch):
    """
    Train the model for one epoch.
    Args:
        train_loader: DataLoader for training data.
        model: Model to be trained.
        num_classes: Number of output classes.
        criterion: Loss function.
        optimizer: Optimizer for training.
        epoch: Current epoch number.
        device: Device to run the training (CPU or GPU).
        log_queue: Queue to log training details.
    Returns:
        train_loss: Average training loss.
        train_accuracy: Average training accuracy.
        conf_matrix: Training confusion matrix as a numpy array.
    """
    # Initialize metrics
    loss_tracker = AverageMeter()
    accuracy_top1 = AverageMeter()
    accuracy_top5 = AverageMeter()

    model.train()
    confusion_matrix = torch.zeros(num_classes, num_classes).to(args.device)

    for batch_idx, data in enumerate(train_loader):
        # Unpack and move data to device
        features, targets, domain_labels = [
            [d.to(args.device) for d in data[:-3]],
            data[-3].to(args.device),
            data[-1].to(args.device)
        ]

        # Apply RIDE augmentation every 20 batches
        if batch_idx % 20 == 0:
            features = ride_augmentation(*features)
            args.print_queue.put("--RIDE Applied--")

        # Normalize features
        features = [minmaxscaler(f) for f in features]

        # Convert inputs and targets to Variables
        input_vars = [Variable(f, requires_grad=True) for f in features]
        target_var = Variable(targets)
        domain_label_var = Variable(domain_labels)

        # Calculate alpha for domain adaptation
        progress = float(epoch) / (args.epochs)
        alpha = (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0) * 0.8

        # Forward pass
        outputs = model(*input_vars)

        optimizer.zero_grad()
