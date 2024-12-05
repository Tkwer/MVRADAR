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

        # Compute loss based on architecture
        if 'KA' in args.arch:
            target_labels = [args.class_mapping[x] for x in targets.squeeze(1).cpu().numpy()]
            target_labels = torch.tensor(target_labels).to(args.device)
            loss_main = criterion(outputs[0], targets.squeeze(1))
            aux_losses = [criterion(output, targets.squeeze(1)) for output in outputs[2:]]
            loss = loss_main + sum(aux_losses) + 0.1 * criterion(outputs[1], target_labels)
        elif 'Fuzzy' in args.arch:
            target_labels = getlabel(targets.squeeze(1).cpu().numpy(), args.batch_size)
            target_labels = [torch.LongTensor(label).to(args.device) for label in target_labels]
            losses = [criterion(output, target) for output, target in zip(outputs[:-1], target_labels)]
            domain_loss = criterion(outputs[-1], domain_label_var.squeeze(1))
            loss = getloss(*losses, domain_loss, alpha)
        else:
            loss = criterion(outputs, targets.squeeze(1))

        loss_tracker.update(loss.item(), len(targets))

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Compute accuracy and confusion matrix
        if 'Fuzzy' in args.arch:
            top1_accuracy = accuracy_Fuzzy(outputs.cpu(), targets.cpu())
            accuracy_top1.update(top1_accuracy, len(targets))
        else:
            prec1, prec5 = accuracy(outputs.cpu(), targets.cpu(), topk=(1, 5))
            predictions = torch.argmax(outputs, dim=1)
            confusion_matrix = confusion_matrix_compute(predictions.cpu(), targets.cpu(), confusion_matrix)
            accuracy_top1.update(prec1.item(), len(targets))
            accuracy_top5.update(prec5.item(), len(targets))

        # Log training progress
        args.print_queue.put(
            f"Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] "
            f"LR: {optimizer.param_groups[-1]['lr']:.5f} "
            f"Loss: {loss_tracker.val:.4f} ({loss_tracker.avg:.4f}) "
            f"Top1: {accuracy_top1.val:.3f} ({accuracy_top1.avg:.3f}) "
            f"Alpha: {alpha:.2f}"
        )

    return loss_tracker.avg, accuracy_top1.avg / 100, confusion_matrix.cpu().numpy()


def getlabel(target, batch_size):
    length = len(target)
    label1 = np.zeros([4,length])
    dic = [[0,3], [4,5], [2,6], [1]]
    for i in range(3):
        arr = target.copy()
        arr[(arr != dic[i][0]) & (arr != dic[i][1])] = 7
        arr[arr == 0] = 2 # BACK 
        arr[arr == 1] = 1 # DB
        arr[arr == 2] = 2 # DOWN
        arr[arr == 3] = 1 # FRONT
        arr[arr == 4] = 2 # LEFT
        arr[arr == 5] = 1 # RIGHT
        arr[arr == 6] = 1 # UP
        arr[arr == 7] = 0 # OTHER
        # if len(arr) != args.batch_size:
        #     print('-------'+str(len(arr)))
        label1[i] = arr
    arr = target.copy()
    arr[(arr != dic[3][0])] = 7
    arr[arr == 0] = 2 # BACK 
    arr[arr == 1] = 1 # DB
    arr[arr == 2] = 2 # DOWN
    arr[arr == 3] = 1 # FRONT
    arr[arr == 4] = 2 # LEFT
    arr[arr == 5] = 1 # RIGHT
    arr[arr == 6] = 1 # UP
    arr[arr == 7] = 0 # OTHER
    label1[3] = arr
    return label1


def accuracy_Fuzzy(output, target):
    correct = (output == target.squeeze()).sum().item()
    total = target.size(0)
    accuracy = correct / total * 100
    # accuracy = torch.tensor(accuracy)
    return accuracy
    
def getloss(loss1,loss2,loss3,loss4,loss5,alpha):
    # loss = loss1*1.2 + loss2 + loss3 + loss4
    loss = loss1*1.12 + loss2*1.07 + loss3*1.02 + loss4 + alpha * loss5*0.5
    return loss


def getoutput(output1, output2, output3, output4, batch_size):
    dic1 = [[7,3,0], [7,5,4], [7,6,2], [7,1]]
    length = len(output1)
    output = np.zeros(length)
    for i in range(length):
        max1 = torch.max(output1[i],0)
        max2 = torch.max(output2[i],0)
        max3 = torch.max(output3[i],0)
        max4 = torch.max(output4[i],0)
        max_list = [max1, max2, max3, max4]
        if all(t[1] == 0 for t in max_list): # 如果所有分类结果都是other，则考察置信度第二高的结果
            sorted_tensor, indices = torch.sort(output1[i], descending=True)
            max1 = (sorted_tensor[1], indices[1])
            sorted_tensor, indices = torch.sort(output2[i], descending=True)
            max2 = (sorted_tensor[1], indices[1])
            sorted_tensor, indices = torch.sort(output3[i], descending=True)
            max3 = (sorted_tensor[1], indices[1])
            sorted_tensor, indices = torch.sort(output4[i], descending=True)
            max4 = (sorted_tensor[1], indices[1])
            max_list = [max1, max2, max3, max4]
        # print(max_list)
        # maxresult = max((t for t in max_list if t[1] != 0), key=lambda x: x[0])
        maxresult = max((t for t in max_list if t[1] != 0), key=lambda x: x[0])
        sort = max_list.index(maxresult)
        output[i] = dic1[sort][maxresult[1]]

    

    output = torch.LongTensor(output)
    output = Variable(output).cuda()
    # print(output)
    return output
