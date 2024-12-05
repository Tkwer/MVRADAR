import torch
from torch.autograd import Variable
from bin.train_utils import *
import numpy as np
from utils.helpers import AverageMeter, minmaxscaler
from utils.common import accuracy, confusion_matrix_compute


def validate_model(args, val_loader, num_classes, model, criterion):
    """
    Validate the model on validation or test data.
    Args:
        val_loader: DataLoader for validation data.
        num_classes: Number of output classes.
        model: Model to validate.
        criterion: Loss function.
        args.print_queue: Queue to log validation details.
    Returns:
        tuple: (top1 accuracy, top5 accuracy, average loss, normalized accuracy, confusion matrix, error list)
    """
    loss_tracker = AverageMeter()
    accuracy_top1 = AverageMeter()
    accuracy_top5 = AverageMeter()
    list_err = []
    device = args.device
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes).to(args.device)

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Unpack and normalize features
            features, targets, file_paths, domain_labels = [
                [d.to(device) for d in data[:-3]],
                data[-3].to(device),
                data[-2],
                data[-1].to(device)
            ]

            features = [minmaxscaler(f) for f in features]
            input_vars = [Variable(f, requires_grad=True) for f in features]
            target_var = Variable(targets)
            domain_label_var = Variable(domain_labels)

            # Compute outputs
            outputs = model(*input_vars)

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
                loss = getloss(*losses, domain_loss, 1)
            else:
                loss = criterion(outputs, targets.squeeze(1))

            loss_tracker.update(loss.item(), len(targets))

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
            args.print_queue.put('Test: [{0}/{1}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            batch_idx, len(val_loader),
            loss=loss_tracker,top1=accuracy_top1)
            )

            if 'Fuzzy' in args.arch:
                a = outputs.data.cpu().numpy()
            else:
                a = outputs.data.cpu().numpy().argmax(axis=1)
                
            d = targets.data.cpu().numpy().squeeze(1) 
                    # # 打印出识别错误的数据路径
            if args.is_test:
                if not (np.array((d == a)).all()):
                    list_err.extend(np.array(file_paths)[np.argwhere((d == a) == False).squeeze(1)].tolist())
            else:
                list_err = None
                
    return accuracy_top1.avg, accuracy_top5.avg, loss_tracker.avg, accuracy_top1.avg / 100, confusion_matrix.cpu().numpy(), list_err