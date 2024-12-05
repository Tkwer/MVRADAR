import os
import numpy as np

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# 更新混淆矩阵
def confusion_matrix_compute(preds, labels, conf_matrix):
    
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def freeze_model_params(model):
    """
    冻结模型参数
    :param model: 模型对象
    """
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = False

def save_training_results(save_path, train_loss, train_acc, val_loss, val_acc, train_cm, val_cm):
    """
    保存训练结果
    :param save_path: 保存路径
    :param train_loss: 训练损失
    :param train_acc: 训练准确率
    :param val_loss: 验证损失
    :param val_acc: 验证准确率
    :param train_cm: 训练混淆矩阵
    :param val_cm: 验证混淆矩阵
    """
    output_dir = os.path.join(save_path, 'output')
    np.savetxt(os.path.join(output_dir, 'curve.txt'), 
               (train_loss, train_acc, val_loss, val_acc), fmt='%.6f')
    np.savetxt(os.path.join(output_dir, 'train_cmat.txt'), train_cm.T, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'val_cmat.txt'), val_cm.T, fmt='%d')

# lr_step设置为100 这句话执行不了的。动态降低学习率
def adjust_learning_rate(optimizer, lr_step, epoch):
    if not epoch % lr_step and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.6
    return optimizer