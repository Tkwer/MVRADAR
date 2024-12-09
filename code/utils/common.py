import os
import yaml
import numpy as np
from datetime import datetime

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


# Load YAML content into a dictionary  
def load_yaml_to_dict(file_path):  
    with open(file_path, 'r') as file:  
        return yaml.safe_load(file)  

# Update args with values from the loaded YAML  
def update_args_from_yaml(args, yaml_data):  
    for key, value in yaml_data.items():  
        if hasattr(args, key):  
            setattr(args, key, value)  
            
# Identify serializable types  
def is_serializable(value):  
    """Check if the value is serializable to YAML."""  
    serializable_types = (str, int, float, bool, list, tuple, dict, type(None))  # Basic serializable types  
    return isinstance(value, serializable_types)
  
# Save args to YAML  
def save_args_to_yaml(args, save_path):  
    args_dict = vars(args)  # Convert Namespace to dictionary  
    # Define types to exclude (e.g., queue.Queue)  
    # Filter out non-serializable values and those with attributes starting with double underscores  
    filtered_args = {  
        k: v for k, v in args_dict.items()  
        if is_serializable(v) and not k.startswith('__')  
    } 
    with open(save_path, 'w') as yaml_file:  
        yaml.dump(filtered_args, yaml_file, default_flow_style=False) 

def build_directory_structure(selected_features, optional_features, fusion_mode, method):
    # 判断特征是否为 "all"
    selected_features = sorted(selected_features) 
    if selected_features == sorted(optional_features):
        features_dir = "all"
    else:
        features_dir = "_".join(selected_features)

    # 处理fusion_mode
    # 假设fusion_mode只有一个父模式和对应子模式列表（根据你实际逻辑可调整）
    if fusion_mode:
        parent_dir = fusion_mode
        # 对子选项进行处理，这里假设只有互斥的一个被选中，或无子项
        if method and len(method) > 0:
            # 假设互斥只会有一个child选中
            child_dir = method 
            # 合成路径
            final_path = os.path.join(features_dir, parent_dir, child_dir)
        else:
            # 无子项的父模式
            final_path = os.path.join(features_dir, parent_dir)
    else:
        # 无fusion_mode则只有特征目录
        final_path = features_dir

    # 获取当前时间作为目录名的一部分
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_path = os.path.join(final_path, now_str)

    return final_path

