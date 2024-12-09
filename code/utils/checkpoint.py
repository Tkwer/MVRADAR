import os
import torch
import shutil


def load_existing_model(model, model_path, print_queue):
    """
    加载已有模型
    :param model: 模型对象
    :param model_path: 模型路径
    :param print_queue: 打印队列
    :return: 当前最佳精度
    """
    if os.path.exists(model_path):
        model_info = torch.load(model_path)
        print_queue.put(f"==> Loading existing model")
        model.load_state_dict(model_info['state_dict'])
        print_queue.put(f"==> Best Accuracy: {model_info['best_prec']}")
        return model_info['best_prec']
    return 0


def save_checkpoint(state, path, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), path+'/'+'model_best.pth.tar')


def get_checkpoint_path(save_modelpath, target_string='checkpoint.pth.tar'):  
    checkpoint_path = None
    # 遍历目录  
    for root, _, files in os.walk(save_modelpath):  
        for file in files:  
            if target_string in file:  
                checkpoint_path = os.path.join(root, file)  
                
    return checkpoint_path 