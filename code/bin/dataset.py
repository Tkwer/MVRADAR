import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, num_features, root_dirs, ratios, dataset_type, dict_class, transform=None):
        """
        自定义多模态/视图数据集
        :param num_features: 每个类别的特征数量
        :param root_dirs: 数据集根目录的列表
        :param ratios: 每个根目录的样本比例
        :param dataset_type: 数据集类型（'train' 或 'vali'）
        :param transform: 变换函数（可选）
        """
        self.num_features = num_features
        self.root_dirs = root_dirs
        self.ratios = ratios
        self.dataset_type = dataset_type
        self.dict_class = dict_class
        self.transform = transform
        self.classes = []  # 每个根目录的类别列表
        self.total_counts = []  # 每个类别的文件总数量
        self.sample_counts = []  # 每个类别的采样数量
        self.accumulated_counts = []  # 累积计数，用于索引
        
        # 初始化数据集统计信息
        for root_dir in self.root_dirs:
            class_list = sorted(os.listdir(root_dir))
            self.classes.append(class_list)
            
            counts = [len(os.listdir(os.path.join(root_dir, c))) for c in class_list]
            self.total_counts.append(counts)
            
            sample_count = [round(c / num_features * ratio) for c, ratio in zip(counts, [self.ratios[0]] * len(counts))]
            self.sample_counts.append(sample_count)
            
            acc_count = np.cumsum(sample_count)
            if self.accumulated_counts:
                acc_count += self.accumulated_counts[-1][-1]
            self.accumulated_counts.append(acc_count)

    def __len__(self):
        """返回数据集的总长度"""
        return int(sum(sum(np.array(self.sample_counts))))

    def __getitem__(self, idx):
        """根据索引获取数据样本"""
        for dataset_idx, acc_counts in enumerate(self.accumulated_counts):
            for class_idx, acc_count in enumerate(acc_counts):
                if idx < acc_count:
                    target_dataset = dataset_idx
                    target_class = class_idx
                    break
            else:
                continue
            break

        class_dir = os.path.join(self.root_dirs[target_dataset], self.classes[target_dataset][target_class])
        offset = 0
        if self.dataset_type == 'vali':
            offset = int(self.total_counts[target_dataset][target_class] / 5 - self.sample_counts[target_dataset][target_class])

        if target_class > 0:
            previous_count = self.accumulated_counts[target_dataset][target_class - 1]
        elif target_dataset > 0:
            previous_count = self.accumulated_counts[target_dataset - 1][-1]
        else:
            previous_count = 0

        sample_idx = idx - previous_count + offset
        file_path = os.path.join(class_dir, sorted(os.listdir(class_dir))[sample_idx])
        # # 调试信息打印
        # print(f"Label: {self.dict_class[target_class]}, Class: {self.classes[target_dataset][target_class]}, "
        #     f"Feature: {os.path.basename(file_path)[:12]}, Dataset Type: {self.dataset_type}, "
        #     f"Position: {os.path.basename(file_path)[-9:-4]}")
        # print(f"File Path: {file_path}")

        # 加载特征文件
        def load_and_expand(feature_name):
            feature_path = os.path.join(class_dir, feature_name + os.path.basename(file_path)[-18:])
            feature = np.load(feature_path).astype(np.float32)
            if feature_name=='DT' or feature_name=='RT':
                return torch.tensor(np.expand_dims(feature, axis=0))
            else:
                return torch.tensor(np.expand_dims(feature, axis=1))
            
        art_feature = load_and_expand('ART')
        dt_feature = load_and_expand('DT')
        ert_feature = load_and_expand('ERT')
        rdt_feature = load_and_expand('RDT')
        rt_feature = load_and_expand('RT')

        label = torch.tensor([target_class], dtype=torch.long)
        dataset_label = torch.tensor([target_dataset], dtype=torch.long)

        return art_feature, dt_feature, ert_feature, rdt_feature, rt_feature, label, file_path, dataset_label