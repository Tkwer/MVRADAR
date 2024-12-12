import os
import torch
import torch.nn as nn
from bin.dataset import MultiViewDataset
from bin.train_utils import train_model
from bin.validate_utils import validate_model
from models.model import MultiViewFeatureFusion
from utils.common import *
from utils.common import freeze_model_params
from utils.common import adjust_learning_rate, save_training_results
from utils.checkpoint import *
import threading as th
import utils.globalvar as gl
import numpy as np
import re 


def initialize_model(args, num_classes):
    """
    Initialize the MultiViewFeatureFusion model based on the given architecture.

    Args:
        args: Arguments containing configurations and hyperparameters.
        num_classes: Number of classes.

    Returns:
        Initialized MultiViewFeatureFusion model.
    """

    # Get other fusion-related parameters from args or set defaults
    if args.fusion_mode == 'concatenate': 
        method = getattr(args, 'method', 'concat')
    elif args.fusion_mode == 'attention':  
        method = getattr(args, 'method', 'attention')

        
    bottleneck_dim = getattr(args, 'bottleneck_dim', None)

    # Create input_feature_shapes dictionary (should be provided in args)
    input_feature_shapes = args.input_feature_shapes  # Must be a dict mapping feature names to shapes

    # Initialize the MultiViewFeatureFusion model
    model = MultiViewFeatureFusion(
        backbone=args.backbone,
        cnn_output_size=args.cnn_output_size,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        fc_size=args.fc_size,
        num_domains=args.num_domains,
        input_feature_shapes=input_feature_shapes,
        fusion_mode=args.fusion_mode,
        method=method,
        is_sharedspecific=args.is_sharedspecific,
        bottleneck_dim=bottleneck_dim,
        selected_features=args.selected_features,
        is_domain_loss=args.is_domain_loss
    )

    # Add a classifier layer to output logits for num_classes
    model.classifier = nn.Linear(args.fc_size, num_classes)

    return model

class TrainRunner(th.Thread):
    """
    线程类，用于训练模型并收集结果
    """
    def __init__(self, name, data_dir, train_ratios, args):
        """
        :param name: 线程名称
        :param data_dir: 训练和验证数据目录
        :param train_ratios: 训练集划分比例
        :param print_queue: 用于存储打印信息的队列
        :param metrics: 用于存储训练和验证指标的字典
        """
        super().__init__(name=name)
        self.data_dir = data_dir
        self.train_ratios = train_ratios
        self.args = args

    def run(self):
        """
        线程运行的入口
        """
        run_training(self.args, self.data_dir, self.train_ratios)
        self.args.print_queue.put('---- Training Complete! ----')


def run_training(args, data_dir, train_ratios):
    """
    运行模型训练并收集指标
    :param data_dir: 数据目录
    :param train_ratios: 训练集划分比例
    :param print_queue: 打印信息队列
    :param metrics: 存储训练和验证指标的字典
    """
    train_loss_q = []
    train_acc_q = []
    val_loss_q = []
    val_acc_q = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.print_queue.put(f"Device being used: {device}")
    # 将device添加到metrics字典
    args.device = device
    args.is_test = False
    args.num_domains = len(train_ratios) if isinstance(train_ratios, list) else 1
    # 数据加载
    train_dataset = MultiViewDataset(5, data_dir, train_ratios, 'train', args.class_mapping)
    val_dataset = MultiViewDataset(5, data_dir, [1 - x for x in train_ratios], 'vali', args.class_mapping)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    # 模型保存路径
    final_path = build_directory_structure(args.selected_features, args.optional_features, args.fusion_mode, args.method)
    save_path = os.path.join(args.model_path, final_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'output'), exist_ok=True)
    args.save_path = save_path
    with open(os.path.join(save_path, 'output/log.txt'), 'w') as f:  
        f.write(f"train dataset: {data_dir}\n")  
        f.write(f"train ratios: {train_ratios}")  
    # Define the path for saving the YAML file  
    save_path_yaml = os.path.join(save_path, 'output', 'args.yaml')  
    
    # Make sure the output directory exists  
    os.makedirs(os.path.dirname(save_path_yaml), exist_ok=True)  
    
    # Save the arguments to YAML  
    save_args_to_yaml(args, save_path_yaml) 
    # 模型初始化
    model = initialize_model(args, len(train_dataset.classes[0]))
    model.to(device)

    # 加载已有模型
    if gl.get_value('train_model') is not None:
        existing_model_path = gl.get_value('train_model')
        best_prec = load_existing_model(model, existing_model_path, args.print_queue)
    else:
        best_prec = 0

    # 冻结参数
    if args.freeze:
        freeze_model_params(model)
        args.print_queue.put('------ Parameters Frozen ------')

    args.print_queue.put(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.6f}M')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, args.lr_step, epoch)
        args.print_queue.put(f"------ Training: Epoch {epoch + 1} ------")

        # 单个 epoch 的训练
        train_loss, train_acc, train_cm = train_model(args,
            train_loader, model, len(train_dataset.classes[0]), criterion, optimizer, epoch
        )
        train_loss_q.append(train_loss)
        train_acc_q.append(train_acc)
        args.loss_train.put(train_loss_q)
        args.acc_train.put(train_acc_q)
        args.confusion_train.put(train_cm)

        # 验证集评估
        args.print_queue.put("------ Validation ------")
        val_prec, _, val_loss, val_acc, val_cm, _ = validate_model(args,
            val_loader, len(train_dataset.classes[0]), model, criterion
        )
        args.print_queue.put("------Validation accuracy: {prec: .2f} %  ------".format(prec=val_prec))
        val_loss_q.append(val_loss)
        val_acc_q.append(val_acc)
        args.loss_val.put(val_loss_q)
        args.acc_val.put(val_acc_q)
        args.confusion_val.put(val_cm)

        # 保存训练结果
        save_training_results(
            save_path, train_loss_q, train_acc_q, val_loss_q, val_acc_q, train_cm, val_cm
        )

        # 更新最佳模型
        is_best = val_prec > best_prec
        best_prec = max(val_prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'selected_features': args.selected_features,
            'fusion_mode': args.fusion_mode,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, save_path, is_best)   



class TestRunner(th.Thread):
    """
    线程类，用于处理模型测试的线程。
    """
    def __init__(self, name, test_data_dirs, args):
        """
        :param name: 线程名称
        :param data_dir: 训练和验证数据目录
        :param train_ratios: 训练集划分比例
        :param print_queue: 用于存储打印信息的队列
        :param metrics: 用于存储训练和验证指标的字典
        """
        super().__init__(name=name)
        self.data_dir = test_data_dirs
        self.args = args

    def run(self):
        """
        线程运行的入口
        """
        run_testing(self.args, self.data_dir)
        self.args.print_queue.put('---- Testing complete!  ----')


def run_testing(args, testInfo):
    """
    使用指定的模型对给定的数据目录执行测试。
    :param testInfo: 数据目录和模型目录
    :param print_queue: 打印信息队列
    :param metrics: 存储训练和验证指标的字典
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.print_queue.put(f"Device being used: {device}")
    args.device = device
    
    # Load test dataset
    test_dataset = MultiViewDataset(
        num_features=5,
        root_dirs=testInfo[0],
        ratios=[1] * len(testInfo[0]),
        dataset_type="train",
        dict_class=args.class_mapping,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model_architecture  = testInfo[1][0]
    args.print_queue.put(f"Testing using architecture: {model_architecture}")
    criterion = nn.CrossEntropyLoss().to(device)

    model_dir = args.model_path
    available_models = []
    # Define a regex pattern for matching YYYY-MM-DD_HH-MM-SS format  
    date_pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'  

    # Walk through the directory tree  
    for root, dirs, files in os.walk(model_dir):  
        for dir_name in dirs:  
            # Check if the directory name matches the date pattern  
            if re.match(date_pattern, dir_name):  
                # Construct the full path to the directory  
                full_path = os.path.join(root, dir_name)  
                available_models.append(full_path)  
    model_dirs = sorted(filter(lambda path: testInfo[1][0] in path, available_models))[0]
    # Define the path for the YAML file  
    yaml_path = os.path.join(model_dirs, 'output', 'args.yaml')  

    # Check if the YAML file exists  
    if os.path.isfile(yaml_path):  
        # Load the YAML file  
        yaml_data = load_yaml_to_dict(yaml_path)  
        
        # Update the args with values from the YAML  
        update_args_from_yaml(args, yaml_data)  
    args.is_test = True
    best_model_path = os.path.join(model_dirs, "model_best.pth.tar")


    if not os.path.exists(best_model_path):
        args.print_queue.put("Model not found!")
        return
        # Load the best model
    model_info = torch.load(best_model_path)
    args.print_queue.put(f"Loaded model from: {model_dirs}")
    
    model = initialize_model(args, len(test_dataset.classes[0]))

    model.load_state_dict(model_info["state_dict"])
    model.to(device)
    model.eval()    

    # Log model parameters and metrics
    args.print_queue.put(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.6f}M")
    args.print_queue.put(f"Best Accuracy: {model_info['best_prec']:.2f}%")

    train_loss, train_acc, val_loss, val_acc = np.loadtxt(os.path.join(model_dirs, "output", "curve.txt"))
    # Push metrics to queues
    args.loss_train.put(train_loss)
    args.acc_train.put(train_acc)
    args.loss_val.put(val_loss)
    args.acc_val.put(val_acc)

    # Log output
    with open(os.path.join(model_dirs, "output", "log.txt"), "r") as log_file:
        args.print_queue.put(log_file.read())

    # Perform validation/testing
    args.print_queue.put("------ Testing ------")
    top1_acc, _, _, _, confusion_matrix, errors = validate_model(args, 
        test_loader, len(test_dataset.classes[0]), model, criterion)

    # Save results
    save_path = os.path.join(model_dirs, "output")
    np.savetxt(os.path.join(save_path, "errors.txt"), errors, fmt="%s")
    np.savetxt(os.path.join(save_path, "confusion_matrix.txt"), confusion_matrix.T, fmt="%d")
    args.confusion_val.put(confusion_matrix)

    # Log final results
    args.print_queue.put(f"Top-1 Accuracy: {top1_acc:.2f}%")
    args.print_queue.put("-------------------------------")
