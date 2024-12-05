import os
import torch
import torch.nn as nn
from bin.dataset import MultiViewDataset
from bin.train_utils import train_model
from bin.validate_utils import validate_model
from models.model import ContrastModel, FuzzyModel_V2, MFFNet
from utils.common import freeze_model_params
from utils.common import adjust_learning_rate, save_training_results
from utils.checkpoint import load_existing_model, save_checkpoint
import threading as th
import utils.globalvar as gl
import datetime
import numpy as np

def initialize_model(args, num_classes, architecture, train_ratios):
    """
    初始化模型
    :param num_classes: 分类数
    :param architecture: 模型架构
    :param train_ratios: 训练集划分比例
    :return: 初始化的模型
    """
    if 'ALL_' in architecture:
        return MFFNet(num_classes, architecture, args.lstm_layers, args.hidden_size, args.fc_size)
    elif 'F-DATA_' in architecture:
        return FuzzyModel_V2(64, architecture, args.lstm_layers, args.hidden_size, args.fc_size, len(train_ratios))
    else:
        return ContrastModel(num_classes, architecture, args.lstm_layers, args.hidden_size, args.fc_size)

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
    save_path = os.path.join(args.model, f"{args.arch}_model-{datetime.datetime.now().strftime('%m-%d-%H-%M')}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'output'), exist_ok=True)
    
    np.savetxt(os.path.join(save_path, 'output/log.txt'), 
               ([f"Train Dataset: {data_dir}"], train_ratios), fmt='%s')

    # 模型初始化
    model = initialize_model(args, len(train_dataset.classes[0]), args.arch, train_ratios)
    model.to(device)

    # 加载已有模型
    if gl.get_value('train_model') is not None:
        save_modelpath = 'save_model/'+gl.get_value('train_model')
        existing_model_path = os.path.join(save_modelpath, f"{args.arch}checkpoint.pth.tar")
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
            'arch': args.arch,
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
    args.is_test = True
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

    model_dir = args.model
    available_models = sorted(
        (os.path.join(model_dir, file) for file in os.listdir(model_dir)),
        key=os.path.getmtime,
    )
    model_dirs = sorted(filter(lambda path: testInfo[1][0] in path, available_models))
    model_type = testInfo[1][0].split('/')[-1].split('_')[0] + '_' + testInfo[1][0].split('/')[-1].split('_')[1] 
    best_model_path = os.path.join(model_dirs[-1], model_type+"_model_best.pth.tar")

    if not os.path.exists(best_model_path):
        args.print_queue.put("Model not found!")
        return
        # Load the best model
    model_info = torch.load(best_model_path)
    args.print_queue.put(f"Loaded model from: {model_dirs[-1]}")
    model = initialize_model(args, len(test_dataset.classes[0]), model_type, testInfo[0])

    model.load_state_dict(model_info["state_dict"])
    model.to(device)
    model.eval()    

    # Log model parameters and metrics
    args.print_queue.put(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.6f}M")
    args.print_queue.put(f"Best Accuracy: {model_info['best_prec']:.2f}%")

    train_loss, train_acc, val_loss, val_acc = np.loadtxt(os.path.join(model_dirs[-1], "output", "curve.txt"))
    # Push metrics to queues
    args.loss_train.put(train_loss)
    args.acc_train.put(train_acc)
    args.loss_val.put(val_loss)
    args.acc_val.put(val_acc)

    # Log output
    with open(os.path.join(model_dirs[-1], "output", "log.txt"), "r") as log_file:
        args.print_queue.put(log_file.read())

    # Perform validation/testing
    args.print_queue.put("------ Testing ------")
    top1_acc, _, _, _, confusion_matrix, errors = validate_model(args, 
        test_loader, len(test_dataset.classes[0]), model, criterion)

    # Save results
    save_path = os.path.join(model_dirs[-1], "output")
    np.savetxt(os.path.join(save_path, f"{os.path.basename(testInfo[0][0])}_errors.txt"), errors, fmt="%s")
    np.savetxt(os.path.join(save_path, f"{os.path.basename(testInfo[0][0])}_confusion_matrix.txt"), confusion_matrix.T, fmt="%d")
    args.confusion_val.put(confusion_matrix)

    # Log final results
    args.print_queue.put(f"Top-1 Accuracy: {top1_acc:.2f}%")
    args.print_queue.put("-------------------------------")
