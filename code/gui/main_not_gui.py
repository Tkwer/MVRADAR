import os
import yaml
import threading
import argparse
import torch
from queue import Queue 
import utils.globalvar as gl
from bin.train import TrainRunner
from bin.train import TestRunner
import logging

# 定义全局日志开关
GLOBAL_LOG_ENABLE = True
 
# 全局日志等级
GLOBAL_LOG_LEVEL = logging.NOTSET
 
# 定义不同级别的日志颜色
COLORS = {
    'DEBUG': '\033[1;32m',  # 绿色
    'INFO': '\033[1;34m',  # 蓝色
    'WARNING': '\033[38;5;208m',  # 橙色
    'ERROR': '\033[1;31m',  # 红色
    'CRITICAL': '\033[1;35m',  # 紫色
}
 
def get_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
 
    if not GLOBAL_LOG_ENABLE:
        logger.disabled = True
        return logger
 
    if GLOBAL_LOG_LEVEL > logging.NOTSET:
        log_level = GLOBAL_LOG_LEVEL
 
    logger.setLevel(log_level)
    # 定义控制台输出的handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
 
    # 定义不同级别的日志输出颜色
    for level, color in COLORS.items():
        logging.addLevelName(getattr(logging, level), f'{color}{level}{color}')
 
    # 定义输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    # 定义控制台输出的格式
    console_handler.setFormatter(formatter)
 
    # 添加handler
    logger.addHandler(console_handler)
 
    return logger



def start_training(args):
    """
    开始训练过程。如果训练路径和方法正确配置，则启动训练任务；否则提示错误。
    """
    if args.train_and_vali_data_dir and args.fusion_mode:
        collector = TrainRunner('Listener', args.train_and_vali_data_dir, args.train_ratio, args)
        collector.start()
        update_train_progress(args)
        collector.join()   # 等待训练线程完成
    else:
        logger.info('请检查训练路径和选中方法!')

def start_testing(args):
    """
    开始测试过程。如果测试路径正确配置，则启动测试任务；否则提示错误。
    """

    if args.test_data_dir_and_model:
        collector = TestRunner('Listener',args.test_data_dir_and_model, args)
        collector.start()
        update_test_progress(args)
        collector.join()   # 等待训练线程完成
    else:
       logger.info('请检查路径和选中方法!')

def update_test_progress(args):
    """
    更新测试进度界面，包括打印信息和绘制图表。
    """
    update_progress(args, 
        update_callback=lambda: update_test_progress(args), 
        print_end_message='---- Testing complete!  ----',
        confusion_matrix1=args.confusion_train,
        confusion_matrix2=args.confusion_val
    )

def update_train_progress(args):
    """
    更新训练进度界面，包括打印信息和绘制图表。
    """
    update_progress(args,
        update_callback=lambda: update_train_progress(args), 
        print_end_message='---- Training Complete! ----',
        confusion_matrix1=args.confusion_train,
        confusion_matrix2=args.confusion_val
    )


    
def update_progress(args, update_callback, print_end_message, confusion_matrix1, confusion_matrix2, enable_ui_callback=None):
    print_str = args.print_queue.get()
    logger.info(print_str)

    if print_str != print_end_message:
        timer = threading.Timer(0.02, update_callback)
        timer.start()
    elif enable_ui_callback:
        enable_ui_callback()

    

# 读取 YAML 文件
def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    gl._init()
    logger = get_logger("main")
    metrics = {
        'loss_train': Queue(),
        'acc_train': Queue(),
        'loss_val': Queue(),
        'acc_val': Queue(),
        'confusion_train': Queue(),
        'confusion_val': Queue(),
        'print_queue': Queue(),
    }

    # 如果还想支持命令行覆盖 YAML 配置
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--config', default='code/examples/conf2.yaml', type=str, help='Path to config file')
    config = load_config(parser.parse_args().config)
    # 添加所有 YAML 参数支持命令行覆盖
    for key, value in config.items():
        arg_type = type(value) if value is not None else str  # 推断类型
        parser.add_argument(f'--{key.replace("_", "-")}', default=value, type=arg_type)

    # 动态添加 metrics 参数
    for key, value in metrics.items():
        # 推断参数类型。如果是 Queue 或复杂类型，设置为 str
        arg_type = str if isinstance(value, (Queue, dict, list)) else type(value)
        parser.add_argument(f'--{key.replace("_", "-")}', default=value, type=arg_type)

    # 解析参数
    args = parser.parse_args()
    args.save_path = []
    
    torch.manual_seed(777) #cuda也固定种子了
    for i in range(10):
        start_training(args)
        args.test_data_dir_and_model =[args.test_data_dir, [os.path.basename(args.save_path)]]
        start_testing(args)
