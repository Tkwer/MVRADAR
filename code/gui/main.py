import sys
import yaml
import argparse
import torch
import numpy as np
import pyqtgraph as pg
from queue import Queue 
from interface import Ui_MainWindow, printlog, get_file_list
from colortrans import pg_get_cmap
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import utils.globalvar as gl
import matplotlib.pyplot as plt
from bin.train import TrainRunner
from bin.train import TestRunner
 
def get_color_list(color_combo_box):  
    """  
    获取颜色映射表列表并填充下拉框  
    
    Args:  
        color_combo_box (QComboBox): 颜色选择下拉框  
    """  
    color_values =  plt.colormaps()
    color_combo_box.addItem("--select--")  
    color_combo_box.addItem("customize")  
    for value in color_values:  
        color_combo_box.addItem(value)  

def set_color_map(color_combo_box, image_list):  
    """  
    设置颜色映射表  
    
    Args:  
        color_combo_box (QComboBox): 颜色选择下拉框  
        image_list (list): 需要应用颜色映射的图像列表  
    """  
    current_color = color_combo_box.currentText()  
    
    if current_color != '--select--' and current_color != '':  
        if current_color == 'customize':  
            pg_colormap = pg_get_cmap(current_color)  
        else:  
            mpl_colormap = plt.cm.get_cmap(current_color)  
            pg_colormap = pg_get_cmap(mpl_colormap)  
        
        lookup_table = pg_colormap.getLookupTable(0.0, 1.0, 256)  
        
        for image in image_list:  
            image.setLookupTable(lookup_table)  
        
        update_display_features() 


def update_display_features():
    """  
    加载并显示多个特征切片  
    
    Args:  
        sample_index (int): 当前样本索引  
        frame_index (int): 当前frame索引  
    """  

    sample_index  = Spinbox.value()
    frame_index = Slider.value()

    if not file_list[0]:  # 检查文件列表是否为空  
        return  

    # 加载所有图像数据 
    dti_data = np.load(file_list[0][sample_index])
    rti_data = np.load(file_list[1][sample_index])
    rdi_data = np.load(file_list[2][sample_index])
    rai_data = np.load(file_list[3][sample_index])
    rei_data = np.load(file_list[4][sample_index])

    # 删除特定索引范围的数据  
    rei_data = np.delete(rei_data,[i for i in range(40,64)],axis=2)
    rai_data = np.delete(rai_data,[i for i in range(40,64)],axis=2)
    # 设置图像及其显示参数   
    img_params = [  
        (image_items[2], rti_data, [0, 1e4]),  
        (image_items[0], rdi_data[frame_index, :, :].T, [2e4, 4e5]),  
        (image_items[4], rei_data[frame_index, :, :].T, [0, 6]),  
        (image_items[3], dti_data, [0, 1000]),  
        (image_items[1], rai_data[frame_index, :, :], [0, 1])  
    ]  

    for image_widget, image_data, level_range in img_params:  
        image_widget.setImage(image_data, levels=level_range)  

def auto_update_figure(current_index=0, max_frame=12):  
    """  
    自动更新图像显示  
    
    Args:  
        current_index (int): 当前索引  
    """  
    if autobtn.isChecked():  
        Slider.setValue(current_index)  
        next_index = (current_index + 1) % max_frame  
        
        update_display_features()  
        
        QtCore.QTimer.singleShot(200, lambda: auto_update_figure(next_index))  

def enable_training_ui():
    """
    启用与训练相关的 UI 控件，允许重新操作。
    """
    comboBox_6.setEnabled(True)
    comboBox_7.setEnabled(True)
    startTestbtn.setEnabled(True)
    groupBox_4.setEnabled(True)
    pushButton_5.setEnabled(True)
    comboBox_5.setEnabled(True)
    comboBox_4.setEnabled(True)
    startTrainbtn.setEnabled(True)

def start_training(args):
    """
    开始训练过程。如果训练路径和方法正确配置，则启动训练任务；否则提示错误。
    """
    train_and_vali_data_dir = gl.get_value('train_and_vali_data_dir')
    train_ratio = gl.get_value('train_ratio')
    selected_features = gl.get_value('selected_features')
    fusion_mode = gl.get_value('fusion_mode')
    if selected_features is not None and fusion_mode is not None:
        args.selected_features = selected_features
        for parent, children in fusion_mode.items():
            args.fusion_mode = parent  # 父类名
            args.method = children[0] if children else None  # 只取第一个子类，如果无子类为 None

    if train_and_vali_data_dir and args.fusion_mode:
        collector = TrainRunner('Listener', train_and_vali_data_dir, train_ratio, args)
        collector.start()
        update_train_progress(args)
    else:
        printlog(textEdit, '请检查训练路径和选中方法!')
        enable_training_ui()

def start_testing(args):
    """
    开始测试过程。如果测试路径正确配置，则启动测试任务；否则提示错误。
    """
    test_data_dir_and_model = gl.get_value('test_data_dir_and_model')

    if test_data_dir_and_model:
        collector = TestRunner('Listener', test_data_dir_and_model, args)
        collector.start()
        update_test_progress(args)
    else:
        printlog(textEdit, '请检查路径和选中方法!')

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
        confusion_matrix2=args.confusion_val,
        enable_ui_callback=enable_training_ui
    )

def update_canvas_data(canvas, line_handle, data_queue):
    """
    更新画布上的数据曲线。
    """
    data = data_queue.get()
    line_handle.set_data(range(len(data)), data)
    canvas.draw()
    
def update_progress(args, update_callback, print_end_message, confusion_matrix1, confusion_matrix2, enable_ui_callback=None):
    """
    通用的进度更新逻辑，适用于训练和测试。
    """
    print_str = args.print_queue.get()
    printlog(textEdit, print_str)

    if not args.loss_train.empty():
        update_canvas_data(canvas, canvas.hl1, args.loss_train)

    if not args.loss_val.empty():
        update_canvas_data(canvas, canvas.hl2, args.loss_val)

    if not args.acc_train.empty():
        update_canvas_data(canvas1, canvas1.hl1, args.acc_train)

    if not args.acc_val.empty():
        update_canvas_data(canvas1, canvas1.hl2, args.acc_val)

    if not confusion_matrix1.empty():
        conf_matrix1.update_show(np.array(confusion_matrix1.get()))

    if not confusion_matrix2.empty():
        conf_matrix2.update_show(np.array(confusion_matrix2.get()))

    if print_str != print_end_message:
        QtCore.QTimer.singleShot(20, update_callback)
    elif enable_ui_callback:
        enable_ui_callback()

def select_file():
    """
    选择文件操作，更新文件路径和显示的特征数据。
    """
    global file_list, sel_combo_box
    data_path = gl.get_value('data_path')

    if sel_combo_box.currentText() not in ['--select--', ''] and data_path:
        gesture_file_path = data_path+'/'+ sel_combo_box.currentText()
        file_list = [[] for _ in range(5)]
        file_list = get_file_list(gesture_file_path, file_list)
        Spinbox.setMaximum(len(file_list[0]) - 1)
        update_display_features()
    
def openfile():
    pass

def Recognition_Gesture(view_gesture):
    # fanhui = ctt.recognize1(net,[i[Spinbox.value()] for i in file_list])
    fanhui = 7
    view_gesture.setPixmap(QtGui.QPixmap("visualization/gesture_icons/"+str(fanhui)+".jpg"))

    
def setup_views(ui):
    """
    设置视图和画布的显示。
    """
    graphicsView = {
        "rdi": ui.graphicsView_6,
        "rai": ui.graphicsView_4,
        "rti": ui.graphicsView,
        "dti": ui.graphicsView_2,
        "rei": ui.graphicsView_3,
    }
    view_boxes=[]
    for key, view in graphicsView.items():
        view_box = view.addViewBox()
        view.setCentralWidget(view_box)  # 去除边界
        view_boxes.append(view_box)

    ui.graphicsView_5.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
    return view_boxes

def setup_image_items(color_combobox, view_boxes):
    """
    初始化图像项并添加到视图中。
    """
    image_items = {}
    for key, view_box in  enumerate(view_boxes):
        img_item = pg.ImageItem(border=None)
        view_box.addItem(img_item)
        image_items[key] = img_item

    color_combobox.setCurrentText("customize")
    set_color_map(color_combobox, image_items)
       
    return image_items


def connect_signals(ui, image_items, args):
    """
    连接控件信号与相应的槽函数。
    """
    ui.pushButton.clicked.connect(lambda: auto_update_figure(0))  # 自动更新
    ui.pushButton_2.clicked.connect(lambda:Recognition_Gesture(ui.graphicsView_5))          # 手势识别
    ui.pushButton_3.clicked.connect(openfile)                     # 打开文件
    ui.pushButton_4.clicked.connect(QtWidgets.QApplication.instance().exit)  # 退出
    ui.pushButton_6.clicked.connect(QtWidgets.QApplication.instance().exit)  # 退出
    ui.pushButton_8.clicked.connect(lambda:start_training(args))               # 开始训练
    ui.pushButton_9.clicked.connect(lambda:start_testing(args))                # 开始测试

    ui.comboBox_2.currentIndexChanged.connect(select_file)         # 文件选择
    ui.horizontalSlider.valueChanged.connect(update_display_features)  # 更新特征显示
    ui.spinBox.valueChanged.connect(update_display_features)           # 更新特征显示
    ui.comboBox.currentIndexChanged.connect(lambda: set_color_map(ui.comboBox, list(image_items.values())))  # 颜色映射


def application(args):
    """
    应用程序入口，初始化和运行主窗口。
    """
    global color_combobox, sel_combo_box, progress_bar, canvas, canvas1, Slider, Spinbox, image_items,  conf_matrix1, conf_matrix2
    global comboBox_4, comboBox_5, comboBox_6, comboBox_7, startTrainbtn, startTestbtn, autobtn, groupBox_4, pushButton_5, textEdit
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window, model_path='./save_model')

    # 记录常用控件
    color_combobox = ui.comboBox
    sel_combo_box = ui.comboBox_2
    comboBox_4 = ui.comboBox_4
    comboBox_5 = ui.comboBox_5
    comboBox_6 = ui.comboBox_6
    comboBox_7 = ui.comboBox_7
    groupBox_4 = ui.groupBox_4
    autobtn = ui.pushButton
    pushButton_5 = ui.pushButton_5
    startTrainbtn = ui.pushButton_8 
    startTestbtn = ui.pushButton_9
    progress_bar = ui.progressBar
    canvas = ui.canvas
    canvas1 = ui.canvas1
    conf_matrix1 = ui.con_matrix1
    conf_matrix2 = ui.con_matrix2
    Slider = ui.horizontalSlider
    Spinbox = ui.spinBox
    textEdit = ui.textEdit
    
    # 设置视图和图像项
    view_boxes = setup_views(ui)
    image_items = setup_image_items(color_combobox, view_boxes)
    get_color_list(color_combobox)
    # 连接信号和槽
    connect_signals(ui, image_items, args)
    # 初始化进度条
    progress_bar.setMaximum(50)  # 设定最大值为 epoch_size

    # 显示主窗口
    main_window.show()
    app.exec()

# 读取 YAML 文件
def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    gl._init()
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
    parser.add_argument('--config', default='code/examples/conf1.yaml', type=str, help='Path to config file')
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

    torch.manual_seed(777) #cuda也固定种子了

    application(args)
    