# MVRADAR: Multi-View MIMO Radar Gesture Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Qt](https://img.shields.io/badge/Qt-%23217346.svg?style=flat&logo=Qt&logoColor=white)](https://www.qt.io/)

A real-time gesture recognition system using multi-dimensional features from MIMO radar data. This project implements a multi-view feature fusion approach to accurately recognize hand gestures captured by millimeter-wave radar.

This project works with radar data collected using [RadarStream](https://github.com/Tkwer/RadarStream), a companion repository for radar data acquisition and streaming.

## 🌟 Overview

MVRADAR is a comprehensive system for real-time gesture recognition using millimeter-wave MIMO radar. The system processes multiple radar data representations (Range-Time, Doppler-Time, Range-Doppler, etc.) and fuses these multi-dimensional features to achieve robust gesture recognition performance.

The system includes:
- Data preprocessing and feature extraction
- Multi-view feature fusion using various attention mechanisms
- Model training and evaluation
- Real-time visualization and recognition interface

## ✨ Features

- **Multi-dimensional Feature Processing**: Handles 5 different radar data representations:
  - Range-Time Image (RTI)
  - Doppler-Time Image (DTI)
  - Range-Doppler Image (RDI)
  - Azimuth-Range Image (ARI)
  - Elevation-Range Image (ERI)

- **Advanced Fusion Methods**: Multiple feature fusion strategies:
  - Concatenation-based fusion
  - Attention-based fusion (Linear Projection, SE Attention, ECA Attention, Adaptive Attention)
  - Domain-specific fusion

- **Flexible Model Architecture**: Supports various backbone networks:
  - Custom CNN
  - LeNet5
  - MobileNet
  - ResNet18

- **Interactive GUI**: Real-time visualization and model training interface

- **Cross-domain Learning**: Supports domain adaptation for improved generalization

## 📁 Directory Structure

```
MVRADAR/
├── code/                      # Main source code
│   ├── bin/                   # Core functionality
│   │   ├── dataset.py         # Dataset handling
│   │   ├── train.py           # Training procedures
│   │   ├── train_utils.py     # Training utilities
│   │   └── validate_utils.py  # Validation utilities
│   ├── examples/              # Configuration examples
│   │   ├── conf1.yaml         # Configuration example 1
│   │   └── conf2.yaml         # Configuration example 2
│   ├── gui/                   # GUI components
│   │   ├── interface.py       # Main interface
│   │   ├── main.py            # GUI entry point
│   │   └── main_not_gui.py    # Command-line version
│   ├── models/                # Model definitions
│   │   ├── decoder.py         # Feature decoders
│   │   ├── encoder.py         # Feature encoders
│   │   ├── model.py           # Main model architecture
│   │   └── methods/           # Fusion methods
│   └── utils/                 # Utility functions
│       ├── checkpoint.py      # Model checkpointing
│       ├── common.py          # Common utilities
│       └── globalvar.py       # Global variables
├── dataset/                   # Dataset storage (not included in repo)
├── save_model/                # Saved model storage (not included in repo)
└── LICENSE                    # MIT License
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MVRADAR.git
   cd MVRADAR
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib pyyaml pyqtgraph PyQt5
   ```

4. Download the dataset and tools:

   - The dataset can be found in the companion repository: [Gesture-Recognition-Based-on-mmwave-MIMO-Radar](https://github.com/Tkwer/Gesture-Recognition-Based-on-mmwave-MIMO-Radar)
   - For real-time radar data acquisition and streaming, use: [RadarStream](https://github.com/Tkwer/RadarStream)

## 📊 Usage

### GUI Mode

Run the system with the graphical interface:

```bash
python code/gui/main.py
```

The GUI provides options to:
- Browse and visualize radar data
- Train models with different fusion methods
- Test trained models
- Perform real-time gesture recognition

### Command-line Mode

For batch processing or server deployment:

```bash
python code/gui/main_not_gui.py --config code/examples/conf2.yaml
```

### Configuration Files

The project includes example configuration files in the `code/examples/` directory:

#### Configuration File Structure

```yaml
# Dataset Paths (for non-GUI mode)
train_and_vali_data_dir:
  - /path/to/dataset/person_1
  - /path/to/dataset/person_2
  - /path/to/dataset/person_3
  - /path/to/dataset/person_4
  - /path/to/dataset/person_5

train_ratio: [0.8, 0.8, 0.8, 0.8, 0.8]  # Training/validation split ratio
test_data_dir: [/path/to/test/dataset]  # Test dataset path

# Model Configuration
backbone: 'resnet18'           # Network backbone
fusion_mode: "attention"       # Feature fusion mode
method: "se_attention"    # Fusion method

# Training Parameters
epochs: 50                     # Number of training epochs
batch_size: 24                 # Batch size
lr: 1.0e-3                     # Learning rate
```

**Important Notes:**
- For the GUI version (`main.py`), you can select data paths through the interface
- For the non-GUI version (`main_not_gui.py`), you must update the dataset paths in the configuration file
- The dataset should follow the structure described in the [dataset repository](https://github.com/Tkwer/Gesture-Recognition-Based-on-mmwave-MIMO-Radar)

## ⚙️ Configuration Options

### Backbone Networks
- `custom`: Custom CNN architecture
- `lenet5`: LeNet5 network
- `mobilenet`: MobileNet
- `resnet18`: ResNet18

### Fusion Modes
- `concatenate`: Simple concatenation of features
  - Methods: `add`, `concat`
- `attention`: Attention-based fusion
  - Methods: `se_attention`, `eca_attention`, `adaptive_attention`, `DScombine`

### Gesture Classes
- `Back`: Backward hand movement
- `Front`: Forward hand movement
- `Up`: Upward hand movement
- `Down`: Downward hand movement
- `Left`: Leftward hand movement
- `Right`: Rightward hand movement
- `Dblclick`: Double click gesture

## Related Repositories

- [RadarStream](https://github.com/Tkwer/RadarStream): Radar data acquisition and streaming tools
- [Gesture-Recognition-Based-on-mmwave-MIMO-Radar](https://github.com/Tkwer/Gesture-Recognition-Based-on-mmwave-MIMO-Radar): Dataset repository with detailed information about the radar data format and collection process

## Citation

If this project helps your research, please consider citing our papers:

```

```
##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
