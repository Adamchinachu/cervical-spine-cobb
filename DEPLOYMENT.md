# 颈椎角度分析工具部署指南

## 环境要求
- Python 3.10
- Conda或Miniconda

## 方法一：使用environment.yml（推荐）
```bash
# 1. 创建新环境
conda env create -f environment.yml

# 2. 激活环境
conda activate vertebrae_analysis

# 3. 验证安装
python -c "import cv2, torch, ultralytics; print('环境配置成功')"
```

## 方法二：使用pip安装
```bash
# 1. 创建Python虚拟环境
python -m venv vertebrae_env

# 2. 激活环境
# Windows:
vertebrae_env\Scripts\activate
# Linux/Mac:
source vertebrae_env/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 方法三：conda-pack打包（适合离线部署）
```bash
# 在原电脑上：
conda install conda-pack
conda pack -n labelme_env -o vertebrae_analysis.tar.gz

# 在目标电脑上：
mkdir vertebrae_analysis
tar -xzf vertebrae_analysis.tar.gz -C vertebrae_analysis
# Windows用户可以用7-zip等工具解压

# 激活环境
source vertebrae_analysis/bin/activate  # Linux/Mac
vertebrae_analysis\Scripts\activate.bat  # Windows
```

## 文件清单
确保以下文件都复制到新电脑：
- `gui_interface.py` - 主GUI程序
- `new_script.py` - 核心处理逻辑
- `c2_c7.pt` - 椎体检测模型
- `leftright.pt` - 左右判断模型
- `requirements.txt` - Python依赖包
- `environment.yml` - Conda环境配置

## 运行程序
```bash
# 激活环境后运行
python gui_interface.py
```

## 常见问题
1. **CUDA问题**：如果目标电脑没有NVIDIA GPU，需要安装CPU版本的PyTorch
2. **路径问题**：确保模型文件(.pt)与程序文件在同一目录
3. **权限问题**：确保程序有读写Excel文件夹的权限

## GPU版本vs CPU版本
- GPU版本：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- CPU版本：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
