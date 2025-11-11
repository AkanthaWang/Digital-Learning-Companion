#!/bin/bash

# Step 1: Create and activate conda environment
echo "Creating and activating the conda environment..."
conda create -n EmotiVoice python=3.8 -y
conda init
conda activate EmotiVoice

# Step 2: Install git-lfs
echo "Installing git-lfs..."
if [ "$(lsb_release -i | grep 'Ubuntu')" ]; then
  sudo apt update
  sudo apt install -y git git-lfs
elif [ "$(lsb_release -i | grep 'CentOS')" ]; then
  sudo yum update
  sudo yum install -y git git-lfs
else
  echo "Unsupported OS. Please install git and git-lfs manually."
  exit 1
fi

# Step 3: Clone the EmotiVoice repository
echo "Cloning the EmotiVoice repository..."
git lfs install
git lfs clone https://github.com/netease-youdao/EmotiVoice.git

# Step 4: Install dependencies
echo "Installing required Python packages..."
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin pypinyin_dict
python -m nltk.downloader "averaged_perceptron_tagger_eng"

# Step 5: Download pre-trained models
echo "Downloading pre-trained models..."
cd EmotiVoice
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
git clone https://www.modelscope.cn/syq163/WangZeJun.git
git clone https://www.modelscope.cn/syq163/outputs.git

# Step 6: Install Streamlit for UI interface
echo "Installing Streamlit..."
pip install streamlit

# Step 7: Run the Streamlit UI interface
echo "Running the Streamlit UI interface..."
streamlit run demo_page.py --server.port 6006 --logger.level debug

# Step 8: Install FastAPI for API service
echo "Installing FastAPI and other dependencies..."
pip install fastapi pydub uvicorn[standard] pyrubberband


