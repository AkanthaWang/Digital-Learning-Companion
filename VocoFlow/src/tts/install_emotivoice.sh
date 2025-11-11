#!/bin/bash

# Step 1: Install git-lfs
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

# Step 2: Clone the EmotiVoice repository
echo "Cloning the EmotiVoice repository..."
git lfs install
git lfs clone https://github.com/netease-youdao/EmotiVoice.git

# Step 3: Install dependencies
echo "Installing required Python packages..."
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin pypinyin_dict
python -m nltk.downloader "averaged_perceptron_tagger_eng"

# Step 4: Download pre-trained models
echo "Downloading pre-trained models..."
cd EmotiVoice
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
git clone https://www.modelscope.cn/syq163/WangZeJun.git
git clone https://www.modelscope.cn/syq163/outputs.git

# Step 5: Install Streamlit for UI interface
echo "Installing Streamlit..."
pip install streamlit

# Step 6: Run the Streamlit UI interface
echo "Running the Streamlit UI interface..."
streamlit run demo_page.py --server.port 6006 --logger.level debug

# Step 7: Install FastAPI for API service
echo "Installing FastAPI and other dependencies..."
pip install fastapi pydub uvicorn[standard] pyrubberband


