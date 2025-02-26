apptainer shell --nv /u/shengcao/ngc_2404/torch.sif

pip install virtualenv

python -m virtualenv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install setuptools==69.5.1

# remove pytorch and bitsandbytes dependencies in pyproject.toml
pip install -e ".[train]"
pip install xgboost --ignore-installed --no-cache-dir --no-deps
pip install huggingface_hub==0.25.0 peft==0.10.0

pip install diffusers[torch]==0.15.0

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

pip install spacy
python -m spacy download en_core_web_lg

pip install open_clip_torch
pip install timm==1.0.3
pip install git+https://github.com/cocodataset/panopticapi.git
