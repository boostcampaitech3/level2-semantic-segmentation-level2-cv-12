# Training
# --model과 --augmentation 설정 꼭 확인해주세요!
python baseline/train.py --model BaseModel --augmentation BaseAugmentation --epochs 1 --lr 1e-4 --lr_decay_step 10


# Inference
# --model_path에 사용하고자하는 pth파일 경로를 설정해주세요
python baseline/inference.py --model_path /opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/BaseModel_BaseAugmentation/epoch0001_mIoU03327.pth