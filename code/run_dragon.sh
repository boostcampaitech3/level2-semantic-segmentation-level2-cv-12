# Training
# --model과 --augmentation 설정 꼭 확인해주세요!
# --exp_name에 실험폴더이름을 정할 수 있습니다 / 안적어주시면 $modelClass_$augmenationClass로 폴더생깁니다.
# python baseline/train.py --model OCRNet_Hr48\
#                          --augmentation jina_aug \
#                          --epochs 200 \
#                          --lr 5e-05 \
#                          --lr_decay_step 15  \
#                          --exp_name "drgon_OCRNet_Hr48_fold2_aug" \
#                          --user dragon \
#                          --batch_size 16 \
#                          --optimizer AdamW \
#                          --fold 2 \
#                          --seed 42 \
#                          --criterion dwc \
#                          --scheduler CosineAnnealingLR
# Inference
# --model_path에 사용하고자하는 pth파일 경로를 설정해주세요
python baseline/inference.py --model_path /opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/drgon_OCRNet_Hr48_fold2_aug_4/epoch0112_mIoU06806.pth

