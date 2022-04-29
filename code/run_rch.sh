# Training
# --model과 --augmentation 설정 꼭 확인해주세요!
# --exp_name에 실험폴더이름을 정할 수 있습니다 / 안적어주시면 $modelClass_$augmenationClass로 폴더생깁니다.
python baseline/train.py --model UNet_Resnet50 \
                        --augmentation Rch_augmentation \
                        --epochs 50 \
                        --lr 1e-4 \
                        --lr_decay_step 15 \
                        --exp_name "" \
                        --user rch
                        --batch_size 16

# Inference
# --model_path에 사용하고자하는 pth파일 경로를 설정해주세요
#python baseline/inference.py --model_path /opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/BaseModel/epoch0048_mIoU06001.pth --model Deeplabv3_Resnet50