## 공통 model 설정
# model=Deeplabv3_Resnet101
# model=BaseModel
model=STL

# Training
# --model과 --augmentation 설정 꼭 확인해주세요!
# --exp_name에 실험폴더이름을 정할 수 있습니다 / 안적어주시면 $modelClass_$augmenationClass로 폴더생깁니다.
python baseline/train.py --model $model \
                         --augmentation Rotate90_CropNonEmptyMaskIfExists \
                         --epochs 70 \
                         --lr 1e-5 \
                         --lr_decay_step 10 \
                         --exp_name "" \
                         --user hongrok \
                         --batch_size 4 \
                         --resize "[512, 512]"


# Inference
# --model_path에 사용하고자하는 pth파일 경로를 설정해주세요

# 가장 최근 저장된 모델 사용하기
pwd=$(pwd)
dir=${pwd%/*}/saved
pth=$(ls $dir/*/*.pth -tr | tail -1)

# python baseline/inference.py --model $model \
#                              --model_path $pth
