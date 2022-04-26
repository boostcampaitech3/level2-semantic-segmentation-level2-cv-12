![image](https://user-images.githubusercontent.com/57215124/165293846-462a4303-f5c5-4909-9cea-75992f4d8c92.png)

# Implementation
### 구현완료
- `Model.py`
- `train.py`
- `dataset.py`
- `inference.py`


### 추가 구현 필요
- `wandb`
- `시각화`
- `scheduler`(`현재 stepLR`)
- `keep ckpt` 기능
- `logging` 기능

<br>
<br>

# Code
### 경로
- `opt/ml/input/code/`
- weight, csv파일 저장경로 --> `opt/code/saved/모델클래스이름/` (디렉토리 없으면 자동생성)
- 중복된 `model` class를 사용했을시 `opt/code/saved/모델클래스이름2/`, `opt/code/saved/모델클래스이름3/`...식으로 폴더가 생성됨. 이 경우엔 `inference`시에 `--model_path` 인자 사용하여 경로(`pth파일`) 지정해줘야함

<br>

### 기능 추가방법
- __Model__: `Model.py`에 사용하고 싶은 모델 `class`로 구현하여 추가하고 `train.py`에서 `--model`파라미터 변경
- __Augmentation__: `dataset.py`에 사용하고 싶은 augmentation `class`로 구현하여 추가하고 `train.py`에서 `--augmentation`파라미터 변경


<br>

## 쉘스크립트
- `code/` 디렉토리에 본인 `.sh`파일 만들어서 사용하는 것을 권장

### Training
- `--model`, `--augmentation` 확인필수 
```bash
python train.py
```
or 
```bash
python train.py --model $modelClass 
                --augmentation $augmentationClass 
                --epochs $epochs 
                --batch_size $batch_size 
                --optimizer $optimizer
```

<br>

### Inference
- `--model`, `--model_path` 확인 필수
```bash
python inference.py
```
or
```bash
python inference.py --model $modelClass
                    --batch_size $batch_size
                    --model_path $model_path
```
