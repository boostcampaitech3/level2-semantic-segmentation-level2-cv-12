import pandas as pd
from tqdm import tqdm
import os

# 앙상블할 output.csv 파일들을 한 폴더에 넣어줍니다.
# 파일명 앞을 점수로 넣어서 성능이 좋은 순서대로 정렬되도록 했습니다.
output_list = os.listdir('/opt/ml/input/level2-semantic-segmentation-level2-cv-12/code/baseline/output_csv')
output_list.sort(reverse=True)

# pandas dataframe으로 만들어줍니다.
df_list = []

for output in output_list:
    df_list.append(pd.read_csv(f'/opt/ml/input/level2-semantic-segmentation-level2-cv-12/code/baseline/output_csv/{output}'))

# submission dataframe
submission = pd.DataFrame()
submission['image_id'] = df_list[0]['image_id']

# pixel-wise hard voting 진행
PredictionString = []

for idx in tqdm(range(len(df_list[0]))):
    # 각 모델이 뽑은 pixel 넣을 리스트
    pixel_list = []
    
    for i in range(len(df_list)):
        pixel_list.append(df_list[i]['PredictionString'][idx].split(' '))

    result = ''

    for i in range(len(pixel_list[0])):
        pixel_count = {'0' : 0, '1' : 0, '2' : 0, 
                      '3' : 0, '4' : 0, '5' : 0,
                      '6' : 0, '7' : 0, '8' : 0,
                      '9' : 0, '10' : 0}
        
        # 각 모델이 뽑은 pixel count
        for j in range(len(pixel_list)):
            pixel_count[pixel_list[j][i]] += 1
        
        # 제일 많이 vote된 pixel 값
        voted_pixel = [key for key, value in pixel_count.items() if value == max(pixel_count.values())]

        # voted_pixel이 1개인 경우
        if len(voted_pixel) == 1:
            result += voted_pixel[0] + ' '
        # 동점이 나온 경우
        else:
            # 성능이 좋았던 모델부터 값이 voted_pixel에 있다면 result로 고르기
            for j in range(len(pixel_list)):
                pixel_candidate = pixel_list[j][i]

                if pixel_candidate in voted_pixel:
                    result += pixel_candidate + ' '
                    break
    
    # 마지막 공백 제거
    result = result[:-1]

    PredictionString.append(result)

# submission csv 만들기
submission['PredictionString'] = PredictionString

folder_path = '/opt/ml/input/code'
folder_list = os.listdir(folder_path)
n = 1
while True:
    file_name = 'hard_voted_output' + str(n) + '.csv'
    if file_name in folder_list:
        n += 1
    else:
        csv_path = folder_path + '/' + file_name
        break    

submission.to_csv(csv_path, index=False) 