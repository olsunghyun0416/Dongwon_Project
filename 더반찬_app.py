################################################################
# 더반찬 판매량 예측 웹 서비스
################################################################
# -------------------------------------------------------------
# 1. 라이브러리 로딩
from flask_restful import reqparse 
from flask import Flask
import numpy as np 
import pandas as pd
import joblib
import json

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언
features = ['Pclass', 'Sex', 'Age', 'SibSp' ,'Parch', 'Fare', 'Embarked']
target = ['GOODS_NM__심방골주부X더반찬_ 시골 돼지짜글이(600g)',
       'GOODS_NM_가정집 오징어불고기/셀프(380g)', 'GOODS_NM_건고사리나물볶음(150g)',
       'GOODS_NM_건표고버섯볶음', 'GOODS_NM_고구마 품은 라자냐(450g)',
       'GOODS_NM_고소한도토리묵무침(360g)', 'GOODS_NM_꼬막무침 (260g)',
       'GOODS_NM_두메산나물비빔밥재료', 'GOODS_NM_메밀소바(2인분)', 'GOODS_NM_셀프두부조림(600g)',
       'GOODS_NM_소고기유니짜장소스(1인분, 200g)', 'GOODS_NM_수제계란말이(350g)',
       'GOODS_NM_숙주나물(300g)', 'GOODS_NM_순살코다리강정(180g)', 'GOODS_NM_양장피',
       'GOODS_NM_열무비빔밥재료믹스(2인분)', 'GOODS_NM_옛날잡채(500g)',
       'GOODS_NM_우삼겹숙주볶음(250g)', 'GOODS_NM_채소계란찜(340g)',
       'GOODS_NM_한돈 제육볶음(700g)']
imputer1_list = ['Embarked']
cat = {'Sex':["female", "male"], 'Embarked':["C", "Q", "S"], 'Pclass':[1,2,3]}

# 필요한 함수 선언
def DBC_DATAPIPELINE(df, scaler):
    temp = df.copy()

    temp = DBC_DATA_REMOVE(temp)

    temp = DBC_SET_OR_SINGLE(temp)

    temp = DBC_CHANGE_TYPE(temp)

    temp = DBC_FILL_NAN(temp)

    temp = DBC_FEATURE_ENGINEERING(temp)

    temp = REMOVE_COLUMNS(temp)

    temp = DBC_OUTLIER(temp)

    temp = DBC_GET_DUMMIES(temp)

    temp = DBC_RESAMPLING(temp)

   
    # 타겟이 되는 20개 제품을 설정한다
    target = ['GOODS_NM__심방골주부X더반찬_ 시골 돼지짜글이(600g)',
       'GOODS_NM_가정집 오징어불고기/셀프(380g)', 'GOODS_NM_건고사리나물볶음(150g)',
       'GOODS_NM_건표고버섯볶음', 'GOODS_NM_고구마 품은 라자냐(450g)',
       'GOODS_NM_고소한도토리묵무침(360g)', 'GOODS_NM_꼬막무침 (260g)',
       'GOODS_NM_두메산나물비빔밥재료', 'GOODS_NM_메밀소바(2인분)', 'GOODS_NM_셀프두부조림(600g)',
       'GOODS_NM_소고기유니짜장소스(1인분, 200g)', 'GOODS_NM_수제계란말이(350g)',
       'GOODS_NM_숙주나물(300g)', 'GOODS_NM_순살코다리강정(180g)', 'GOODS_NM_양장피',
       'GOODS_NM_열무비빔밥재료믹스(2인분)', 'GOODS_NM_옛날잡채(500g)',
       'GOODS_NM_우삼겹숙주볶음(250g)', 'GOODS_NM_채소계란찜(340g)',
       'GOODS_NM_한돈 제육볶음(700g)']
    temp = temp.drop(target, axis=1)
    x_cols = list(temp)
    temp = scaler.transform(temp)

    return pd.DataFrame(temp, columns=x_cols)



# -------------------------------------------------------------
# 3. 웹서비스

@app.route('/predict/', methods=['POST']) 
def predict(): 
    
    # 입력받은 json 파일에서 정보 뽑기(파싱)
    parser = reqparse.RequestParser() 
    for v in features :
        parser.add_argument(v, action='append') 

    # 뽑은 값을 딕셔너리로 저장
    args = parser.parse_args() 
        
    # 딕셔너리를 데이터프레임(2차원으로 만들기)
    x_input = pd.DataFrame(args)

    # 전처리
    x_input = DBC_DATAPIPELINE(x_input, scaler)

    # 예측. 결과는 넘파이 어레이
    pred = model.predict(x_input) 
    
    # 결과를 식별가능한 문자로 변환(0,1로 반환할 때, 타입오류가 날 수 있음.)
    result = np.where(pred == 0, 'Died','Survived')
    
    # result : json 형태로 전송해야 한다.
    out = {'pred': list(result)} 
   
    return out

# -------------------------------------------------------------
# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    imputer1 = joblib.load('preprocess/imputer1_ti1.pkl')
    imputer2 = joblib.load('preprocess/imputer2_ti1.pkl')
    scaler = joblib.load('preprocess/scaler_ti1.pkl')
    model = joblib.load('model/model_ti1.pkl')
    
    # 웹서비스 실행 및 접속 정보
    app.run(host='127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
