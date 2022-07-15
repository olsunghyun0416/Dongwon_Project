################################################################
# 타이타닉 생존 예측 웹 서비스
################################################################
# -------------------------------------------------------------
# 1. 라이브러리 로딩
from flask_restful import reqparse 
from flask import Flask
import numpy as np 
import pandas as pd
import joblib
import json
import import_ipynb
from 더반찬_캐글_파이프라인 import *

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언
features = ['Pclass', 'Sex', 'Age', 'SibSp' ,'Parch', 'Fare', 'Embarked']
target = 'REAL_ORD_QTY'

cat = {'GOODS_NO':[1712010310, 14375, 3273, 13957, 1901012353, 13529, 1803010879, 
                   2012014895, 2204016772, 1823, 4092, 14985, 1808011751, 1903012675, 
                   2010014667, 1806011271, 2103015231, 2010014707, 1805011121, 1909013359]}


def DBC_PIPELINE(df):
    temp = df.copy()
   
    temp = ADD_DATETIME(temp)

    temp = SHIFT(temp)

    temp = FE(temp)
    
    temp = ROLLING(temp)

    temp = LAG(temp)

    temp = TREND(temp)

    temp = MEAN_DATA(temp)

    temp = REPLACE_MISSING_VALUE(temp)

    x_cols = list(temp)
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
    x_input = DBC_PIPELINE(x_input)

    # 예측. 결과는 넘파이 어레이
    pred = model.predict(x_input) 
    
    # 결과를 식별가능한 문자로 변환(0,1로 반환할 때, 타입오류가 날 수 있음.)
    result = pred
    
    # result : json 형태로 전송해야 한다.
    out = {'pred': list(result)} 
   
    return out

# -------------------------------------------------------------
# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    # imputer1 = joblib.load('preprocess/imputer1_ti1.pkl')
    # imputer2 = joblib.load('preprocess/imputer2_ti1.pkl')
    # scaler = joblib.load('preprocess/scaler_ti1.pkl')
    # model = joblib.load('model/model_ti1.pkl')
    
    # 웹서비스 실행 및 접속 정보
    app.run(host='127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
