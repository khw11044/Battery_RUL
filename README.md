# Battery Remaining Useful Life (RUL)
## 배터리 수명 예측 모델링 

[Battery RUL 캐글](https://www.kaggle.com/datasets/ignaciovinuales/battery-remaining-useful-life-rul/data)

## 데이터셋 

하와이 자연 에너지 연구소(Hawaii Natural Energy Institute)는 공칭 용량이 2.8Ah인 NMC-LCO 18650 배터리 14개를 검사했습니다. 이 배터리는 25°C에서 C/2 속도의 CC-CV 충전 속도와 1.5C의 방전 속도로 1000회 이상 순환되었습니다.

해당 소스 데이터 세트에서 각 사이클에 대한 전압 및 전류 동작을 보여주는 기능을 만들었습니다. 이러한 기능은 배터리의 잔여 유효 수명(RUL)을 예측하는 데 사용될 수 있습니다. 데이터 세트에는 14개 배터리의 요약이 포함되어 있습니다.

### 변수들 

- Cycle Index: number of cycle
- F1: Discharge Time (s) -> 최소 전압에 도달하는 시간 (방전 시간).
- F2: Time at 4.15V (s) -> 전압이 4.15V에 도달하는 시간.
- F3: Time Constant Current (s) -> 전류가 최대값으로 일정하게 유지되는 시간.
- F4: Decrement 3.6-3.4V (s) -> 전압이 3.6V에서 3.4V로 감소하는 데 필요한 시간.
- F5: Max. Voltage Discharge (V) -> 전압의 초기값과 최대값.
- F6: Min. Voltage Charge (V) -> 전압의 초기(최소) 값.
- F7: Charging Time (s) -> 전체 충전 시간 
- Total time (s) 
- RUL: target

여기서 RUL은 remaining-useful-life cycles이다. 하나의 완전한 사이클은 전체 충전 및 방전이다. 즉, RUL이 80이면 배터리는 80 사이클 이상 지속될 것으로 예상한다는 뜻이다.

## Objectve

- 전압, 전류 및 시간을 기반으로 소스 데이터 세트에서 새로운 기능을 추출하고 생성합니다.
- PyTorch를 사용하여 feedforward 및 LSTM 신경망을 개발하여 배터리의 잔여 유효 수명(RUL)을 예측합니다.

Motivation: 배터리의 RUL은 일반적으로 용량(mAH)으로 추정할 수 있습니다. 그런데 용량을 측정할 수 없다면 어떻게 될까요? 프로젝트의 기본 아이디어는 전압(V)과 전류(A)만 측정하는 RUL을 예측하는 것입니다.

Source datasets: The public datasets can be found here: https://www.batteryarchive.org/list.html 14 databases are selected from the HNEI source. The .csv files are the time series named 'HNEI_18650_NMC_LCO_25C_0-100_0.5/1.5C_'.

## Data Preprocessing 

공개적으로 사용 가능한 배터리 수명 주기 데이터베이스는 이 프로젝트에 즉시 사용할 수 있는 데이터를 제공하지 않습니다. 

이 데이터베이스들은 전압(V), 전류(A), 시간(S), 방전 및 충전 용량(Ah), 충전 및 방전 에너지(Wh)와 같은 다양한 변수를 포함하고 있지만, 모든 변수들을 이 프로젝트에 사용될 수 있는 것은 아닙니다.

**목표는 오직 전압, 전류 및 시간만 입력으로 사용하는 것입니다.**
그러나 이 변수들을 직접 입력으로 사용하는 것은 의미 없는 정보를 제공하며 모델을 생성하기에 충분하지 않기 때문에 실행 가능하지 않습니다.

따라서, 이것들을 기반으로 새로운 특성을 개발하기 위해 처리하고 조작해야 합니다. 신경망이 훈련할 때 이 새로운 특성들을 사용할 예정입니다.

요약하자면, 전압, 전류, 시간을 사용하여 원본 데이터셋에서 일곱 가지 특성이 생성됩니다. 이 특성들을 사용하여 배터리의 잔여 사용 가능 기간(RUL)을 예측하는 것이 목표입니다.


![Voltage Charging Cycle](https://github.com/khw11044/Basic-RL-for-Process-Control/assets/51473705/6ad1d596-e10b-47c5-99da-7efdd17c3412)

![Voltage Discharging Cycle](https://github.com/khw11044/Basic-RL-for-Process-Control/assets/51473705/fbb396a7-5472-4f2c-a91c-447013b1f3b6)

![Current Charging Cycle](https://github.com/khw11044/Basic-RL-for-Process-Control/assets/51473705/7b05d1d4-0056-4256-b64b-fc803ec86692)


![image](https://github.com/DatrikIntelligence/Stacked-DCNN-RUL-PHM21/assets/51473705/5381c631-0737-4c9b-abde-8066a658f41f)

이번에 from ydata_profiling import ProfileReport를 처음알게 되었는데 정말 데이터분석할때 좋은거 같다. 
> pip install ydata_profiling


## 데이터셋 분석 

![image](https://github.com/khw11044/Battery_RUL/assets/51473705/847398c2-6945-46c3-bccb-c4d5d9bc57c2)

![image](https://github.com/khw11044/Battery_RUL/assets/51473705/97c7359b-4f70-4d82-988a-c417cc8e5b6e)

![image](https://github.com/khw11044/Battery_RUL/assets/51473705/6a46555d-1722-4deb-8d61-ce25b0d3e401)

RUL_metric : RUL 예측값과 실제 RUL 값과의 차이가 100이하면 옳게 예측한것으로 측정한다.

## ML 결과 

Test set

![image](https://github.com/ignavinuales/Battery_RUL_Prediction/assets/51473705/5d97d951-7693-44c3-ad9a-ba80add170d7)

RandomForestRegressor

- RMSE : 29.975473132963696
- MAE  : 15.304969420021964
- MAPE : 0.18772798208130972
- R2 : 0.9913132950785044
- RUL_metric : 0.9871279163314561

![image](https://github.com/khw11044/Iron-CCM-RUL/assets/51473705/a6b23886-d9ab-4e8b-8189-d1fb60d3bd46)

## DL 결과 

![image](https://github.com/khw11044/MY_GOAD/assets/51473705/2ff7b5bd-b12c-432e-9587-5cdf98ad8e30)


- RMSE : 61.69550065151493
- MAE  : 46.079332333534644
- MAPE : 1827158615447511.2
- R2 : 0.9632015131060703
- RUL_metric : 0.9519308125502816

![image](https://github.com/khw11044/MY_GOAD/assets/51473705/c8d9316a-6a92-4c6d-b0f4-97fb6237e8ec)

![image](https://github.com/khw11044/MY_GOAD/assets/51473705/7496c735-e804-46b7-a383-69c69ea2e241)

____________

[원본 깃헙 주소](https://github.com/ignavinuales/Battery_RUL_Prediction/tree/main)

[마이크로소프트 BatteryML](https://github.com/microsoft/BatteryML/tree/main)

[배터리 용어의 모든 것을 알아보자](https://www.samsungsdi.co.kr/column/technology/detail/56402.html?listType=gallery)
