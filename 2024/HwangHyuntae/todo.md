
# ISSUES

## 1. eta_function

BSG(belt driven starter generator)
모터, 발전기 둘 다 사용되는 모델임

- 모터로 사용하는 경우는 엔진(모터)을 처음 시작할 때 이게 회전력을 제공해서 시동 거는걸을 도움 (모터)
- 차량 가속 시에 토크 제공 (모터)
- (경사, 브레이크 등) 여기서 회생제동을 해서 에너지를 생성함 (발전기)

그래서 이 BSG의 효율 함수(eta-function)를 구현하려면 efficiency map이 필요하다
이건 모터에 효율 map이랑 동일하고 (당연히 BSG도 모터니까) 그걸 가져와서 array 형태로 저장해놓고 현재 값에 대해 interpolate해서 쓰는 형식인듯
일단 BSG의 효율 map은 BSG rpm과 torque 2변수에 대해서 efficiency가 결정되는 형태의 eta = f (w, T) 꼴임.
따라서 먼저 map을 구해와야한다

## 2. gear 값

n_g(gear number) : gear 값 인듯
gear 값도 차량 모델링에서 변화를 설정해주기 나름
이건 일단 현대차의 자동변속기에서부터 기어수에 따른 기어비를 가져와서 넣었다

## 3. eta_transmission  

똑같음 eta_bsg랑. 얘도 효율 map이 필요하다. 마찬가지로 2변수 함수 형태이고. ~~~ table로 치면 나오긴 하는데 사실 2변수 함수의 경우는 table로 표현하기가 힘들기 때문에, 예쁜 데이터를 찾아 오든 누가 만들어놓은 map을 이미지 인식을 통해 다시 3차원 table로 가져오든 해야할 듯? 애초에 이걸 어떻게 array 형태로 가져올지도 살짝 미지수...

-----------------------------------------------------------

## 11.08 이후  

최적화 대상은 SoC, 총 주행시간, 평균 속도, 연료 소비율 이 네 가지이다.
각자 리워드 설정을 생각해보면  

- SoC : 0.7에 값이 최대한 가까워야함
- 총 주행시간 : 짧을 수록 좋음
- (???)평균 속도 -> 조금 어려움. "안정적" 이면 좋음. 급격한 속도 변동이 없으면 좋다는 건데 이건 바뀔수도.
- 연료 소비율 -> 작을수록 좋음  

- 드디어 cycle의 의미를 이해했다... 통일된 조건 하에서 비교하게 위해서 속도 프로파일을 다 미리 정의를 해둔 거였구나.
원본 논문에서도 애초에 속도와 주행 시간을 정의한 적이 없었다는게 가장 큰 함정인듯...
일단 그럼 최적화 대상을 약간 수정해야할 듯 싶다.

-----------------------------------------------------------

1. required torque
required torque는 before velocity, current_velocity, time_step_size를 통해서 구할 수 있을 것 같다

견본은

```python

```

정도가 될 것 같은데 이걸 이용해서 필요 torque를 계산하고, action space에서 engine, motor torque를 분배해서 넣어주면 될 것 같아보인다. 
그래서 state에 일단 prev_vel 값이 필요할 듯 

2. torque 분배

required torque를 계산한 후에
현재 차량의 rpm에 따라 낼 수 있는 최대 torque를 엔진의 토크 맵에서부터 가져온 이후에
torque 분배 시 limitation에 따라 다음 분배를 어떻게 할 지 제약을 줘서 해주면 될 것 같다.

T_brk 가 정의되기 위해서는 T_req(정해짐), T_eng, T_bsg 이 두개의 action을 ratio가 아니라 값 자체를 넣어줘야할 것 같은데
ratio로 넣어주게 된다면 T_req = T_eng + T_bsg 이렇게 보장되어야만 할텐데
어떻게 하나?

일단 ratio 말고 값으로 정의 해놓으려 함