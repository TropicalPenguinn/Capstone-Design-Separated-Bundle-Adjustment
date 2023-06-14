# Capstone-Design-Separated-Bundle-Adjustment

## 요약
최근 디지털 트윈,자율주행,로봇의 발달로 산업에서 3D Reconstruction의 쓰임이 다양해지고 보행자 등을 3D 모델로 구축하여, 차량이 적절한 판단을 내릴 수 있도록 도와준다. 로봇 산업에서는 로봇의 환경 인식과 제어에 3D 재구성 기술이 매우 중요하다. 로봇이 주변 환경을 정확하게 파악하여 작업을 수행할 수 있도록 3D 모델을 이용하면 보다 정확하고 효과적인 제어가 가능하다. 3D Reconstruction이 완벽하게 되기 위해서는 이론적으로 모든 점과 카메라의 위치 및 내부 매개 변수를 정확하게 추정해야 하지만 센서의 한계나 노이즈에 의해 추정 오차가 발생할 경우 3D 모델의 정확도가 저하된다. 따라서 Bundle Adjustment 같은 최적화 기술을 이용하여 추정 오차를 최소화하고, 3D Reconstruction  품질을 향상시킬 수 있다.

## 연구배경
 3D Reconstruction은 디지털 트윈,자율주행,로봇 산업 등에서 중요한 역할을 한다. 디지털 트윈은 물리적인 시스템이나 제품의 디지털 복제물로, 이를 이용하여 시스템이나 제품의 성능, 상태 등을 실시간으로 모니터링하고 분석할 수 있다. 이를 위해서는 물리적인 시스템이나 제품의 모델링이 필요하며, 이때 3D Reconstruction 기술이 매우 중요한 역할을 한다. 자율주행 분야에서 3D Reconstruction 기술을 이용하면 도로 상황, 건물,신호등,보행자 등을 3D 모델로 구축하여, 차량이 적절한 판단을 내릴 수 있도록 도와준다. 로봇 산업에서는 로봇의 환경 인식과 제어에 3D 재구성 기술이 매우 중요하다. 로봇이 주변 환경을 정확하게 파악하여 작업을 수행할 수 있도록 3D 모델을 이용하면 보다 정확하고 효과적인 제어가 가능하다. 3D Reconstruction이 완벽하게 되기 위해서는 이론적으로 모든 점과 카메라의 위치 및 내부 매개 변수를 정확하게 추정해야 하지만 센서의 한계나 노이즈에 의해 추정 오차가 발생할 경우 3D 모델의 정확도가 저하된다. 따라서 Bundle Adjustment 같은 최적화 기술을 이용하여 추정 오차를 최소화하고, 3D Reconstruction 품질을 향상시킬 수 있다.
 
 ## 연구 목표
 본 연구는 초기 특징점 위치 추정에서 생기는 오차를 해결하여 Bundle Adjustment가 최적의 Pose와 3D Point를 찾음으로써 Reprojection Error을 최소화하는 것을 목표로 한다.있다. 본 연구에서는 3D Reconstruction의 품질을 높이는 최적화 방법중에 하나인 Bundle Adjustment의 한계점을 개선한 Seperated Bundle Adjustment를 구현한다.
 
 ## 기존 연구의 문제점
 Bundle Adjustment는 두 가지 한계점이 있다. 첫번째는 Bundle Adjustment의 성능이 Feature Matching의 품질에 크게 의존한다는 것이다. Bundle Adjustment의 Cost function은 임의의 이미지의 특징점을 대응하는 다른 이미지의 Camera에 대하여 Reprojection 했을 때 Reprojection된 Pixel과 대응하는 Pixel 간의 픽셀차로(Reprojection Error) 정의하고 이것을 최소화하는 방향으로 카메라 포즈와 매개변수 특징점의 3차원 Points의 위치를 조정한다. 만약 초기 Feature Matching이 제대로 되지 않는다면 Bundle Adjustment를 사용하더라도 최적의 Pose와 3차원 위치를 얻을 수 없다. 이런 경우는 RANSAC 방식을 사용하거나 Deep Learning  을 이용하여 Robust한 Feature Matching을 찾는 연구가 진행되고 있다 [4]. 두번째는 그림 2처럼 초기 특징점의 위치에 오차가 있는 경우, Bundle Adjustment을 사용하더라도 최적화의 한계가 있다는 것이다. 디지털 이미지는 유한한 해상도를 가지기 때문에, 실제 물체를 픽셀로 변환했을 때 위치 차이가 생길 수 있다. 그리고 카메라 센서 자체 노이즈로 인해서도 특징점 위치에 오차가 발생한다.  
 
 ## Separated Bundle Adjustment
 기존 Bundle Adjustment 방식은 특징점 위치의 오차를 해결할 수 없다 이. 문제를 해결하기 위해 본 연구는 특징점의 위치 또한 최적화 변수로 정의 하고 ADMM 방식으로 문제를 분할하여 최적화를 진행하는 Separated Bundle Adjustment  알고리즘을 제안한다. 먼저 그림 3처럼 기존 Bundle Adjustment Reprojection Error 를 사용하여 가 최소화 되는 방향으로 카메라 포즈와 3차원 점들의 위치를 조정한다 . 그 다음 카메라 포즈와 3차원 점들의 위치를 고정하고 Reprojection Error가 최소화 되는 방향으로 그림 4처럼 특징점의 위치를 조정한다 . 이 방법을 통해 초기 특징점 위치 오차를 줄일 수 있다  두 과정을 반복하며 reprojection error 을 최소화하는 카메라 포즈 , 3차원  점들의 위치 특징점의 위치를 계산한다. 
라는 문제에서 아래 두 과정은 T,X,p 가 수렴할 때까지 k=1,2,3... 에 대해서 반복한다. 
변수들을 분리하여 최적화를 수행하기 때문에 Separated Bundle Adjustment라고 부른다.
