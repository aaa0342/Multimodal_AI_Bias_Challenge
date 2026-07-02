# 2026 성균관대학교 멀티모달 AI Bias 챌린지

## 팀 소개

### 팀명 
나프탈렌

### 팀원
동국대학교 인공지능전공 박준서

동국대학교 정보통신학과 강민제

동국대학교 컴퓨터Ai학과 이형준

## 대회 개요

### 요약
이미지-텍스트 기반 질의응답 상황에서 AI 모델이 주어진 정보에 근거해 적절한 선택지를 예측하고, 판단이 어려운 상황에서는 불확실성을 올바르게 인식하는 능력을 평가하는 대회

### 설명
이미지와 텍스트로 구성된 질의응답 데이터를 이해하고, 주어진 질문에 대해 가장 적절한 선택지를 예측하는 멀티모달 AI 모델을 개발하는 대회입니다.
참가자는 성별, 인종·민족 등 사회적 맥락이 포함된 문항에서 명확한 근거가 있는 경우에는 올바른 답을 선택하고, 판단에 필요한 정보가 부족한 경우에는 섣불리 추론하지 않는 모델을 구현해야 합니다.
본 대회에서는 평가 데이터만 제공되며, 학습 데이터는 제출 형식 및 문제 이해를 위한 예시 샘플 1개만 제공됩니다. 추가 학습 데이터는 참가자가 직접 구성하여 활용할 수 있습니다.

### 추론 환경
기준 평가 환경: RTX A6000 48GB, Python 3.10, CUDA 12.4, PyTorch 2.6.0, Ubuntu 20.04

## 파이프라인 소개

본 프로젝트의 최종 파이프라인은 **Selective Verification with Label-Prior Debiasing** 전략을 기반으로 한다.  
핵심 목표는 멀티모달 QA에서 이미지가 제공될 때 발생할 수 있는 성별, 인종, 표정, 외형 등 시각적 단서 기반 편향을 줄이면서도, 실제로 이미지 정보가 필요한 문제에서는 선택적으로 이미지를 활용해 정답률을 높이는 것이다.

전체 파이프라인은 크게 **Label-prior Measurement → Context-only Judgement → Image Necessity Assessment → Multimodal Verification → Final Prediction** 단계로 구성된다.

### 1. Label-prior Measurement

먼저 모델이 특정 선택지 번호(label)에 대해 갖는 고유한 선호도를 측정한다.  
이를 위해 context, question, choices를 모두 `N/A`로 둔 neutral prompt를 입력하고, 각 label에 대한 prior logit을 추정한다.

이 과정은 context-only 단계와 multimodal verification 단계에서 각각 수행된다.  
즉, 텍스트만 입력받는 1차 예측용 prior와 이미지까지 포함하는 2차 검증용 prior를 별도로 측정한다.

### 2. Context-only Judgement

1차 예측에서는 이미지를 사용하지 않고, 오직 `Context`, `Question`, `Choices`만을 입력으로 사용한다.  
이는 이미지에 포함된 성별, 표정, 인종, 외형적 특징 등이 모델의 판단에 불필요하게 개입하는 것을 방지하기 위한 단계이다.

모델은 다음 세 가지 값을 출력한다.

- `Evidence`: 정답 판단에 사용된 최소한의 텍스트 근거
- `NeedImage`: 이미지 검증이 필요한지 여부
- `Answer`: context-only 기반 1차 예측 label

이 단계에서는 명시적인 텍스트 근거가 충분하면 이미지를 보지 않고 답을 결정하고, 텍스트만으로 판단이 어렵거나 객관적 시각 정보가 필요한 경우에만 이후 multimodal verification 단계로 넘긴다.

### 3. Image Necessity Assessment

1차 예측 결과에 대해 이미지 검증이 필요한지 판단한다.  
다음 조건 중 하나라도 만족하면 2차 multimodal verification을 수행한다.

- 1차 예측의 confidence가 낮은 경우
- `NeedImage == yes`로 판단된 경우
- 출력 형식 파싱에 실패한 경우

confidence는 Top-1 logit과 Top-2 logit의 차이인 logit margin으로 정의한다.

$$
margin = z_{top1} - z_{top2}
$$

이 값이 threshold보다 작으면 모델의 예측이 충분히 확실하지 않다고 보고, 이미지 기반 검증을 수행한다.

### 4. Multimodal Verification

2차 검증 단계에서는 이미지와 텍스트 정보를 함께 사용한다.  
입력은 `Image`, `Context`, `Question`, `Choices`, 그리고 1차 예측 결과인 `FirstStageOutput`으로 구성된다.

이 단계의 핵심 원칙은 **1차 예측을 기본값으로 유지하되, 명확한 근거가 있을 때만 답을 변경하는 것**이다.  
따라서 이미지가 제공되더라도 단순한 외형, 표정, 분위기, 성별, 인종, 사회적 고정관념에 기반한 판단은 정답 변경의 근거로 사용하지 않는다.

정답을 변경할 수 있는 근거는 다음과 객관적 정보로 제한되며 반대로 표정, 자세, 외모, 성별, 인종, 나이 추정, 성격 추정, 능력 추정, 위험성 추정 등은 유효한 근거로 사용하지 않는다.

### 5. Label-Prior Debiasing

각 단계에서 얻은 output logit은 label-prior measurement 단계에서 측정한 prior logit을 이용해 보정한다.  
모델이 특정 label 번호를 구조적으로 선호하는 현상을 줄이기 위해, 각 label의 logit에서 해당 label의 prior logit을 차감한다.

$$
\tilde{z}_i(x) = z_i(x) - \alpha z^{prior}_i
$$

여기서,

- $z_i(x)$: 입력 $x$에 대한 label $i$의 원래 output logit
- $z^{prior}_i$: neutral prompt에서 측정한 label $i$의 prior logit
- $\alpha$: prior 제거 강도
- $\tilde{z}_i(x)$: label-prior가 보정된 logit

보정된 logit은 softmax를 통해 확률로 변환한다.

$$
P(y=i|x) = \frac{\exp(\tilde{z}_i(x))}{\sum_j \exp(\tilde{z}_j(x))}
$$

최종 예측은 보정된 확률이 가장 높은 label로 결정한다.

$$
\hat{y} = \arg\max_i P(y=i|x)
$$

발표자료의 hyperparameter analysis에서는 1차 예측과 2차 검증 단계 모두에서 $\alpha = 0.25$를 사용할 때 가장 우수한 성능을 보였다.

### 6. Final Prediction

최종적으로 파이프라인은 다음 기준에 따라 답을 결정한다.

1. 먼저 context-only judge가 텍스트 근거만으로 1차 답변을 생성한다.
2. 이미지가 필요하지 않고 confidence가 충분하면 1차 답변을 유지한다.
3. confidence가 낮거나 이미지 검증이 필요하거나 파싱에 실패한 경우에만 multimodal verifier를 실행한다.
4. verifier는 명확한 객관적 근거가 있을 때만 1차 답변을 수정한다.
5. 각 단계의 logits는 label-prior debiasing을 통해 보정된다.
6. 보정된 logit 기반 확률이 가장 높은 label을 최종 prediction으로 선택한다.

이 방식은 모든 샘플에 이미지를 무조건 사용하는 방식이 아니라, 이미지가 필요한 경우에만 선택적으로 검증을 수행한다.  
따라서 불필요한 이미지 편향 유입을 줄이고, 어려운 샘플에서 섣불리 답을 변경하는 문제를 완화할 수 있다.

최종적으로 confidence 기반 선택적 검증, label-prior debiasing, 근거 기반 multimodal verifier가 모두 결합되었을 때 가장 높은 성능을 달성하였다.
