# DINOv3 활용 가이드: Classification & Open Vocabulary OD

DINOv3로 학습된 강력한 Backbone을 가지고 실제 어플리케이션(Classification, Object Detection)에 어떻게 적용할 수 있는지 가이드해 줄게.

---

## 1. Classification (이미지 분류)

가장 기본적인 활용법이야. DINOv3는 이미지를 아주 훌륭한 벡터(Feature)로 변환해주니까, 그 위에 얇은 분류기(Linear Layer) 하나만 얹으면 돼.

### 방법 A: Linear Probing (추천)
Backbone은 고정(Freeze)하고, 마지막 분류기만 학습하는 방식이야. DINOv3 성능이 워낙 좋아서 이것만으로도 충분할 때가 많아.

**구현 단계:**
1.  **모델 로드**: `DinoVisionTransformer`를 불러와서 학습된 가중치를 로드해.
2.  **Feature 추출**:
    *   이미지를 모델에 넣고 `forward_features`를 실행해.
    *   반환된 딕셔너리에서 `x_norm_clstoken` (CLS 토큰)을 가져와. 이게 이미지 전체를 요약한 벡터야.
3.  **분류기 통과**:
    *   `nn.Linear(embed_dim, num_classes)` 레이어를 하나 만들어.
    *   CLS 토큰을 이 레이어에 통과시켜서 Logit을 얻어.

**코드 예시 (개념적):**
```python
import torch
from dinov3.models import vision_transformer

# 1. 모델 준비
backbone = vision_transformer.vit_large()
# ... 체크포인트 로드 ...
backbone.eval() # 백본은 평가 모드로!

# 2. 분류기 준비
classifier = torch.nn.Linear(1024, 1000) # 예: ImageNet 1000개 클래스

# 3. 추론
img = load_image("cat.jpg")
with torch.no_grad():
    features = backbone.forward_features(img)
    cls_token = features["x_norm_clstoken"] # [Batch, Embed_Dim]

logits = classifier(cls_token)
prediction = torch.argmax(logits, dim=1)
```

> **참고**: `dinov3/eval/linear.py` 파일에 이 전체 과정이 아주 상세하게 구현되어 있어. 실제 학습을 돌릴 땐 그 파일을 참고하면 좋아.

---

## 2. Open Vocabulary Object Detection (열린 어휘 객체 탐지)

이건 좀 더 재밌는 주제야. "학습 때 본 적 없는 물체"도 텍스트 설명만으로 찾고 싶은 거지. DINOv3 자체는 탐지기(Detector)가 아니지만, **텍스트와 정렬(Alignment)**되는 기능을 활용하면 가능해.

DINOv3 코드의 `eval/text` 폴더를 보면 `DINOTxt`라는 모델이 있어. 이게 핵심이야.

### 핵심 아이디어: Image-Text Matching (CLIP 방식)
DINOv3가 만든 이미지 벡터와, 텍스트 모델이 만든 텍스트 벡터가 같은 공간(Space)에 있도록 학습시키는 거야.

**구현 단계:**

1.  **Region Proposal (물체 후보 찾기)**:
    *   DINOv3만으로는 "어디에 물체가 있는지" 정확한 박스를 치기 어려울 수 있어.
    *   보통은 **RPN(Region Proposal Network)**이나 **Selective Search** 같은 외부 알고리즘을 써서 "물체가 있을 법한 사각형 박스들"을 먼저 찾아.
    *   혹은 DINOv3의 Attention Map을 분석해서 높은 Attention이 모인 곳을 후보로 쓸 수도 있어.

2.  **Region Feature Extraction (후보 영역 특징 추출)**:
    *   찾아낸 박스 영역(Crop)을 잘라내서 DINOv3에 넣어.
    *   거기서 나온 `x_norm_clstoken`이 그 박스 영역의 특징 벡터가 돼.

3.  **Text Feature Extraction (텍스트 특징 추출)**:
    *   찾고 싶은 물체들의 이름을 텍스트로 준비해 (예: "a photo of a cat", "a photo of a dog").
    *   `dinov3/eval/text/text_tower.py`에 있는 텍스트 모델을 써서 이 문장들을 벡터로 변환해.

4.  **Similarity Calculation (유사도 계산)**:
    *   (이미지 박스 벡터) • (텍스트 벡터) 내적(Dot Product)을 구해.
    *   가장 유사도가 높은 텍스트가 그 박스의 정체가 되는 거야!

**코드 구조 (`dinov3/eval/text/dinotxt_model.py` 참고):**

```python
class DINOTxt(nn.Module):
    def get_logits(self, image, text):
        # 1. 텍스트 인코딩
        text_features = self.encode_text(text, normalize=True)
        
        # 2. 이미지 인코딩 (여기서는 전체 이미지지만, OD에서는 Crop된 이미지)
        image_features = self.encode_image(image, normalize=True)
        
        # 3. 유사도 계산 (Logit Scale은 온도 상수)
        # 행렬 곱: [Image_N, Dim] x [Text_M, Dim]^T = [Image_N, Text_M]
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        
        return image_logits
```

### 요약 가이드
*   **Classification**을 하려면? -> Backbone 위에 `Linear` 레이어 하나 얹어서 학습시키기. (`eval/linear.py` 참고)
*   **Open Vocabulary OD**를 하려면? -> `DINOTxt` 모델을 활용해서 이미지 패치(Patch/Crop)와 텍스트 프롬프트 간의 유사도를 비교하기. (`eval/text/` 폴더 참고)
