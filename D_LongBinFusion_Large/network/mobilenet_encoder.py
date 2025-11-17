import torch.nn as nn
import torchvision.models as models
import torch

class MobileNetV2EncoderCore(models.MobileNetV2):
    """표준 3채널 MobileNetV2 구조 (특징 추출기 전용)"""
    def __init__(self):
        # num_classes=1000은 분류기 관련 부분이며, 여기서는 무시
        super(MobileNetV2EncoderCore, self).__init__(num_classes=1000)
        
        # 최종 분류기 관련 부분 제거
        self.classifier = nn.Sequential() # 빈 시퀀스로 대체
        
    def forward(self, x):
        return self.features(x)

def mobilenet_single_image_input(pretrained=False):
    model = MobileNetV2EncoderCore()
    
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        
        try:
            loaded = weights.get_state_dict(progress=True)
        except AttributeError:
            loaded = torch.hub.load_state_dict_from_url(weights.url, progress=True)

        # 'classifier' 가중치가 누락되거나 크기가 다를 수 있으므로 strict=False 사용
        model.load_state_dict(loaded, strict=False)
        
    return model

class MobileNetEncoder(nn.Module):
    """
    MobileNetV2 기반의 인코더 (N개 이미지를 배치 차원에서 처리)
    """
    def __init__(self, pretrained, num_input_images=1):
        super(MobileNetEncoder, self).__init__()
        
        # ⚠️ 이제 num_input_images는 forward 함수에서 입력 텐서를 재구성하는 데만 사용
        self.num_input_images = num_input_images
        
        # MobileNetV2의 주요 특징 채널: [32, 24, 32, 96, 320]
        self.num_ch_enc = torch.tensor([16, 24, 32, 96, 320])
        
        # 3채널 인코더 로드
        self.encoder = mobilenet_single_image_input(pretrained)
        
        # MobileNetV2의 features 시퀀스를 특징 추출을 위해 분리합니다.
        # MobileNetV2 구조: features[0] (Conv1) -> features[1..17] (Inverted Residual Blocks)
        self.stages = nn.ModuleList([
            nn.Sequential(self.encoder.features[:2]),    # Stage 0: Conv1 + Inverted Residual Block 1
            nn.Sequential(self.encoder.features[2:4]),   # Stage 1: Blocks 2 + 3
            nn.Sequential(self.encoder.features[4:7]),   # Stage 2: Blocks 4, 5, 6
            nn.Sequential(self.encoder.features[7:14]),  # Stage 3: Blocks 7 ... 13
            nn.Sequential(self.encoder.features[14:18])    # Stage 4: Blocks 14 ... 18
        ])

    def forward(self, input_image):
        # input_image의 shape은 (B * N, 3, H, W)여야 합니다.
        self.features = []
        
        # 1. 입력 정규화 (ResNetEncoder와 동일)
        x = (input_image - 0.45) / 0.225
        
        # 2. MobileNetV2의 단계별 특징 추출
        # MobileNetV2는 MaxPool이 없으므로 features[0]은 1/2 해상도만 가집니다.
        
        # Stage 0 (1/4 해상도): features[0] + features[1] (첫 번째 Residual Block)
        x = self.stages[0](x) # x.shape -> (B*N, 24, H/4, W/4)
        self.features.append(x) 

        # Stage 1 (1/8 해상도): features[2] + features[3]
        x = self.stages[1](x) # x.shape -> (B*N, 32, H/8, W/8)
        self.features.append(x)
        
        # Stage 2 (1/16 해상도): features[4] + features[5] + features[6]
        x = self.stages[2](x) # x.shape -> (B*N, 96, H/16, W/16)
        self.features.append(x)
        
        # Stage 3 (1/32 해상도): features[7] ... features[13]
        #x = self.stages[3](x) # x.shape -> (B*N, 160, H/32, W/32)
        #self.features.append(x)

        # Stage 4 (1/32 해상도): features[14] ... features[17]
        #x = self.stages[4](x) # x.shape -> (B*N, 320, H/32, W/32)
        #self.features.append(x) 

        return self.features