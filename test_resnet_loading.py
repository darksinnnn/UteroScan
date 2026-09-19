import sys
from pathlib import Path
import torch

SYS_PATH = Path("DCGAN&MobileNetV2/DCGAN&MobileNetV2")
sys.path.insert(0, str(SYS_PATH))

from nets import get_model_from_name
import torchvision.models as models

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check model registry
print(f"Registered models in nets.__init__: {list(get_model_from_name.keys())}")
assert "resnet50" in get_model_from_name, "resnet50 missing from get_model_from_name!"

# Instantiate ResNet50 from nets
model = get_model_from_name["resnet50"](num_classes=2, pretrained=False)
print(f"Successfully instantiated {model.__class__.__name__} from nets/resnet.py!")
print(f"Head output dimension: {model.fc.out_features} (classes: UF=0, NUF=1)")

# Load ImageNet pretrained backbone weights from torchvision cache
tv_res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
tv_state = tv_res.state_dict()
pretrained_backbone = {k: v for k, v in tv_state.items() if not k.startswith("fc.")}
missing_keys, unexpected_keys = model.load_state_dict(pretrained_backbone, strict=False)

print(f"Pretrained weights loaded into nets/resnet.py ResNet50:")
print(f"  Missing keys (head): {missing_keys}")
print(f"  Unexpected keys    : {unexpected_keys}")

# Test freeze_backbone
model.freeze_backbone()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"After freeze_backbone():")
print(f"  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,} (head only)")
assert trainable_params == 2048 * 2 + 2, f"Expected 4098 trainable params in head, got {trainable_params}"

# Test forward pass with 224x224 input
dummy = torch.randn(2, 3, 224, 224)
out = model(dummy)
print(f"Forward pass output shape: {out.shape} (Expected: [2, 2])")
assert out.shape == (2, 2)
print(">>> ResNet50 verification successful! <<<")
