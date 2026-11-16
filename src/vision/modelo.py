"""Modelo de visión (D3): EfficientNet-B0 (principal) o ResNet50 con cabeza nueva de 3 clases (D1)."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

RAIZ = Path(__file__).resolve().parent.parent.parent
CLASES = ["codigo_qr", "foto_natural", "texto_incrustado"]  # orden alfabético = ImageFolder
RUTA_MODELO = RAIZ / "models" / "cnn_efficientnet_b0.pt"
MEDIA, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def construir_modelo(backbone: str = "efficientnet_b0", n_clases: int = 3, weights: str | None = "imagenet", dropout: float = 0.3) -> nn.Module:
    """weights='imagenet' descarga pesos preentrenados (Colab/local); weights=None para pruebas sin red."""
    usar = weights == "imagenet"
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if usar else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n_clases))
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if usar else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_clases))
    else:
        raise ValueError(backbone)
    return m


def cabeza(m: nn.Module) -> nn.Module:
    return m.classifier if hasattr(m, "classifier") else m.fc


def congelar_backbone(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    for p in cabeza(m).parameters():
        p.requires_grad = True


def descongelar_ultimos(m: nn.Module, n_bloques: int = 2) -> None:
    """Etapa 2: descongela los últimos `n_bloques` bloques del backbone (+ la cabeza)."""
    bloques = list(m.features.children()) if hasattr(m, "features") else [m.layer1, m.layer2, m.layer3, m.layer4]
    for b in bloques[-n_bloques:]:
        for p in b.parameters():
            p.requires_grad = True


def transform_train() -> transforms.Compose:
    # Sin flip horizontal: invierte el texto y los QR.
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.ColorJitter(0.2, 0.2, 0.1),
                               transforms.ToTensor(), transforms.Normalize(MEDIA, STD)])


def transform_eval() -> transforms.Compose:
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEDIA, STD)])


def guardar(m: nn.Module, ruta: Path = RUTA_MODELO, backbone: str = "efficientnet_b0", clases: list[str] | None = None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in m.state_dict().items()}, "backbone": backbone, "clases": clases or CLASES}, ruta)


_CACHE: dict[str, tuple[nn.Module, list[str]]] = {}


def cargar(ruta: Path = RUTA_MODELO, device: torch.device | None = None) -> tuple[nn.Module, list[str]]:
    clave = str(ruta)
    if clave not in _CACHE:
        ck = torch.load(ruta, map_location="cpu", weights_only=True)
        m = construir_modelo(ck["backbone"], len(ck["clases"]), weights=None)
        m.load_state_dict(ck["state_dict"])
        _CACHE[clave] = (m.eval().to(device or torch.device("cpu")), ck["clases"])
    return _CACHE[clave]


@torch.no_grad()
def predecir(imagen: str | Path | Image.Image, modelo: nn.Module | None = None, clases: list[str] | None = None) -> tuple[str, float, dict[str, float]]:
    """(clase, probabilidad, todas las probabilidades). Sin `modelo` carga models/cnn_efficientnet_b0.pt."""
    if modelo is None:
        modelo, clases = cargar()
    clases = clases or CLASES
    im = imagen if isinstance(imagen, Image.Image) else Image.open(imagen)
    x = transform_eval()(im.convert("RGB")).unsqueeze(0)
    dev = next(modelo.parameters()).device
    p = torch.softmax(modelo.eval()(x.to(dev)), dim=1)[0].cpu().tolist()
    probs = dict(zip(clases, p))
    mejor = max(probs, key=probs.get)
    return mejor, probs[mejor], probs
