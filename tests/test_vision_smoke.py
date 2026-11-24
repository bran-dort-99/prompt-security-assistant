import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from src.vision import entrenar as E
from src.vision import generar_imagenes as G
from src.vision import modelo as M


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    G.generar(d, None, {"train": 10, "val": 4, "test_sintetico": 4}, dataset=None)
    return d


def test_generador_crea_3_clases_y_tamano(datos):
    for split in G.SPLITS:
        for c in G.CLASES:
            archivos = list((datos / split / c).glob("*.jpg"))
            assert archivos and min(Image.open(archivos[0]).size) == G.LADO


def test_test_sintetico_usa_fuentes_no_vistas(datos):
    import csv

    filas = list(csv.DictReader((datos / "manifest.csv").open(encoding="utf-8")))
    f = lambda s: {r["fuente_tipografica"] for r in filas if r["split"] == s and r["fuente_tipografica"]}  # noqa: E731
    assert not f("train") & f("test_sintetico")


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50"])
def test_forma_de_salida(backbone):
    m = M.construir_modelo(backbone, 3, weights=None).eval()
    assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 3)


def test_entrena_una_epoca_y_predice(datos, tmp_path):
    m, ds, hist = E.entrenar(datos, ep1=1, ep2=0, batch=8, workers=0, weights=None, salida=tmp_path / "m.pt")
    assert hist and hist[0]["etapa"] == 1
    c, p, probs = M.predecir(next((datos / "val" / "codigo_qr").glob("*.jpg")), *M.cargar(tmp_path / "m.pt"))
    assert c in M.CLASES and abs(sum(probs.values()) - 1) < 1e-5
