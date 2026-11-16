"""Pruebas para pytest (panel Testing de VS Code):  pytest -v"""
import pytest

from tests import bateria, conversaciones

# Las evasiones (paráfrasis, ofuscación, otros idiomas) se miden en data/bateria_prompts.json.


@pytest.mark.parametrize("nombre", list(conversaciones.CONVERSACIONES))
def test_conversacion(nombre):
    resultado = next(c for c in conversaciones.ejecutar() if c["conversacion"] == nombre)
    fallos = [t for t in resultado["turnos"] if not t["ok"]]
    assert not fallos, fallos


def test_bateria_directos_y_benignos():
    resumen = bateria.resumen(bateria.ejecutar())
    assert resumen["directo"]["tasa"] >= 0.9
    # El híbrido (umbral 0.30) cambia recall por falsos positivos: 8/10 benignos (las reglas solas: 9/10).
    assert resumen["benigno"]["tasa"] >= 0.8
