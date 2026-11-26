from math import log, sin
from typing import NamedTuple

import pandas as pd
import pytest
import torch

from btc_forecast.data.dataset import construir_muestras
from btc_forecast.data.splits import corte_cronologico, folds_mensuales_expansivos
from btc_forecast.models.decoder import Decoder


@pytest.fixture
def velas():
    """320 horas: cubren las 200 de calentamiento y una ventana de 96."""
    inicio_ms = pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000
    hora_ms = 3_600_000
    filas = []
    apertura = 100.0
    for i in range(320):
        instante = inicio_ms + i * hora_ms
        cierre = 100 + i / 20 + 2 * sin(i / 3)
        filas.append((instante, instante + hora_ms - 1, apertura, cierre,
                      max(apertura, cierre) + 2, min(apertura, cierre) - 2, 50 + i % 7))
        apertura = cierre
    return pd.DataFrame(filas, columns=["open_time", "close_time", "open", "close",
                                        "high", "low", "volume"])


def test_cambiar_cierre_siguiente_conserva_X_y_cambia_y(velas):
    i = 305
    t = pd.to_datetime(velas.at[i, "close_time"], unit="ms", utc=True)
    original = next(m for m in construir_muestras(velas) if m.fecha_prediccion == t)

    copia = velas.copy()
    # Usar el máximo conserva la coherencia OHLC.
    nuevo_cierre = copia.at[i + 1, "high"]
    copia.at[i + 1, "close"] = nuevo_cierre
    modificada = next(m for m in construir_muestras(copia) if m.fecha_prediccion == t)

    assert original.X.shape == (96, 8)
    assert original.X.index[-1] == t
    pd.testing.assert_frame_equal(original.X, modificada.X, check_exact=True)
    cierre_t = velas.at[i, "close"]
    assert original.y == pytest.approx(log(velas.at[i + 1, "close"] / cierre_t))
    assert modificada.y == pytest.approx(log(nuevo_cierre / cierre_t))
    assert modificada.y != original.y


@pytest.mark.parametrize("minutos", [0, 30], ids=["en_el_cierre", "entre_cierres"])
def test_entrenamiento_solo_con_etiquetas_conocidas(velas, minutos):
    muestras = list(construir_muestras(velas))
    cierre = pd.to_datetime(velas.at[305, "close_time"], unit="ms", utc=True)
    corte = cierre + pd.Timedelta(minutes=minutos)
    entrenamiento, validacion = corte_cronologico(muestras, corte)

    assert entrenamiento and validacion
    assert all(m.fecha_objetivo <= corte for m in entrenamiento)
    assert entrenamiento[-1].fecha_objetivo == cierre
    primera_validacion = cierre if minutos == 0 else cierre + pd.Timedelta(hours=1)
    assert validacion[0].fecha_prediccion == primera_validacion


def test_decoder_no_usa_posiciones_posteriores_a_t(velas):
    muestra = next(construir_muestras(velas))
    ventana = torch.from_numpy(muestra.X.to_numpy(dtype="float32", copy=True)).unsqueeze(0)
    t = ventana.size(1) // 2 - 1
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(27)
        modelo = Decoder(numero_variables=ventana.size(-1)).eval()
        alterada = ventana.clone()
        futuro = alterada[:, t + 1:]
        futuro.add_(torch.randn_like(futuro))
        original = modelo(ventana)
        perturbada = modelo(alterada)

    assert original.shape == perturbada.shape == (1, 96, 64)
    torch.testing.assert_close(original[:, :t + 1], perturbada[:, :t + 1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(original[:, t + 1:], perturbada[:, t + 1:], rtol=1e-5, atol=1e-6)


class MuestraTemporal(NamedTuple):
    fecha_prediccion: pd.Timestamp
    fecha_objetivo: pd.Timestamp


@pytest.fixture(scope="module")
def muestras_horarias():
    # Octubre aporta el cierre que falta a la última etiqueta de septiembre.
    fechas = pd.date_range("2025-01-01 00:59:59.999", "2025-10-01 02:59:59.999", freq="h", tz="UTC")
    return [MuestraTemporal(t, siguiente) for t, siguiente in zip(fechas[:-1], fechas[1:])]


def test_folds_expansivos_sin_solapar_evaluaciones(muestras_horarias):
    folds = list(folds_mensuales_expansivos(muestras_horarias, "2025-01-01"))
    assert [corte.month for corte, *_ in folds] == [4, 5, 6, 7, 8, 9]
    historia_anterior = set()
    puntuadas = set()
    for corte, ajuste, interna, evaluacion in folds:
        fin_mes = corte + pd.offsets.MonthBegin()
        historia = {m.fecha_prediccion for m in ajuste + interna}
        fechas = {m.fecha_prediccion for m in evaluacion}
        assert historia_anterior < historia
        assert puntuadas <= historia
        assert fechas and fechas.isdisjoint(puntuadas)
        assert all(corte <= m.fecha_prediccion < m.fecha_objetivo < fin_mes for m in evaluacion)
        assert max(m.fecha_objetivo for m in ajuste) <= min(m.fecha_prediccion for m in interna)
        assert max(m.fecha_objetivo for m in interna) <= corte
        historia_anterior = historia
        puntuadas.update(fechas)


def test_purga_fin_de_etiqueta_en_frontera_interna(muestras_horarias):
    _, ajuste, interna, _ = next(folds_mensuales_expansivos(muestras_horarias, "2025-01-01"))
    ultima = ajuste[-1]
    frontera = interna[0].fecha_prediccion
    assert ultima.fecha_objetivo == frontera
    # Mismo origen, etiqueta disponible una hora después.
    retrasada = MuestraTemporal(ultima.fecha_prediccion, frontera + pd.Timedelta(hours=1))
    modificadas = [retrasada if m == ultima else m for m in muestras_horarias]
    _, ajuste, interna, evaluacion = next(folds_mensuales_expansivos(modificadas, "2025-01-01"))
    assert interna[0].fecha_prediccion == frontera
    assert ultima.fecha_prediccion not in {m.fecha_prediccion for m in ajuste + interna + evaluacion}


def test_septiembre_excluye_el_objetivo_que_necesita_octubre(muestras_horarias):
    reserva = pd.Timestamp("2025-10-01", tz="UTC")
    aceptada = pd.Timestamp("2025-09-30 22:59:59.999", tz="UTC")
    descartada = aceptada + pd.Timedelta(hours=1)
    objetivos = dict(muestras_horarias)
    assert objetivos[aceptada] == descartada
    assert objetivos[descartada] == pd.Timestamp("2025-10-01 00:59:59.999", tz="UTC")

    folds = list(folds_mensuales_expansivos(muestras_horarias, "2025-01-01", inicio_reserva=reserva))
    assert folds[-1][3][-1].fecha_prediccion == aceptada
    for _, ajuste, interna, evaluacion in folds:
        for tramo in (ajuste, interna, evaluacion):
            assert all(m.fecha_prediccion != descartada and m.fecha_objetivo < reserva for m in tramo)
