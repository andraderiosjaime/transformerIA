from math import isfinite, sin
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import torch

from btc_forecast.data.dataset import construir_muestras
from btc_forecast.data.splits import corte_con_validacion_interna
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import EVENTOS
from btc_forecast.features.transforms import aplicar_escalado
from btc_forecast.models.forecast import pronosticar
from btc_forecast.training.losses import nll_student_t
from btc_forecast.training.trainer import cargar_checkpoint, entrenar, guardar_checkpoint


def velas_sinteticas():
    inicio = pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000
    filas = []
    apertura = 100.0
    for i in range(320):
        cierre = 100 + 0.03 * i + 2 * sin(i / 3)
        instante = inicio + i * 3_600_000
        filas.append((instante, instante + 3_599_999, apertura, max(apertura, cierre) + 1,
                      min(apertura, cierre) - 1, cierre, 100 + i % 11))
        apertura = cierre
    return pd.DataFrame(filas, columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])


def entradas(muestras, escalado):
    return torch.stack([
        torch.from_numpy(aplicar_escalado(m.X, escalado).to_numpy(dtype="float32", copy=True))
        for m in muestras
    ])


def test_pipeline_cpu():
    velas = velas_sinteticas()
    fechas = pd.to_datetime(velas.close_time, unit="ms", utc=True)
    corte = fechas.iloc[315]
    ajuste, interna, evaluacion = corte_con_validacion_interna(construir_muestras(velas), corte)
    assert (len(ajuste), len(interna), len(evaluacion)) == (16, 4, 4)
    assert max(m.fecha_objetivo for m in ajuste) <= interna[0].fecha_prediccion
    assert max(m.fecha_objetivo for m in interna) <= corte == evaluacion[0].fecha_prediccion

    semilla = 42
    modelo, escalado, historial, mejor = entrenar(
        ajuste, interna, epocas=5, paciencia=2, tamano_lote=8, semilla=semilla, dispositivo="cpu"
    )
    filas = pd.concat([m.X for m in ajuste])
    unicas = filas.loc[~filas.index.duplicated()]
    pd.testing.assert_series_equal(escalado.centro, unicas.mean(), check_names=False)
    pd.testing.assert_series_equal(escalado.escala, unicas.std(ddof=0), check_names=False)
    assert len(historial) >= 2 and all(isfinite(v) for _, a, b in historial for v in (a, b))
    epoca_minima, _, nll_minima = min(historial, key=lambda fila: fila[2])
    assert mejor == epoca_minima

    muestra = evaluacion[0]
    rsi, macd = RSIIncremental(), MACDIncremental()
    for cierre in velas.loc[fechas <= muestra.fecha_prediccion, "close"]:
        rsi.actualizar(cierre)
        macd.actualizar(cierre)

    def pronostico(red, parametros):
        red.eval()
        with torch.inference_mode():
            salida = red(entradas([muestra], parametros))
        valores = tuple(v.item() for v in salida)
        eventos, cuantiles = pronosticar(*valores, muestra, rsi, macd)
        return valores, eventos, cuantiles

    antes, eventos, cuantiles = pronostico(modelo, escalado)
    assert set(eventos.index) == set(EVENTOS) and eventos.index.is_unique
    assert eventos.probabilidad.between(0, 1).all()
    assert eventos.loc[~eventos.elegible, "probabilidad"].eq(0).all()
    assert cuantiles.index.tolist() == [0.1, 0.5, 0.9]
    assert set(cuantiles.columns) == {"cierre", "RSI", "MACD"}
    assert all(isfinite(v) for v in cuantiles.to_numpy().flat)
    assert cuantiles.diff().iloc[1:].ge(0).all().all()

    with TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="pipeline-") as temporal:
        ruta = Path(temporal) / "mejor.pt"
        guardar_checkpoint(ruta, modelo, escalado, ventana=len(muestra.X), semilla=semilla,
                           cortes={"inicio_validacion_interna": interna[0].fecha_prediccion,
                                   "inicio_evaluacion": corte}, epoca=mejor)
        cargado, parametros, meta = cargar_checkpoint(ruta)
    assert meta["epoca"] == mejor and meta["ventana"] == 96
    assert next(cargado.parameters()).device.type == "cpu" and not cargado.training
    pd.testing.assert_frame_equal(escalado, parametros, check_exact=True)
    with torch.inference_mode():
        y = torch.tensor([m.y for m in interna], dtype=torch.float32)
        nll = nll_student_t(y, *cargado(entradas(interna, parametros)))
    assert nll.item() == pytest.approx(nll_minima, abs=1e-6)
    despues, eventos_cpu, cuantiles_cpu = pronostico(cargado, parametros)
    assert antes == despues
    pd.testing.assert_frame_equal(eventos, eventos_cpu, check_exact=True)
    pd.testing.assert_frame_equal(cuantiles, cuantiles_cpu, check_exact=True)
