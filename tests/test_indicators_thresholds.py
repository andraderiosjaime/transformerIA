"""Contrasta umbrales en t con etiquetas obtenidas al incorporar un cierre en t+1."""

import pandas as pd
import pytest

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import etiquetar_velas
from btc_forecast.features.thresholds import precio_critico_macd, precio_critico_rsi


def observaciones(historia, candidato):
    cierres = [*historia, candidato]
    tabla = pd.DataFrame({
        "open_time": [i * 3_600_000 for i in range(len(cierres))],
        "close": cierres,
    })
    filas = etiquetar_velas(tabla, calentamiento=0)
    return filas.iloc[-2], filas.iloc[-1]


@pytest.mark.parametrize("ultimo", [9, 10, 11], ids=["hist_negativo", "hist_cero", "hist_positivo"])
@pytest.mark.parametrize("desvio", [-0.25, 0.25], ids=["debajo", "encima"])
def test_cruces_macd(ultimo, desvio):
    historia = [10] * 34 + [ultimo]
    estado = MACDIncremental()
    for cierre in historia:
        valores = estado.actualizar(cierre)
    h = valores[-1]
    umbral = precio_critico_macd(estado)
    candidato = umbral + desvio
    t, t1 = observaciones(historia, candidato)

    assert (t1.histograma > 0) == (candidato > umbral)
    assert (t1.histograma < 0) == (candidato < umbral)
    assert t.macd_alcista == (h <= 0 and candidato > umbral) == (ultimo <= 10 and desvio > 0)
    assert t.macd_bajista == (h >= 0 and candidato < umbral) == (ultimo >= 10 and desvio < 0)


def test_igualdad_macd():
    historia = [10] * 34
    estado = MACDIncremental()
    for cierre in historia:
        estado.actualizar(cierre)
    umbral = precio_critico_macd(estado)
    assert umbral == 10
    t, t1 = observaciones(historia, umbral)
    assert t.histograma == t1.histograma == 0
    assert t.macd_alcista == t.macd_bajista == 0


def historia_rsi(subida, bajada, base=1000):
    # 15 cierres: una subida, una bajada y 12 cambios nulos.
    return [base, base + subida] + [base + subida - bajada] * 13


def comprobar_rsi(historia, candidato, nivel, esperado):
    estado = RSIIncremental()
    for cierre in historia:
        rsi = estado.actualizar(cierre)
    umbral = precio_critico_rsi(estado, nivel)
    t, t1 = observaciones(historia, candidato)
    if nivel == 70:
        por_precio = rsi <= 70 and candidato > umbral
        observada = t.rsi_sobrecompra
    else:
        por_precio = rsi >= 30 and candidato < umbral
        observada = t.rsi_sobreventa
    assert por_precio == esperado
    assert observada == por_precio, (t.rsi, t1.rsi, candidato, umbral)
    return t.rsi, t1.rsi, umbral


@pytest.mark.parametrize("nivel,umbral", [(70, 1364), (30, 636)])
@pytest.mark.parametrize("desvio", [-1, 0, 1])
def test_entrada_rsi(nivel, umbral, desvio):
    # G = D = 21; cada umbral dista 364 del cierre.
    candidato = umbral + desvio
    esperado = desvio > 0 if nivel == 70 else desvio < 0
    _, siguiente, calculado = comprobar_rsi(historia_rsi(294, 294), candidato, nivel, esperado)
    assert calculado == umbral
    if desvio == 0:
        assert siguiente == nivel


@pytest.mark.parametrize("nivel,subida,bajada,paso", [(70, 98, 42, 1), (30, 42, 98, -1)])
def test_rsi_elegible_desde_el_nivel(nivel, subida, bajada, paso):
    historia = historia_rsi(subida, bajada)
    actual, _, _ = comprobar_rsi(historia, historia[-1] + paso, nivel, True)
    assert actual == nivel


@pytest.mark.parametrize("nivel,subida,bajada,paso,final", [
    (70, 78, 22, -1, 76),
    (30, 22, 78, 1, 24),
])
def test_permanecer_en_la_zona_es_inelegible(nivel, subida, bajada, paso, final):
    historia = historia_rsi(subida, bajada)
    # Con Wilder (13/14), el RSI pasa de 78 a 76 o de 22 a 24.
    candidato = historia[-1] + paso * (13 / 14) * 100 * (78 / 76 - 1)
    actual, siguiente, umbral = comprobar_rsi(historia, candidato, nivel, False)
    assert actual == pytest.approx(subida)
    assert siguiente == pytest.approx(final)
    assert (candidato > umbral) if nivel == 70 else (candidato < umbral)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="En la igualdad matemática Wilder redondea el RSI a 70.00000000000001.")
def test_redondeo_en_el_umbral_de_sobrecompra():
    # Con cierre 252, el RSI teórico es exactamente 70.
    comprobar_rsi(historia_rsi(42, 42, base=200), 252, 70, False)
