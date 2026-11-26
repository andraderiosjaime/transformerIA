from math import log, sin
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import EVENTOS
from btc_forecast.features.thresholds import precio_critico_rsi
from btc_forecast.features.transforms import VARIABLES
from btc_forecast.models import baselines
from scripts.run_walkforward import evaluar_corte


def estado_de_prueba(cierres):
    rsi, macd = RSIIncremental(), MACDIncremental()
    for cierre in cierres:
        nivel = rsi.actualizar(cierre)
        valores = macd.actualizar(cierre)
    x = pd.DataFrame([{"rsi_100": nivel / 100, "histograma_cierre": valores[-1] / cierre,
                       "volatilidad_ewma": 0.125}])
    muestra = SimpleNamespace(X=x, cierre_t=cierre)
    return muestra, rsi, macd, baselines.Niveles(nivel, valores[2])


def test_rsi_persistencia_y_cierre_constante():
    _, rsi, macd, actual = estado_de_prueba([10, 11] * 20)
    assert 0 < actual.rsi < 100 and actual.rsi != 50
    assert baselines.persistencia(actual).rsi == actual.rsi
    assert baselines.cierre_constante(rsi, macd).rsi == pytest.approx(actual.rsi, abs=1e-12)


def test_macd_alcista_inelegible_tiene_probabilidad_cero():
    muestra, rsi, macd, _ = estado_de_prueba([10] * 34 + [11])
    assert muestra.X.histograma_cierre.iloc[-1] > 0
    frecuencias = {e: baselines.Frecuencia(positivos=1, conocidos=2, elegibles=2) for e in EVENTOS}
    historica = baselines.pronosticar_eventos(muestra, frecuencias)
    gaussiana = baselines.ewma_gaussiana(muestra, rsi, macd)
    empirica = baselines.ewma_empirica(muestra, rsi, macd, (-8, 8))
    assert historica["macd_alcista"] == 0
    assert gaussiana.at["macd_alcista", "Probabilidad"] == 0
    assert empirica.at["macd_alcista", "Empírica"] == 0


@pytest.mark.parametrize("evento,nivel", [("rsi_sobrecompra", 70), ("rsi_sobreventa", 30)])
def test_empirica_excluye_igualdad(evento, nivel):
    muestra, rsi, macd, actual = estado_de_prueba([10, 11] * 20)
    assert 30 < actual.rsi < 70
    umbral = log(precio_critico_rsi(rsi, nivel) / muestra.cierre_t)
    sigma = muestra.X.volatilidad_ewma.iloc[-1]
    z = umbral / sigma
    assert sigma * z == umbral  # Sigma es una potencia de dos.
    # Dos empates y colas de distinto tamaño.
    residuos = (z - 1, z, z, z + 1, z + 2)
    fila = baselines.ewma_empirica(muestra, rsi, macd, residuos).loc[evento]
    favorables = 2 if nivel == 70 else 1
    assert fila["Favorables"] == favorables
    assert fila["Escenarios"] == 5
    assert fila["Empírica"] == favorables / 5


def test_cambiar_el_futuro_conserva_los_pronosticos_en_t():
    horas = pd.RangeIndex(336)
    inicio = pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000
    precios = pd.Series([100 + 3 * sin(i / 2) for i in horas])
    tabla = pd.DataFrame({"open_time": inicio + horas * 3_600_000,
                          "close_time": inicio + (horas + 1) * 3_600_000 - 1,
                          "close": precios, "open": precios, "high": precios + 1,
                          "low": precios - 1, "volume": 100 + horas % 7})
    corte = pd.to_datetime(tabla.at[320, "close_time"], unit="ms", utc=True)
    t = pd.to_datetime(tabla.at[327, "close_time"], unit="ms", utc=True)
    cambiada = tabla.copy()
    cambiada.loc[328:, ["open", "high", "low", "close"]] *= 1.25

    def en_t(velas):
        filas, n_entrenamiento, _ = evaluar_corte(velas, corte)
        assert n_entrenamiento > 0
        return filas.loc[filas.fecha_prediccion == t].set_index(["modelo", "objetivo"]).sort_index()

    antes, despues = en_t(tabla), en_t(cambiada)
    assert len(antes) == 16 and antes.prediccion.notna().all()
    assert antes.at[("persistencia", "macd"), "observado"] != despues.at[("persistencia", "macd"), "observado"]
    pd.testing.assert_series_equal(antes.prediccion, despues.prediccion, check_exact=True)


def test_student_t_lineal_usa_solo_la_ultima_fila():
    torch.manual_seed(42)
    modelo = baselines.StudentTLineal(len(VARIABLES)).eval()
    primera = torch.randn(1, 96, len(VARIABLES))
    segunda = primera.clone()
    segunda[:, :95, :] += 1.0
    assert (primera[:, :95, :] != segunda[:, :95, :]).all()
    assert torch.equal(primera[:, -1, :], segunda[:, -1, :])

    with torch.inference_mode():
        antes = modelo(primera)
        despues = modelo(segunda)
    for a, b in zip(antes, despues, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("mu_cero", [False, True], ids=["mu_aprendida", "mu_cero"])
def test_student_t_lineal_parametros_del_lote(mu_cero):
    torch.manual_seed(42)
    modelo = baselines.StudentTLineal(len(VARIABLES)).eval()
    modelo.cabeza.mu_cero = mu_cero
    ventanas = torch.randn(2, 96, len(VARIABLES))
    with torch.no_grad():
        # Prueba las restricciones con salidas crudas negativas.
        modelo.cabeza.proyeccion.bias[1].fill_(-5.0)
        modelo.cabeza.raw_nu.fill_(-5.0)
        mu, s, nu = modelo(ventanas)
    assert mu.shape == s.shape == (2,)
    assert nu.ndim == 0
    assert torch.isfinite(s).all() and (s > 0).all()
    assert torch.isfinite(nu) and nu > 2
    if mu_cero:
        assert torch.equal(mu, torch.zeros_like(mu))
    else:
        assert torch.count_nonzero(mu) > 0
