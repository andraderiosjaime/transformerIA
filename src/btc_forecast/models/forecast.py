"""Pronósticos a una vela a partir de Student-t o de regresión directa de indicadores."""

import argparse
from copy import deepcopy
from math import exp, isclose, isfinite, log, sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

from btc_forecast.data.dataset import ObjetivosDirectos, construir_muestras
from btc_forecast.data.splits import corte_cronologico
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.models.baselines import Niveles, elegibilidad, ewma_gaussiana, umbrales_eventos


def pronosticar(mu, s, nu, muestra, rsi, macd):
    """Usa parámetros en unidades de retorno logarítmico y estados al cierre de t."""
    if not all(map(isfinite, (mu, s, nu))) or s <= 0 or nu <= 2:
        raise ValueError("Se requieren parámetros finitos, escala positiva y nu > 2.")
    cierre = muestra.cierre_t
    if cierre != rsi.ultimo_cierre:
        raise ValueError("El estado RSI y la muestra deben corresponder al mismo cierre.")
    distribucion = student_t(df=nu, loc=mu, scale=s)
    elegibles = elegibilidad(muestra)
    filas = []
    for evento, (critico, superior) in umbrales_eventos(rsi, macd).items():
        requerido = log(critico / cierre) if critico > 0 else None
        if not elegibles[evento]:
            probabilidad = 0.0
        elif critico <= 0:
            probabilidad = float(superior)
        elif superior:
            probabilidad = distribucion.sf(requerido)
        else:
            probabilidad = distribucion.cdf(requerido)
        filas.append((evento, elegibles[evento], critico, requerido, probabilidad))
    eventos = pd.DataFrame(filas, columns=["evento", "elegible", "cierre_critico",
                                          "retorno_critico", "probabilidad"]).set_index("evento")

    niveles = []
    for q in (0.1, 0.5, 0.9):
        retorno = distribucion.ppf(q)
        candidato = cierre * exp(retorno)
        # Cada cuantil parte del estado observado en t.
        nivel_rsi = deepcopy(rsi).actualizar(candidato)
        nivel_macd = deepcopy(macd).actualizar(candidato)[2]
        niveles.append((q, candidato, nivel_rsi, nivel_macd))
    cuantiles = pd.DataFrame(niveles, columns=["cuantil", "cierre", "RSI", "MACD"]).set_index("cuantil")
    return eventos, cuantiles


def reconstruir_regresion_directa(salida_estandarizada, escalado_objetivos, *, rsi_t, macd_t, cierre_t):
    """Recibe el par de la cabeza para un origen y horizonte. Devuelve deltas y niveles sin recorte."""
    if not all(map(isfinite, (rsi_t, macd_t, cierre_t))) or cierre_t <= 0:
        raise ValueError("Se requieren indicadores finitos y un cierre de origen positivo y finito.")
    cambios = []
    for nombre, z in zip(ObjetivosDirectos._fields, salida_estandarizada, strict=True):
        media = escalado_objetivos.at[nombre, "media"]
        escala = escalado_objetivos.at[nombre, "escala"]
        if not all(map(isfinite, (z, media, escala))) or escala <= 0:
            raise ValueError(f"Salida o escalado inválido para {nombre}.")
        cambios.append(media + escala * float(z))
    deltas = ObjetivosDirectos(*cambios)
    estimados = Niveles(rsi_t + deltas.delta_rsi, macd_t + cierre_t * deltas.delta_macd_sobre_cierre)
    return deltas, estimados


def _estados_al_cierre(velas, fecha):
    fechas = pd.to_datetime(velas["close_time"], unit="ms", utc=True)
    historia = velas.set_index(fechas)["close"].sort_index().loc[:fecha]
    anterior = None
    for instante, precio in historia.items():
        if anterior is None or instante - anterior != pd.Timedelta(hours=1):
            rsi, macd = RSIIncremental(), MACDIncremental()
        nivel_rsi = rsi.actualizar(float(precio))
        valores_macd = macd.actualizar(float(precio))
        anterior = instante
    if anterior != fecha or nivel_rsi is None or valores_macd is None:
        raise ValueError("Faltan indicadores inicializados en la fecha de predicción.")
    return rsi, macd, Niveles(nivel_rsi, valores_macd[2])


def demostrar_regresion_directa(velas):
    from btc_forecast.training.trainer import _pronosticar_ventana, cargar_checkpoint

    ruta = Path("artifacts/mejor_regresion_directa_20250120_semilla43.pt")
    red, escalado_x, meta = cargar_checkpoint(ruta)
    muestras = construir_muestras(velas, longitud=meta["ventana"])
    _, evaluacion = corte_cronologico(muestras, meta["cortes"]["inicio_evaluacion"])
    if not evaluacion:
        raise ValueError("Faltan ventanas de evaluación con objetivo conocido.")
    m = evaluacion[0]
    rsi, macd, actual = _estados_al_cierre(velas, m.fecha_prediccion)
    z = _pronosticar_ventana(red, m.X, escalado_x)[0, 0].tolist()
    escalado_y = red.escalado_objetivos
    deltas, estimado = reconstruir_regresion_directa(
        z, escalado_y, rsi_t=actual.rsi, macd_t=actual.macd, cierre_t=m.cierre_t)
    observado = Niveles(rsi.actualizar(m.cierre_t1), macd.actualizar(m.cierre_t1)[2])
    print(f"Origen UTC {m.fecha_prediccion} | Cierre {m.cierre_t:.2f} USDT")
    print(f"Observación UTC {m.fecha_objetivo} | Cierre {m.cierre_t1:.2f} USDT")
    tabla = pd.DataFrame({"Nivel en t": actual, "Salida z": z, "Delta recuperado": deltas,
                          "Estimado t+1": estimado, "Observado t+1": observado},
                         index=["RSI (delta en puntos)", "MACD (delta relativo al cierre)"])
    print(tabla.to_string(float_format=lambda v: f"{v:.10f}"))
    print("RSI estimado fuera de [0, 100] antes de recortar:", "sí" if not 0 <= estimado.rsi <= 100 else "no")

    # Delta cero en unidades originales: z = −media / escala.
    z_cero = [-escalado_y.at[n, "media"] / escalado_y.at[n, "escala"] for n in ObjetivosDirectos._fields]
    cambios, niveles = reconstruir_regresion_directa(
        z_cero, escalado_y, rsi_t=actual.rsi, macd_t=actual.macd, cierre_t=m.cierre_t)
    assert all(isclose(d, 0.0, abs_tol=1e-12) for d in cambios)
    assert all(isclose(a, b, rel_tol=0.0, abs_tol=1e-10) for a, b in zip(niveles, actual, strict=True))
    print(f"Deltas cero verificados | RSI {niveles.rsi:.10f} | MACD {niveles.macd:.10f}")


def main(regresion_directa=False):
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    velas = pd.read_parquet(ruta)
    if regresion_directa:
        demostrar_regresion_directa(velas)
        return
    fecha = pd.Timestamp("2025-01-20T17:59:59.999Z")
    muestra = next(m for m in construir_muestras(velas) if m.fecha_prediccion == fecha)
    rsi, macd, _ = _estados_al_cierre(velas, fecha)

    mu, nu = 0.0, 5.0
    sigma = muestra.X["volatilidad_ewma"].iloc[-1]
    s = sigma * sqrt((nu - 2) / nu)
    eventos, cuantiles = pronosticar(mu, s, nu, muestra, rsi, macd)
    gauss = ewma_gaussiana(muestra, rsi, macd).loc["macd_alcista"]
    alza = eventos.loc["macd_alcista"]
    assert isclose(student_t.std(df=nu, scale=s), sigma)
    assert alza.cierre_critico == gauss["Crítico"] and alza.retorno_critico == gauss["log(U/C)"]
    assert isclose(cuantiles.at[0.5, "cierre"], muestra.cierre_t)
    assert cuantiles.diff().iloc[1:].ge(0).all().all()

    print(f"Origen UTC: {fecha} | Cierre: {muestra.cierre_t:.2f}")
    print(f"mu={mu:g}, nu={nu:g}, s={s:.10f}, sigma EWMA={sigma:.10f}")
    print(eventos.to_string(float_format=lambda x: f"{x:.10g}"))
    print(cuantiles.to_string(float_format=lambda x: f"{x:.8f}"))
    print(f"Cruce alcista: cierre crítico compartido {alza.cierre_critico:.8f} USDT,"
          f" retorno crítico {alza.retorno_critico:.10f}")
    print(f"Student-t: {alza.probabilidad:.8%} | Gaussiana: {gauss['Probabilidad']:.8%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regresion-directa", action="store_true",
                        help="Reconstruye la primera predicción de evaluación del checkpoint directo de enero.")
    main(parser.parse_args().regresion_directa)
