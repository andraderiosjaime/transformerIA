"""Referencias a una hora para los niveles y eventos de RSI y MACD."""

from copy import deepcopy
from dataclasses import dataclass
from math import erfc, isfinite, log, sqrt
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from btc_forecast.data.dataset import Muestra, construir_muestras
from btc_forecast.data.splits import corte_cronologico
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import EVENTOS
from btc_forecast.features.thresholds import precio_critico_macd, precio_critico_rsi


class Niveles(NamedTuple):
    rsi: float
    macd: float


def persistencia(actual: Niveles) -> Niveles:
    """Pronostica para t+1 los niveles observados en t."""
    return actual


def cierre_constante(rsi: RSIIncremental, macd: MACDIncremental) -> Niveles:
    """Avanza copias del estado una vela usando el último cierre observado."""
    if rsi.ganancia_media is None or macd.senal.valor is None:
        raise ValueError("Ambos indicadores deben haber completado su inicialización.")
    cierre = rsi.ultimo_cierre
    siguiente_rsi = deepcopy(rsi).actualizar(cierre)
    siguiente_macd = deepcopy(macd).actualizar(cierre)
    return Niveles(siguiente_rsi, siguiente_macd[2])


@dataclass
class Frecuencia:
    positivos: int = 0
    conocidos: int = 0
    elegibles: int = 0

    @property
    def tasa(self) -> float | None:
        return self.positivos / self.elegibles if self.elegibles else None


def elegibilidad(muestra: Muestra) -> dict[str, bool]:
    actual = muestra.X.iloc[-1]
    return {
        "macd_alcista": actual.histograma_cierre <= 0,
        "macd_bajista": actual.histograma_cierre >= 0,
        "rsi_sobrecompra": actual.rsi_100 <= 0.70,
        "rsi_sobreventa": actual.rsi_100 >= 0.30,
    }


def ajustar_frecuencias(entrenamiento):
    """Cuenta una vez por origen y evento, solo cuando se conoce su etiqueta."""
    frecuencias = {evento: Frecuencia() for evento in EVENTOS}
    for muestra in entrenamiento:
        for evento, elegible in elegibilidad(muestra).items():
            etiqueta = muestra.etiquetas[evento]
            if pd.isna(etiqueta):
                continue
            cuenta = frecuencias[evento]
            cuenta.conocidos += 1
            cuenta.positivos += int(etiqueta)
            cuenta.elegibles += int(elegible)
    return frecuencias


def pronosticar_eventos(muestra, frecuencias):
    """Usa las tasas ya ajustadas; None indica falta de soporte en entrenamiento."""
    return {
        evento: frecuencias[evento].tasa if elegible else 0.0
        for evento, elegible in elegibilidad(muestra).items()
    }


def evaluar_umbral_gaussiano(cierre, critico, sigma, superior, elegible):
    """Devuelve log(U/C), su distancia con signo en sigmas y la probabilidad."""
    if not (isfinite(cierre) and cierre > 0 and isfinite(sigma) and sigma > 0 and isfinite(critico)):
        raise ValueError("Se requieren cierre y sigma positivos y finitos, y umbral finito.")
    if critico <= 0:
        # Todo cierre futuro es positivo. El cociente no admite logaritmo real.
        return None, None, float(elegible and superior)
    retorno = log(critico / cierre)
    distancia = retorno / sigma
    # erfc calcula también colas pequeñas sin restarlas de uno.
    argumento = distancia if superior else -distancia
    probabilidad = 0.5 * erfc(argumento / sqrt(2)) if elegible else 0.0
    return retorno, distancia, probabilidad


def ewma_gaussiana(muestra, rsi, macd):
    """Pronóstico a una hora con los estados y la volatilidad disponibles en t."""
    sigma = muestra.X["volatilidad_ewma"].iloc[-1]
    critico_macd = precio_critico_macd(macd)
    umbrales = {
        "macd_alcista": (critico_macd, True),
        "macd_bajista": (critico_macd, False),
        "rsi_sobrecompra": (precio_critico_rsi(rsi, 70), True),
        "rsi_sobreventa": (precio_critico_rsi(rsi, 30), False),
    }
    elegibles = elegibilidad(muestra)
    filas = []
    for evento, (critico, superior) in umbrales.items():
        retorno, distancia, probabilidad = evaluar_umbral_gaussiano(
            muestra.cierre_t, critico, sigma, superior, elegibles[evento]
        )
        filas.append((evento, elegibles[evento], ">" if superior else "<", muestra.cierre_t,
                      critico, retorno, sigma, distancia, probabilidad))
    return pd.DataFrame(filas, columns=["Evento", "Elegible", "Condición", "Cierre",
                        "Crítico", "log(U/C)", "Sigma", "Distancia", "Probabilidad"]).set_index("Evento")


def main():
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    velas = pd.read_parquet(ruta).sort_values("close_time")
    corte = pd.Timestamp("2025-01-20T11:59:59.999Z")
    entrenamiento, validacion = corte_cronologico(construir_muestras(velas), corte)
    frecuencias = ajustar_frecuencias(entrenamiento)
    probabilidades = {
        m.fecha_prediccion: pronosticar_eventos(m, frecuencias) for m in validacion
    }
    # El cierre es positivo: histograma/cierre conserva el signo del histograma.
    muestra = next((m for m in validacion if m.X["histograma_cierre"].iloc[-1] < 0), None)
    if muestra is None:
        raise ValueError("No hay muestras de validación con histograma MACD negativo.")

    fechas = pd.to_datetime(velas["close_time"], unit="ms", utc=True)
    historia = velas.set_index(fechas)["close"].loc[:muestra.fecha_prediccion]
    anterior = None
    for fecha, precio in historia.items():
        if anterior is None or fecha - anterior != pd.Timedelta(hours=1):
            rsi, macd = RSIIncremental(), MACDIncremental()
        anterior = fecha
        nivel_rsi = rsi.actualizar(float(precio))
        valores_macd = macd.actualizar(float(precio))

    actual = Niveles(nivel_rsi, valores_macd[2])
    persistente = persistencia(actual)
    constante = cierre_constante(rsi, macd)
    gaussiana = ewma_gaussiana(muestra, rsi, macd)
    observado = Niveles(rsi.actualizar(muestra.cierre_t1), macd.actualizar(muestra.cierre_t1)[2])

    print(f"Predicción UTC: {muestra.fecha_prediccion.isoformat()} | Histograma: {valores_macd[-1]:.8f}")
    print(f"Observación UTC: {muestra.fecha_objetivo.isoformat()}")
    print(f"Cierre en t: {muestra.cierre_t:.2f} | Cierre real en t+1: {muestra.cierre_t1:.2f}")
    tabla = pd.DataFrame([actual, persistente, constante, observado],
                         index=["Actual", "Persistencia", "Cierre constante", "Observado"]).T
    print(tabla.to_string(float_format=lambda valor: f"{valor:.8f}"))

    print("\nEntrenamiento: evento | positivos | conocidos | elegibles | positivos/elegibles")
    for evento, cuenta in frecuencias.items():
        tasa = "indefinida" if cuenta.tasa is None else f"{cuenta.tasa:.4%}"
        print(f"{evento} | {cuenta.positivos} | {cuenta.conocidos} | {cuenta.elegibles} | {tasa}")
    alcista = frecuencias["macd_alcista"]
    p = probabilidades[muestra.fecha_prediccion]["macd_alcista"]
    asignada = "indefinida" if p is None else f"{p:.4%}"
    print(f"Cruce alcista en {muestra.fecha_prediccion.isoformat()} (elegible): "
          f"{alcista.positivos}/{alcista.elegibles} = {asignada}")
    print(f"\nEWMA gaussiana en {muestra.fecha_prediccion.isoformat()}")
    print(gaussiana.to_string(na_rep="indefinido", formatters={
        "Elegible": lambda e: "sí" if e else "no",
        "Probabilidad": lambda p: f"{p:.8g}",
    }))


if __name__ == "__main__":
    main()
