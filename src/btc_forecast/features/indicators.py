"""MACD(12,26,9) y RSI(14) de Wilder incrementales sobre cierres horarios."""

import argparse
from copy import copy
from math import isfinite
from pathlib import Path

import pandas as pd


class _EMA:
    def __init__(self, periodo):
        self.periodo = periodo
        self.alfa = 2 / (periodo + 1)
        self.cantidad = 0
        self.suma = 0.0
        self.valor = None

    def actualizar(self, dato):
        if self.valor is None:
            self.suma += dato
            self.cantidad += 1
            if self.cantidad == self.periodo:
                self.valor = self.suma / self.periodo
        else:
            self.valor += self.alfa * (dato - self.valor)
        return self.valor


class MACDIncremental:
    """Usa una instancia por tramo continuo y una actualización por cierre."""

    def __init__(self):
        self.rapida = _EMA(12)
        self.lenta = _EMA(26)
        self.senal = _EMA(9)

    def actualizar(self, cierre):
        """Devuelve EMA12, EMA26, MACD, señal e histograma al completar la semilla."""
        if not isfinite(cierre) or cierre <= 0:
            raise ValueError("El cierre debe ser positivo y finito.")
        ema12 = self.rapida.actualizar(cierre)
        ema26 = self.lenta.actualizar(cierre)
        if ema26 is None:
            return None
        macd = ema12 - ema26
        senal = self.senal.actualizar(macd)
        if senal is None:
            return None
        return ema12, ema26, macd, senal, macd - senal


class RSIIncremental:
    """Inicializa G y D con 14 variaciones (15 cierres) de un tramo continuo."""

    def __init__(self):
        self.ultimo_cierre = None
        self.variaciones = 0
        self.suma_ganancias = 0.0
        self.suma_perdidas = 0.0
        self.ganancia_media = None
        self.perdida_media = None

    def actualizar(self, cierre):
        """Devuelve el RSI tras incorporar el cierre, o None durante la semilla."""
        if not isfinite(cierre) or cierre <= 0:
            raise ValueError("El cierre debe ser positivo y finito.")
        anterior = self.ultimo_cierre
        self.ultimo_cierre = cierre
        if anterior is None:
            return None

        variacion = cierre - anterior
        ganancia = max(variacion, 0.0)
        perdida = max(-variacion, 0.0)
        if self.variaciones < 14:
            self.suma_ganancias += ganancia
            self.suma_perdidas += perdida
            self.variaciones += 1
            if self.variaciones < 14:
                return None
            self.ganancia_media = self.suma_ganancias / 14
            self.perdida_media = self.suma_perdidas / 14
        else:
            self.ganancia_media = (13 * self.ganancia_media + ganancia) / 14
            self.perdida_media = (13 * self.perdida_media + perdida) / 14

        total = self.ganancia_media + self.perdida_media
        return 50.0 if total == 0 else 100 * self.ganancia_media / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calentamiento", type=int, default=200,
                        help="Velas iniciales que se procesan antes de mostrar resultados.")
    calentamiento = parser.parse_args().calentamiento
    if calentamiento < 0:
        parser.error("El calentamiento debe ser no negativo.")
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    tabla = pd.read_parquet(ruta, columns=["open_time", "close"]).sort_values("open_time")
    if tabla["open_time"].duplicated().any():
        raise ValueError("Hay aperturas duplicadas.")

    anterior = None
    ejemplo = []
    for apertura, precio in tabla.itertuples(index=False, name=None):
        if anterior is None or apertura - anterior != 3_600_000:
            indicador = MACDIncremental()
            wilder = RSIIncremental()
            cantidad = 0
            ejemplo.clear()
        anterior = apertura
        cierre = float(precio)
        valores = indicador.actualizar(cierre)
        cierre_previo = wilder.ultimo_cierre
        rsi = wilder.actualizar(cierre)
        cantidad += 1
        if cantidad > calentamiento and valores is not None:
            fecha = pd.to_datetime(apertura, unit="ms", utc=True)
            ejemplo.append((fecha, cierre, *valores))
            if len(ejemplo) == 2:
                break
    if len(ejemplo) < 2:
        raise ValueError("Faltan dos velas consecutivas tras el calentamiento.")
    print("Apertura UTC | cierre | EMA12 | EMA26 | MACD | señal | histograma")
    for fecha, *valores in ejemplo:
        print(fecha.isoformat(), *(f"{v:.8f}" for v in valores), sep=" | ")

    print("Semilla RSI: suma de ganancias y pérdidas en las primeras 14 variaciones")
    print(f"{wilder.suma_ganancias:.2f} / 14 | {wilder.suma_perdidas:.2f} / 14")
    simulado = copy(wilder)
    rsi_repetido = simulado.actualizar(cierre)
    print("Apertura UTC | cierre anterior | cierre actual | variación | ganancia media | pérdida media | RSI")
    for etiqueta, previo, estado, resultado in [
        (ejemplo[-1][0].isoformat(), cierre_previo, wilder, rsi),
        ("Siguiente (hipotética)", cierre, simulado, rsi_repetido),
    ]:
        fila = (previo, cierre, cierre - previo, estado.ganancia_media, estado.perdida_media, resultado)
        print(etiqueta, *(f"{v:.8f}" for v in fila), sep=" | ")


if __name__ == "__main__":
    main()
