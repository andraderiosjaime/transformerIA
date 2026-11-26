"""Umbrales del siguiente cierre para los cruces del MACD y las entradas del RSI."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyarrow.parquet import ParquetFile

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental


def precio_critico_macd(estado: MACDIncremental) -> float:
    """Cierre candidato para el que el siguiente histograma sería cero."""
    rapida, lenta, senal = estado.rapida, estado.lenta, estado.senal
    if senal.valor is None:
        raise ValueError("Faltan cierres para inicializar la señal del MACD.")
    a = (1 - rapida.alfa) * rapida.valor - (1 - lenta.alfa) * lenta.valor
    b = rapida.alfa - lenta.alfa
    return (senal.valor - a) / b


def precio_critico_rsi(estado: RSIIncremental, nivel: int) -> float:
    """Precio que separa los cierres por encima y por debajo del nivel de RSI."""
    if nivel not in (30, 70):
        raise ValueError("El nivel de entrada debe ser 30 o 70.")
    ganancia, perdida = estado.ganancia_media, estado.perdida_media
    if ganancia is None or perdida is None:
        raise ValueError("Faltan cierres para inicializar el RSI.")
    if ganancia == perdida == 0:
        # Con ambos promedios a cero, una subida lleva a 100 y una bajada a 0.
        return estado.ultimo_cierre
    q = nivel / (100 - nivel)
    if ganancia <= q * perdida:
        variacion = 13 * (q * perdida - ganancia)
    else:
        variacion = 13 * (perdida - ganancia / q)
    return estado.ultimo_cierre + variacion


def main():
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    hora_ms = 3_600_000
    calentamiento = 200
    with ParquetFile(ruta) as archivo:
        aperturas = archivo.read(columns=["open_time"])["open_time"].to_pylist()
        if aperturas != sorted(set(aperturas)):
            raise ValueError("Se requieren aperturas únicas en orden cronológico.")

        estado = MACDIncremental()
        wilder = RSIIncremental()
        cantidad = 0
        cierres = archivo.iter_batches(batch_size=1, columns=["close"])
        for i, (apertura, fila) in enumerate(zip(aperturas, cierres)):
            if i and apertura - aperturas[i - 1] != hora_ms:
                estado = MACDIncremental()
                wilder = RSIIncremental()
                cantidad = 0
            cierre = float(fila.column(0)[0].as_py())
            valores = estado.actualizar(cierre)
            rsi = wilder.actualizar(cierre)
            cantidad += 1
            if cantidad <= calentamiento or valores is None:
                continue
            _, _, macd, senal, histograma = valores
            if histograma >= 0 or i + 1 == len(aperturas) or aperturas[i + 1] != apertura + hora_ms:
                continue

            critico = precio_critico_macd(estado)
            distancia = 100 * (critico - cierre) / cierre
            fecha = datetime.fromtimestamp(apertura / 1000, timezone.utc)
            fin = fecha + timedelta(hours=1, milliseconds=-1)
            print(f"Apertura UTC: {fecha.isoformat()}")
            print(f"Cierre UTC: {fin.isoformat(timespec='milliseconds')}")
            print(f"Cierre BTCUSDT: {cierre:.2f}")
            print(f"MACD: {macd:.8f} | Señal: {senal:.8f} | Histograma: {histograma:.8f}")
            print(f"Precio crítico: {critico:.8f} USDT | Distancia: {distancia:+.8f}%")
            eventos = [
                ("Cruce alcista MACD", histograma, histograma <= 0, critico, ">"),
                ("Cruce bajista MACD", histograma, histograma >= 0, critico, "<"),
                ("Entrada en sobrecompra", rsi, rsi <= 70, precio_critico_rsi(wilder, 70), ">"),
                ("Entrada en sobreventa", rsi, rsi >= 30, precio_critico_rsi(wilder, 30), "<"),
            ]
            print("Evento | Estado (H o RSI) | Elegible | Cierre crítico (USDT) | Condición en t+1")
            for evento, actual, elegible, umbral, signo in eventos:
                condicion = f"cierre {signo} umbral"
                print(f"{evento} | {actual:.8f} | {'Sí' if elegible else 'No'} | {umbral:.8f} | {condicion}")
            return
    raise ValueError("No hay una vela elegible con una vela consecutiva disponible.")


if __name__ == "__main__":
    main()
