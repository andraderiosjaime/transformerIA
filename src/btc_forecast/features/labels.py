"""Etiquetas observadas de MACD y RSI a una hora sobre velas cerradas."""

from pathlib import Path

import pandas as pd

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental


EVENTOS = ("macd_alcista", "macd_bajista", "rsi_sobrecompra", "rsi_sobreventa")
HORA = pd.Timedelta(hours=1)


def cruce_alcista_observado(histograma_t, histograma_t1):
    return (histograma_t <= 0) & (histograma_t1 > 0)


def etiquetar_velas(tabla, calentamiento=200):
    """Devuelve indicadores y etiquetas por apertura UTC. Cada hueco reinicia el tramo."""
    if calentamiento < 0:
        raise ValueError("El calentamiento debe ser no negativo.")
    aperturas = pd.to_datetime(tabla["open_time"], unit="ms", utc=True)
    if aperturas.isna().any() or aperturas.duplicated().any():
        raise ValueError("Las aperturas deben ser válidas y únicas.")
    cierres = tabla.set_index(aperturas)["close"].sort_index()
    histogramas = []
    niveles_rsi = []
    anterior = None
    for fecha, cierre in cierres.items():
        if anterior is None or fecha - anterior != HORA:
            macd = MACDIncremental()
            wilder = RSIIncremental()
            cantidad = 0
        anterior = fecha
        precio = float(cierre)
        valores = macd.actualizar(precio)
        rsi = wilder.actualizar(precio)
        cantidad += 1
        histogramas.append(valores[-1] if cantidad > calentamiento and valores else None)
        niveles_rsi.append(rsi if cantidad > calentamiento else None)

    # Float64 propaga los valores desconocidos.
    resultado = pd.DataFrame({"histograma": histogramas, "rsi": niveles_rsi},
                             index=cierres.index, dtype="Float64")
    # Empareja por fecha para respetar los huecos.
    siguientes = resultado.shift(freq=-HORA).reindex(resultado.index)
    h, h1 = resultado["histograma"], siguientes["histograma"]
    r, r1 = resultado["rsi"], siguientes["rsi"]
    reglas = {
        "macd_alcista": cruce_alcista_observado(h, h1),
        "macd_bajista": (h >= 0) & (h1 < 0),
        "rsi_sobrecompra": (r <= 70) & (r1 > 70),
        "rsi_sobreventa": (r >= 30) & (r1 < 30),
    }
    for evento, regla in reglas.items():
        indicador = "histograma" if evento.startswith("macd") else "rsi"
        observado = resultado[indicador].notna() & siguientes[indicador].notna()
        resultado[evento] = regla.where(observado).astype("Int8")
    return resultado


def main():
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    tabla = pd.read_parquet(ruta, columns=["open_time", "close"])
    resultado = etiquetar_velas(tabla)
    etiquetas = resultado.loc[:, list(EVENTOS)]
    print(f"Filas con cuatro etiquetas: {len(etiquetas.dropna())} de {len(tabla)}")
    print("Positivos por evento:")
    print(etiquetas.sum().to_string())
    positivos = etiquetas.eq(1).any(axis=1)
    if positivos.any():
        t = positivos[positivos].index[0]
        print("Ejemplo con aperturas UTC:")
        print(resultado.loc[[t, t + HORA], ["histograma", "rsi"]].to_string())
        print("Etiquetas del origen:", etiquetas.loc[t].to_dict())


if __name__ == "__main__":
    main()
