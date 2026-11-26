"""Auditoría de las velas horarias de Binance Spot en UTC."""

from pathlib import Path

import pandas as pd


PARQUET = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")


def auditar(tabla, inicio, fin, ahora):
    """Devuelve los problemas y las aperturas afectadas en [inicio, fin)."""
    aperturas = pd.to_datetime(tabla["open_time"], unit="ms", utc=True, errors="coerce")
    cierres = pd.to_datetime(tabla["close_time"], unit="ms", utc=True, errors="coerce")
    esperadas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    faltantes = esperadas.difference(aperturas.dropna())
    problemas = {}
    if len(faltantes):
        problemas["Hora ausente"] = faltantes.tolist()

    def registrar(motivo, mascara):
        if mascara.any():
            problemas[motivo] = [
                fecha if pd.notna(fecha) else f"fila {indice} (apertura ilegible)"
                for indice, fecha in aperturas[mascara].items()
            ]

    registrar("Apertura fuera de la cuadrícula esperada", ~aperturas.isin(esperadas))
    registrar("Apertura duplicada", aperturas.duplicated(keep=False) & aperturas.notna())
    cierre_esperado = aperturas + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    registrar("Cierre horario incorrecto", cierres.isna() | cierres.ne(cierre_esperado))
    registrar("Vela aún abierta", cierres.ge(ahora))

    numeros = tabla[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan"))
    for campo in ("open", "high", "low", "close"):
        registrar(f"Precio inválido o no positivo ({campo})",
                  numeros[campo].isna() | numeros[campo].le(0))
    for campo in ("open", "close"):
        registrar(f"{campo} fuera de [low, high]",
                  numeros[campo].lt(numeros["low"]) | numeros[campo].gt(numeros["high"]))
    registrar("Volumen inválido o negativo", numeros["volume"].isna() | numeros["volume"].lt(0))
    return problemas


def imprimir_informe(titulo, problemas):
    print(titulo)
    if not problemas:
        print("  Sin problemas.")
    for motivo, fechas in problemas.items():
        print(f"  {motivo}: {', '.join(map(str, fechas))}")


def main():
    tabla = pd.read_parquet(PARQUET)
    inicio = pd.Timestamp("2025-01-01", tz="UTC")
    fin = pd.Timestamp("2025-02-01", tz="UTC")
    ahora = pd.Timestamp.now(tz="UTC")
    imprimir_informe(f"Enero de 2025 ({len(tabla)} filas)", auditar(tabla, inicio, fin, ahora))

    objetivo = pd.Timestamp("2025-01-15 12:00", tz="UTC")
    indices = tabla.index[tabla["open_time"].eq(objetivo.value // 1_000_000)]
    if len(indices) != 1:
        print(f"Demostración omitida: se esperaba una sola vela para {objetivo}.")
        return
    copia = tabla.drop(index=indices[0])
    imprimir_informe(f"Copia en memoria ({len(copia)} filas)", auditar(copia, inicio, fin, ahora))


if __name__ == "__main__":
    main()
