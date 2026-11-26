"""Métricas a una hora por indicador o evento, con fechas de predicción UTC."""

from pathlib import Path

import pandas as pd

from btc_forecast.data.dataset import construir_muestras
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.models.baselines import cierre_constante, ewma_gaussiana


def _alinear(predicciones, observaciones):
    """Una tabla por modelo: índice de fechas de origen y una columna por objetivo."""
    tablas = []
    for tabla in (predicciones, observaciones):
        fechas = pd.to_datetime(tabla.index, utc=True)
        if fechas.hasnans or not fechas.is_unique or not tabla.columns.is_unique:
            raise ValueError("Se necesitan fechas válidas y claves fecha/objetivo únicas.")
        valores = tabla.apply(pd.to_numeric, errors="coerce").astype(float)
        valores.index = fechas
        tablas.append(valores.replace([float("inf"), -float("inf")], float("nan")))
    return tablas[0].align(tablas[1], join="outer")


def mae_niveles(predicciones, observaciones):
    """Media del error absoluto y número de pares finitos por columna de nivel."""
    p, y = _alinear(predicciones, observaciones)
    errores = (p - y).abs()
    return pd.DataFrame({"MAE": errores.mean(), "n": errores.count()})


def brier_eventos(predicciones, observaciones):
    """Brier por evento; solo cuentan pares con p en [0,1] e y binaria."""
    p, y = _alinear(predicciones, observaciones)
    validos = p.ge(0) & p.le(1) & y.isin([0, 1])
    terminos = (p - y).where(validos).pow(2)
    return pd.DataFrame({"Brier": terminos.mean(), "n": terminos.count()})


def cobertura_intervalo(q10, q90, observaciones):
    """Fracción de observaciones en [q10, q90] con los tres valores disponibles."""
    inferior, y = _alinear(q10, observaciones)
    superior, y = _alinear(q90, y)
    inferior = inferior.reindex_like(y)
    if inferior.gt(superior).any().any():
        raise ValueError("El cuantil 10 supera al cuantil 90.")
    validos = inferior.notna() & superior.notna() & y.notna()
    dentro = (y.ge(inferior) & y.le(superior)).where(validos)
    return pd.DataFrame({"n": dentro.count(), "dentro": dentro.sum().astype(int),
                         "cobertura": dentro.mean()})


def precision_promedio(pares):
    """AP no interpolada; las predicciones empatadas entran en el mismo umbral."""
    grupos = pares.groupby("probabilidad")["etiqueta"].agg(["sum", "size"]).sort_index(ascending=False)
    positivos = grupos["sum"].sum()
    if positivos == 0:
        return float("nan")
    precision = grupos["sum"].cumsum() / grupos["size"].cumsum()
    return (precision * grupos["sum"]).sum() / positivos


def resumir_eventos(predicciones):
    """Calcula AP y calibración por modelo, evento y población con pares válidos."""
    eventos = predicciones.loc[predicciones.tipo.eq("evento")]
    if eventos.duplicated(["modelo", "evento", "fecha_prediccion"]).any():
        raise ValueError("Hay predicciones duplicadas para un modelo, evento y fecha.")
    resumen, calibracion = [], []
    bordes = [i / 10 for i in range(11)]
    for (modelo, evento), grupo in eventos.groupby(["modelo", "evento"]):
        validos = grupo.probabilidad.between(0, 1) & grupo.etiqueta.isin([0, 1])
        for poblacion, mascara in [("todas", validos), ("elegibles", validos & grupo.elegible)]:
            pares = grupo.loc[mascara]
            claves = {"modelo": modelo, "evento": evento, "poblacion": poblacion}
            resumen.append({**claves, "n": len(pares), "positivos": int(pares.etiqueta.sum()),
                            "AP": precision_promedio(pares)})
            intervalos = pd.cut(pares.probabilidad, bordes, include_lowest=True,
                                labels=["[0, 0.1]"] + [f"({i/10:g}, {(i+1)/10:g}]" for i in range(1, 10)])
            tabla = pares.groupby(intervalos, observed=False).agg(
                n=("etiqueta", "size"), positivos=("etiqueta", "sum"),
                prob_media=("probabilidad", "mean"), frecuencia=("etiqueta", "mean"))
            calibracion.append(tabla.rename_axis("intervalo").reset_index().assign(**claves))
    return pd.DataFrame(resumen), pd.concat(calibracion, ignore_index=True)


def cobertura_enero(velas, predicciones, checkpoint):
    """Reconstruye los cuantiles omitidos en el Parquet y contrasta la mediana."""
    import torch
    from btc_forecast.features.transforms import aplicar_escalado
    from btc_forecast.models.forecast import pronosticar
    from btc_forecast.training.trainer import cargar_checkpoint

    modelo, escalado, meta = cargar_checkpoint(checkpoint)
    niveles = predicciones.loc[predicciones.modelo.eq("transformer") & predicciones.tipo.eq("nivel")]
    medianas = niveles.pivot(index="fecha_prediccion", columns="objetivo", values="prediccion")
    y = niveles.pivot(index="fecha_prediccion", columns="objetivo", values="observado")
    muestras = [m for m in construir_muestras(velas, longitud=meta["ventana"]) if m.fecha_prediccion in y.index]
    fechas = pd.DatetimeIndex([m.fecha_prediccion for m in muestras])
    if not fechas.equals(y.index):
        raise ValueError("Las fechas reconstruidas difieren de las evaluadas.")
    parametros = {}
    with torch.inference_mode():
        for inicio in range(0, len(muestras), 16):
            lote = muestras[inicio:inicio + 16]
            x = torch.stack([torch.from_numpy(aplicar_escalado(m.X, escalado).to_numpy(dtype="float32", copy=True))
                             for m in lote])
            mu, s, nu = modelo(x)
            for m, centro, escala in zip(lote, mu.tolist(), s.tolist(), strict=True):
                parametros[m.fecha_prediccion] = (m, centro, escala, nu.item())

    limites = {0.1: [], 0.5: [], 0.9: []}
    anterior = None
    cierres = velas.set_index(pd.to_datetime(velas.close_time, unit="ms", utc=True))["close"].sort_index()
    for fecha, cierre in cierres.loc[:fechas[-1]].items():
        if anterior is None or fecha - anterior != pd.Timedelta(hours=1):
            rsi, macd = RSIIncremental(), MACDIncremental()
        anterior = fecha
        rsi.actualizar(float(cierre))
        macd.actualizar(float(cierre))
        if fecha in parametros:
            m, mu, s, nu = parametros[fecha]
            _, cuantiles = pronosticar(mu, s, nu, m, rsi, macd)
            for q, filas in limites.items():
                filas.append(cuantiles.loc[q].rename({"RSI": "rsi", "MACD": "macd"}))
    tablas = {q: pd.DataFrame(filas, index=y.index) for q, filas in limites.items()}
    pd.testing.assert_frame_equal(tablas[0.5][medianas.columns], medianas, check_names=False,
                                  rtol=1e-6, atol=1e-6)
    y["cierre"] = [m.cierre_t1 for m in muestras]
    return cobertura_intervalo(tablas[0.1], tablas[0.9], y)


def comprobar_ejemplo_conocido():
    fechas = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    p = pd.DataFrame({"macd": [4, 9, 100], "rsi": [50, None, 60]}, index=fechas)
    y = pd.DataFrame({"rsi": [54, 49], "macd": [7, 5]}, index=fechas[:2][::-1])
    niveles = mae_niveles(p, y)
    assert niveles.loc["macd"].to_dict() == {"MAE": 1.5, "n": 2}
    assert niveles.loc["rsi"].to_dict() == {"MAE": 1, "n": 1}
    p = pd.DataFrame({"alcista": [0.25, 0.5, None], "bajista": [0, 1, 0]}, index=fechas)
    y = pd.DataFrame({"bajista": [1, 0], "alcista": [1, 0]}, index=fechas[:2][::-1])
    eventos = brier_eventos(p, y)
    assert eventos.loc["alcista"].to_dict() == {"Brier": 0.15625, "n": 2}
    assert eventos.loc["bajista"].to_dict() == {"Brier": 0, "n": 2}
    print("Ejemplo conocido: MAE MACD = (1 + 2)/2 = 1.5; Brier alcista = (0.25² + 0.5²)/2 = 0.15625")


def main():
    comprobar_ejemplo_conocido()
    tabla = pd.read_parquet(Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet"))
    t = pd.Timestamp("2025-01-20T17:59:59.999Z")
    muestra = next(m for m in construir_muestras(tabla) if m.fecha_prediccion == t)
    fechas = pd.to_datetime(tabla["close_time"], unit="ms", utc=True)
    historia = tabla.set_index(fechas)["close"].sort_index().loc[:t]
    anterior = None
    for fecha, cierre in historia.items():
        if anterior is None or fecha - anterior != pd.Timedelta(hours=1):
            rsi, macd = RSIIncremental(), MACDIncremental()
        anterior = fecha
        rsi.actualizar(float(cierre))
        macd.actualizar(float(cierre))

    niveles = cierre_constante(rsi, macd)
    probabilidades = ewma_gaussiana(muestra, rsi, macd)["Probabilidad"]
    observados = {"rsi": rsi.actualizar(muestra.cierre_t1),
                  "macd": macd.actualizar(muestra.cierre_t1)[2]}
    mae = mae_niveles(pd.DataFrame([niveles._asdict()], index=[t]),
                     pd.DataFrame([observados], index=[t]))
    brier = brier_eventos(pd.DataFrame([probabilidades], index=[t]),
                         pd.DataFrame([muestra.etiquetas], index=[t]))
    print(f"Predicción: {t} | Observación: {muestra.fecha_objetivo}")
    print(mae.to_string())
    print(brier.to_string(float_format=lambda valor: f"{valor:.10g}"))
    evento = "macd_alcista"
    print(f"Error absoluto MACD = |{niveles.macd:.8f} - {observados['macd']:.8f}|"
          f" = {mae.at['macd', 'MAE']:.8f}")
    print(f"Brier alcista = ({probabilidades[evento]:.12g} - {muestra.etiquetas[evento]})²"
          f" = {brier.at[evento, 'Brier']:.12g}")

    carpeta = Path("artifacts")
    predicciones = pd.read_parquet(carpeta / "predicciones_BTCUSDT_1h_1737374399999000000_transformer_s42.parquet")
    print("\nCobertura del intervalo central del 80 % (Transformer)")
    print(cobertura_enero(tabla, predicciones, carpeta / "mejor_student_t_20250120_semilla42.pt").to_string())
    ap, calibracion = resumir_eventos(predicciones)
    print("\nPrecisión promedio")
    print(ap.to_string(index=False, na_rep="indefinida"))
    print("\nCalibración (intervalos con muestras)")
    print(calibracion.loc[calibracion.n.gt(0)].to_string(index=False))


if __name__ == "__main__":
    main()
