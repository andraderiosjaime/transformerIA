"""Cruces MACD a 15 minutos y pronósticos de RSI y MACD a 45–90 minutos."""
import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from torch import nn

from btc_forecast.data.ingest import descargar_rango
from btc_forecast.evaluation.metrics import precision_promedio
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.thresholds import precio_critico_macd
from btc_forecast.features.transforms import VARIABLES
from btc_forecast.models.decoder import Decoder

PASO = pd.Timedelta(minutes=15)
VENTANA = 96
SEMILLA = 43
DESPLAZAMIENTO = 0.5
ENTRADAS = (*VARIABLES, "distancia_critica_sigma", "pendiente_histograma", "aceleracion_histograma")
LOGISTICA = ("distancia_critica_sigma", "retorno_log", "volatilidad_ewma",
             "pendiente_histograma", "aceleracion_histograma")
DATOS = Path("data/raw/binance_spot/BTCUSDT/15m")
DESTINO = Path("artifacts/clasificacion_macd_15m_s43")
MODELOS = ("EWMA", "Logistica", "Transformer")


def leer_mes(mes, descargar, *, hasta=None):
    inicio = pd.Timestamp(mes + "-01", tz="UTC")
    fin_mes = inicio + pd.offsets.MonthBegin()
    fin = fin_mes if hasta is None else min(hasta, fin_mes)
    archivo = DATOS / f"{mes}.parquet"
    nuevo = not archivo.exists()
    if nuevo:
        if not descargar:
            raise FileNotFoundError(f"Falta {archivo}. Puede usar --descargar-faltantes.")
        columnas = pd.read_parquet(DATOS / "2026-01.parquet").columns
        tabla = pd.DataFrame(descargar_rango(inicio, fin, intervalo="15m"), columns=columnas)
    else:
        filtros = None if fin == fin_mes else [("open_time", "<", fin.value // 1_000_000)]
        tabla = pd.read_parquet(archivo, filters=filtros)
    esperado = pd.date_range(inicio, fin, inclusive="left", freq=PASO)
    aperturas = pd.DatetimeIndex(pd.to_datetime(tabla.open_time, unit="ms", utc=True))
    cierres = pd.DatetimeIndex(pd.to_datetime(tabla.close_time, unit="ms", utc=True))
    if not aperturas.equals(esperado) or not cierres.equals(esperado + PASO - pd.Timedelta(milliseconds=1)):
        raise ValueError(f"Serie discontinua, duplicada o incompleta en {mes}")
    ohlcv = tabla[["open", "high", "low", "close", "volume"]].astype(float)
    if (not np.isfinite(ohlcv).all().all() or (ohlcv.low <= 0).any() or (ohlcv.volume < 0).any()
            or (ohlcv.high < ohlcv[["open", "close"]].max(axis=1)).any()
            or (ohlcv.low > ohlcv[["open", "close"]].min(axis=1)).any()):
        raise ValueError(f"OHLCV inválido en {mes}")
    if nuevo and fin == fin_mes:
        with NamedTemporaryFile(dir=DATOS, suffix=".tmp", delete=False) as tmp:
            temporal = Path(tmp.name)
        try:
            tabla.to_parquet(temporal, index=False)
            temporal.replace(archivo)
        finally:
            temporal.unlink(missing_ok=True)
    return ohlcv.set_axis(cierres.rename("origen_utc"))


def preparar(velas):
    """Cada fila se emite al cierre. Solo las columnas de resultado miran t+1."""
    if not velas.index.to_series().diff().iloc[1:].eq(PASO).all():
        raise ValueError("Hay un hueco entre meses")
    c = velas.close
    macd, rsi = MACDIncremental(), RSIIncremental()
    filas = []
    for precio in c:
        estado = macd.actualizar(precio)
        nivel = rsi.actualizar(precio)
        if estado is None:
            filas.append((np.nan,) * 5)
        else:
            filas.append((nivel, *estado[2:], precio_critico_macd(macd)))
    d = pd.DataFrame(filas, columns=["rsi", "macd", "senal", "histograma", "critico"], index=c.index)
    r = np.log(c / c.shift())
    sigma = r.pow(2).ewm(alpha=0.06, adjust=False).mean().pow(0.5).clip(lower=1e-12)
    d["retorno_log"] = r
    d["volatilidad_ewma"] = sigma
    d["rango_relativo"] = (velas.high - velas.low) / c
    d["volumen_relativo_20"] = velas.volume / velas.volume.rolling(20).mean().replace(0, np.nan)
    d["rsi_100"] = d.rsi / 100
    for col in ("macd", "senal", "histograma"):
        d[col + "_cierre"] = d[col] / c
    # Un umbral no positivo permite cualquier cierre. El piso evita log(0).
    d["distancia_critica_sigma"] = np.log((d.critico / c).clip(lower=1e-12)) / sigma
    d["pendiente_histograma"] = d.histograma.diff() / c
    d["aceleracion_histograma"] = d.histograma.diff().diff() / c
    d["cierre"] = c
    d["cierre_siguiente"] = c.shift(-1)
    d["fin_etiqueta"] = c.index.to_series().shift(-1)
    d["elegible"] = d.histograma <= 0
    d["z_siguiente"] = r.shift(-1) / sigma
    d["etiqueta"] = (d.elegible & (d.histograma.shift(-1) > 0)
                     & (d.z_siguiente > DESPLAZAMIENTO)).astype("Int64")
    d.loc[d.fin_etiqueta.isna(), "etiqueta"] = pd.NA
    # Contrasta la etiqueta observada con el precio crítico calculado en t.
    conocidos = d.critico.notna() & d.fin_etiqueta.notna()
    por_precio = d.elegible & (d.cierre_siguiente > d.critico) & (d.z_siguiente > DESPLAZAMIENTO)
    if not (d.loc[conocidos, "etiqueta"] == por_precio[conocidos]).all():
        raise ValueError("El cruce observado discrepa del umbral analítico")
    d.loc[d.index[:200], list(ENTRADAS)] = np.nan
    d["ventana_valida"] = np.isfinite(d[list(ENTRADAS)]).all(axis=1).rolling(VENTANA).sum().eq(VENTANA)
    return d


def periodo(d, inicio, fin):
    desde, hasta = pd.Timestamp(inicio, tz="UTC"), pd.Timestamp(fin, tz="UTC")
    return np.flatnonzero(d.ventana_valida & (d.index >= desde) & (d.fin_etiqueta < hasta))


def origenes_elegibles(d, posiciones):
    return posiciones[d.elegible.to_numpy()[posiciones]]


class Clasificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(len(ENTRADAS))
        self.cabeza = nn.Linear(64, 1)

    def forward(self, x):
        return self.cabeza(self.decoder(x)[:, -1]).squeeze(-1)


def ventanas(x, indices):
    return x[indices[:, None] + torch.arange(1 - VENTANA, 1, device=x.device)]


def tensor_entradas(d, cfg, dispositivo):
    x = (d[list(ENTRADAS)].to_numpy() - cfg["centro"]) / cfg["escala"]
    return torch.as_tensor(x, dtype=torch.float32, device=dispositivo)


@torch.inference_mode()
def prob_red(red, x, idx):
    red.eval()
    indices = torch.as_tensor(idx, device=x.device)
    return torch.cat([red(ventanas(x, b)).sigmoid() for b in indices.split(128)]).cpu().numpy()


def ap(y, p):
    return float(precision_promedio(pd.DataFrame({"etiqueta": y, "probabilidad": p})))


def entrenar(d, ajuste, seleccion, cfg, dispositivo):
    x = tensor_entradas(d, cfg, dispositivo)
    y = torch.as_tensor(d.etiqueta.to_numpy(dtype="float32", na_value=np.nan), device=dispositivo)
    indices = torch.as_tensor(ajuste, device=dispositivo)
    red = Clasificador().to(dispositivo)
    optim = torch.optim.AdamW(red.parameters(), lr=3e-4)
    perdida = nn.BCEWithLogitsLoss()
    mejor_ap, mejor = -np.inf, None
    for epoca in range(1, 13):
        red.train()
        orden = indices[torch.randperm(len(indices), device=dispositivo)]
        for b in orden.split(64):
            optim.zero_grad()
            loss = perdida(red(ventanas(x, b)), y[b])
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita")
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            optim.step()
        actual = ap(y[seleccion].cpu().numpy(), prob_red(red, x, seleccion))
        print(f"Época {epoca}: AP interna con filtro {actual:.4f}", flush=True)
        if actual > mejor_ap:
            mejor_ap, mejor = actual, deepcopy(red.state_dict())
            cfg["epoca"] = epoca
        elif mejor is None:
            raise ValueError("No hay positivos para seleccionar la época")
        elif epoca - cfg["epoca"] >= 3:
            break
    red.load_state_dict(mejor)
    return red


def matriz_logistica(d, cfg):
    x = (d[list(ENTRADAS)] - cfg["centro"]) / cfg["escala"]
    return np.column_stack([np.ones(len(d)), x[list(LOGISTICA)]])


def ajustar_logistica(x, y):
    def objetivo(w):
        z = x @ w
        penalizacion = w.copy()
        penalizacion[0] = 0
        return (np.logaddexp(0, z).sum() - y @ z + 0.5 * penalizacion @ penalizacion,
                x.T @ (expit(z) - y) + penalizacion)
    res = minimize(objetivo, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x.tolist()


def puntuar(d, idx, red, cfg, device):
    g = d.iloc[idx].copy()
    g["regimen_activo"] = g.volatilidad_ewma >= cfg["sigma_min"]
    # La igualdad con el umbral no activa el evento.
    limite = np.maximum(g.distancia_critica_sigma, cfg["desplazamiento_sigma"])
    z = cfg["residuos_ewma"]
    g["EWMA"] = 1 - np.searchsorted(z, limite, side="right") / len(z)
    g["Logistica"] = expit(matriz_logistica(g, cfg) @ cfg["coef_logistica"])
    g["Transformer"] = prob_red(red, tensor_entradas(d, cfg, device), idx)
    return g


def ajustar(d, destino, device):
    ajuste = periodo(d, "2026-01-01", "2026-03-01")
    eleg = origenes_elegibles(d, ajuste)
    internas = origenes_elegibles(d, periodo(d, "2026-03-01", "2026-03-16"))
    cal = origenes_elegibles(d, periodo(d, "2026-03-16", "2026-04-01"))
    # Cada fecha pesa una vez, aunque aparezca en varias ventanas.
    usadas = np.unique(eleg[:, None] + np.arange(1 - VENTANA, 1))
    x = d.iloc[usadas][list(ENTRADAS)]
    cfg = dict(entradas=list(ENTRADAS), ventana=VENTANA, semilla=SEMILLA,
               desplazamiento_sigma=DESPLAZAMIENTO, centro=x.mean().tolist(),
               escala=x.std(ddof=0).replace(0, 1).tolist(),
               sigma_min=float(d.iloc[ajuste].volatilidad_ewma.median()),
               residuos_ewma=np.sort(d.iloc[ajuste].z_siguiente).tolist())
    cfg["coef_logistica"] = ajustar_logistica(matriz_logistica(d.iloc[eleg], cfg),
                                              d.iloc[eleg].etiqueta.to_numpy(dtype=float))
    sel = internas[d.volatilidad_ewma.iloc[internas].to_numpy() >= cfg["sigma_min"]]
    red = entrenar(d, eleg, sel, cfg, device)
    g = puntuar(d, cal, red, cfg, device)
    cfg["umbrales"] = {m: {key: float(g.loc[g.regimen_activo, m].quantile(q))
                            for key, q in [("P95", .95), ("P99", .99)]} for m in MODELOS}
    cfg["tramos"] = {"ajuste": ["2026-01-01", "2026-03-01"],
                     "seleccion": ["2026-03-01", "2026-03-16"],
                     "calibracion": ["2026-03-16", "2026-04-01"]}
    destino.mkdir(parents=True, exist_ok=True)
    torch.save({"modelo": red.cpu().state_dict(), "config": cfg}, destino / "clasificador.pt")
    print(f"Ajuste {len(eleg)} elegibles ({d.iloc[eleg].etiqueta.sum()} positivos). "
          f"Selección {len(sel)} con filtro. Calibración {g.regimen_activo.sum()} con filtro. "
          f"Época {cfg['epoca']}. Sigma mínima {cfg['sigma_min']:.8f}.")
    return g, cfg


def informar(g, cfg, destino, nombre):
    resumen = []
    for modelo in MODELOS:
        for clave, limite in cfg["umbrales"][modelo].items():
            alerta = g.regimen_activo & g[modelo].gt(limite)
            g[f"alerta_{modelo}_{clave}"] = alerta
            for poblacion, filas in [("elegibles", g), ("con_filtro", g.loc[g.regimen_activo])]:
                avisos = alerta.loc[filas.index]
                solo_umbral = filas[modelo].gt(limite)
                vp = int(filas.loc[avisos, "etiqueta"].sum())
                positivos = int(filas.etiqueta.sum())
                resumen.append(dict(modelo=modelo, corte=clave, poblacion=poblacion,
                                    n=len(filas), positivos=positivos, umbral=limite, avisos=int(avisos.sum()),
                                    aciertos=vp, falsas=int(avisos.sum())-vp,
                                    precision=vp/avisos.sum() if avisos.any() else np.nan,
                                    recall=vp/positivos if positivos else np.nan,
                                    AP=ap(filas.etiqueta.astype(int), filas[modelo]),
                                    avisos_sin_filtro=int(solo_umbral.sum()),
                                    aciertos_sin_filtro=int(filas.loc[solo_umbral, "etiqueta"].sum())))
    tabla = pd.DataFrame(resumen)
    tabla.to_csv(destino / f"metricas_{nombre}.csv", index=False)
    g.to_parquet(destino / f"predicciones_{nombre}.parquet")
    print(tabla.round(4).to_string(index=False))


# Horizontes en velas de 15 minutos.
PLAZOS = (("rsi", 3), ("macd", 4), ("macd", 5), ("macd", 6))
RUN_PLAZOS = Path("artifacts/anticipacion_45_90m_s43")


def referencias_plazos(d):
    """Prolonga ambas EMA desde t suponiendo C[t+k]=C[t] para todo k."""
    estado = MACDIncremental()
    filas = []
    for cierre, rsi in zip(d.cierre, d.rsi):
        valores = estado.actualizar(cierre)
        if valores is None:
            filas.append([rsi, np.nan, np.nan, np.nan])
            continue
        rapida, lenta = valores[:2]
        macd = [(rapida - cierre) * (1 - estado.rapida.alfa)**k
                - (lenta - cierre) * (1 - estado.lenta.alfa)**k for _, k in PLAZOS[1:]]
        filas.append([rsi, *macd])
    return np.array(filas)


def resultados_plazos(d):
    """Los objetivos son exclusivos del ajuste y de la puntuación retrospectiva."""
    reales = np.column_stack([d[col].shift(-k) for col, k in PLAZOS])
    futuros = np.column_stack([d.histograma.shift(-k) for k in range(1, 7)])
    positivos = futuros > 0
    primero = np.where(positivos.any(axis=1), positivos.argmax(axis=1) + 1, 0)
    # −1: origen inválido. 0: seis velas sin cruce.
    primero[~(d.elegible.to_numpy() & np.isfinite(futuros).all(axis=1))] = -1
    return reales, primero


class RedPlazos(nn.Module):
    """Cuatro correcciones de nivel y un logit para primer cruce en [4, 6]."""
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(len(ENTRADAS))
        self.salida = nn.Linear(64, len(PLAZOS) + 1)

    def forward(self, x):
        return self.salida(self.decoder(x)[:, -1])


def escala_unidades(d):
    cierre = d.cierre.to_numpy()
    return np.column_stack([np.ones(len(d)), cierre, cierre, cierre])


def indices_plazos(d, inicio, fin):
    # El último objetivo también debe caer dentro del tramo.
    return np.flatnonzero(d.ventana_valida & (d.index >= pd.Timestamp(inicio, tz="UTC"))
                          & (d.index + 6 * PASO < pd.Timestamp(fin, tz="UTC")))


@torch.inference_mode()
def salidas_plazos(red, x, idx):
    red.eval()
    partes = [red(ventanas(x, b)).cpu() for b in torch.as_tensor(idx, device=x.device).split(128)]
    return torch.cat(partes).numpy()


def ajustar_plazos(d, destino, device):
    ajuste = indices_plazos(d, "2026-01-01", "2026-03-01")
    validacion = indices_plazos(d, "2026-03-01", "2026-03-16")
    cal = origenes_elegibles(d, indices_plazos(d, "2026-03-16", "2026-04-01"))
    filas = np.unique(ajuste[:, None] + np.arange(1 - VENTANA, 1))
    entradas = d.iloc[filas][list(ENTRADAS)]
    reales, primero = resultados_plazos(d)
    residuo = (reales - referencias_plazos(d)) / escala_unidades(d)
    media = residuo[ajuste].mean(axis=0)
    sd = residuo[ajuste].std(axis=0)
    sd[sd == 0] = 1.
    cfg = dict(centro=entradas.mean().tolist(), escala=entradas.std(ddof=0).replace(0, 1).tolist(),
               media_y=media.tolist(), escala_y=sd.tolist(), entradas=list(ENTRADAS),
               plazos=list(PLAZOS), semilla=SEMILLA, ventana=VENTANA, prueba="2026-06",
               ajuste=["2026-01-01","2026-03-01"], seleccion=["2026-03-01","2026-03-16"],
               calibracion=["2026-03-16","2026-04-01"], cuantila_aviso=.95)
    x = tensor_entradas(d, cfg, device)
    y = torch.as_tensor((residuo - media) / sd, dtype=torch.float32, device=device)
    evento = torch.as_tensor(primero >= 4, dtype=torch.float32, device=device)
    eleg = torch.as_tensor(primero >= 0, dtype=torch.float32, device=device)
    red = RedPlazos().to(device)
    opt = torch.optim.AdamW(red.parameters(), lr=3e-4)

    def perdida(pred, idx):
        niveles = nn.functional.huber_loss(pred[:, :-1], y[idx])
        bce = nn.functional.binary_cross_entropy_with_logits(pred[:, -1], evento[idx], reduction="none")
        return niveles + (bce * eleg[idx]).sum() / eleg[idx].sum().clamp_min(1)

    mejor = np.inf
    ix = torch.as_tensor(ajuste, device=device)
    for epoca in range(1, 13):
        red.train()
        for b in ix[torch.randperm(len(ix), device=device)].split(64):
            opt.zero_grad()
            loss = perdida(red(ventanas(x, b)), b)
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        sal = torch.as_tensor(salidas_plazos(red, x, validacion), device=device)
        val = perdida(sal, validacion).item()
        if not np.isfinite(val):
            raise ValueError("Pérdida de validación no finita")
        print(f"Época {epoca}  validación {val:.5f}", flush=True)
        if val < mejor:
            mejor, cfg["epoca"], pesos = val, epoca, deepcopy(red.state_dict())
        elif epoca - cfg["epoca"] >= 3:
            break
    red.load_state_dict(pesos)
    e = origenes_elegibles(d, ajuste)
    cfg["logistica"] = ajustar_logistica(matriz_logistica(d.iloc[e], cfg), (primero[e] >= 4).astype(float))
    p_red = expit(salidas_plazos(red, x, cal)[:, -1])
    p_log = expit(matriz_logistica(d.iloc[cal], cfg) @ cfg["logistica"])
    cfg["umbrales"] = [float(np.quantile(p, cfg["cuantila_aviso"])) for p in (p_red, p_log)]
    destino.mkdir(parents=True, exist_ok=True)
    torch.save(dict(modelo=red.cpu().state_dict(), config=cfg), destino / "plazos.pt")
    print("Ajuste, selección, calibración elegible:", len(ajuste), len(validacion), len(cal),
          "| Mejor época:", cfg["epoca"], "|", destino)


def emitir_plazos(d, idx, red, cfg, device):
    """Funciona también sobre el último cierre, aunque todos los objetivos falten."""
    salida = salidas_plazos(red, tensor_entradas(d, cfg, device), idx)
    refs = referencias_plazos(d)[idx]
    pred = refs + (salida[:, :-1] * cfg["escala_y"] + cfg["media_y"]) * escala_unidades(d)[idx]
    g = d.iloc[idx][["cierre", "rsi", "macd", "histograma", "elegible"]].copy()
    for j, (indicador, k) in enumerate(PLAZOS):
        nombre = f"{indicador}_{15*k}m"
        g["fecha_" + nombre] = g.index + k * PASO
        g["ref_" + nombre] = refs[:, j]
        g["pred_" + nombre] = pred[:, j]
    for modelo, umbral, p in zip(("red", "logistica"), cfg["umbrales"],
                                (expit(salida[:, -1]), expit(matriz_logistica(d.iloc[idx], cfg) @ cfg["logistica"]))):
        g["p_" + modelo] = np.where(g.elegible, p, np.nan)
        g["aviso_" + modelo] = g.elegible & (p > umbral)
    return g


def evaluar_plazos(d, idx, g, destino, mes):
    reales, primero = resultados_plazos(d)
    g["primer_cruce_velas"] = primero[idx]
    for j, (indicador, k) in enumerate(PLAZOS):
        nombre = f"{indicador}_{15*k}m"
        g["real_" + nombre] = reales[idx, j]
        error = (g["pred_" + nombre] - g["real_" + nombre]).abs()
        base = (g["ref_" + nombre] - g["real_" + nombre]).abs()
        # Remuestrea días UTC con ambos errores emparejados.
        diario = pd.DataFrame({"red": error, "base": base}).groupby(g.index.normalize()).sum()
        dias = np.random.default_rng(SEMILLA).integers(len(diario), size=(10000, len(diario)))
        ea, eb = diario.to_numpy()[dias].sum(axis=1).T
        inferior, superior = np.percentile(100 * (1 - ea/eb), [2.5, 97.5])
        print(f"{nombre}: n={len(g)} MAE red {error.mean():.4f}, referencia {base.mean():.4f}, "
              f"mejora {100*(1-error.mean()/base.mean()):+.2f}% (IC95 {inferior:+.2f}, {superior:+.2f}%)")
    e = g[g.elegible]
    y = e.primer_cruce_velas >= 4
    for modelo in ("red", "logistica"):
        avisos = e["aviso_" + modelo]
        aciertos = e.loc[avisos & y]
        fechas = aciertos.index + aciertos.primer_cruce_velas * PASO
        print(f"{modelo}: {y.sum()}/{len(e)} orígenes positivos, AP {ap(y, e['p_'+modelo]):.4f}, "
              f"{avisos.sum()} avisos, {len(aciertos)} correctos para {fechas.nunique()} cruces distintos.")
    g.to_parquet(destino / f"predicciones_{mes}.parquet")


def ejecutar_plazos(args):
    ajustar_red = args.modo == "ajustar-plazos"
    solo_origen = args.modo == "pronosticar-plazos"
    destino = args.destino or RUN_PLAZOS
    if solo_origen:
        if args.origen == "ahora":
            origen = pd.Timestamp.now(tz="UTC").floor(PASO) - pd.Timedelta(milliseconds=1)
        else:
            origen = pd.Timestamp(args.origen)
        if origen.tzinfo is None:
            raise ValueError("Indique --origen con fecha exacta de cierre y zona UTC.")
        if origen < pd.Timestamp("2026-04-01", tz="UTC"):
            raise ValueError("El checkpoint y sus umbrales están disponibles desde abril de 2026.")
        if origen >= pd.Timestamp.now(tz="UTC") or (origen + pd.Timedelta(milliseconds=1)).floor(PASO) != origen + pd.Timedelta(milliseconds=1):
            raise ValueError("Se requiere un instante de cierre de 15 minutos ya observado.")
        fin = origen.tz_convert("UTC").strftime("%Y-%m")
    else:
        fin = "2026-03" if ajustar_red else args.mes
    meses = pd.period_range("2026-01", fin, freq="M")
    hasta = origen + pd.Timedelta(milliseconds=1) if solo_origen else None
    v = pd.concat([leer_mes(str(m), args.descargar_faltantes, hasta=hasta) for m in meses])
    if solo_origen:
        if origen not in v.index:
            raise ValueError("El origen debe coincidir con un cierre real de 15 minutos.")
        v = v.loc[:origen]
    d = preparar(v)
    if ajustar_red:
        ajustar_plazos(d, destino, args.dispositivo)
        return
    ck = torch.load(destino / "plazos.pt", map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg["entradas"] != list(ENTRADAS) or cfg["plazos"] != list(PLAZOS):
        raise ValueError("Checkpoint incompatible con las entradas o los horizontes.")
    red = RedPlazos().to(args.dispositivo)
    red.load_state_dict(ck["modelo"])
    inicio = pd.Timestamp(fin + "-01")
    idx = np.array([len(d)-1]) if solo_origen else indices_plazos(d, inicio, inicio + pd.offsets.MonthBegin())
    if not len(idx) or not d.ventana_valida.iloc[idx].all():
        raise ValueError("Se necesita más historia consecutiva para emitir el pronóstico.")
    g = emitir_plazos(d, idx, red, cfg, args.dispositivo)
    if solo_origen:
        print(g.T.to_string())
    else:
        evaluar_plazos(d, idx, g, destino, fin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modo", choices=["ajustar", "evaluar", "ajustar-plazos", "evaluar-plazos", "pronosticar-plazos"])
    parser.add_argument("--mes", choices=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--origen", help="Cierre UTC exacto o 'ahora' para la última vela cerrada.")
    parser.add_argument("--dispositivo", default="cuda:0")
    parser.add_argument("--destino", type=Path)
    parser.add_argument("--descargar-faltantes", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEMILLA)
    if args.modo.endswith("-plazos"):
        if args.modo == "pronosticar-plazos" and not args.origen:
            parser.error("pronosticar-plazos requiere --origen")
        args.mes = args.mes or "2026-06"
        ejecutar_plazos(args)
        return
    args.mes = args.mes or "2026-04"
    if args.mes == "2026-06":
        parser.error("El experimento original usa abril o mayo.")
    args.destino = args.destino or DESTINO
    fin = "2026-03" if args.modo == "ajustar" else args.mes
    meses = pd.period_range("2026-01", fin, freq="M")
    d = preparar(pd.concat([leer_mes(str(m), args.descargar_faltantes) for m in meses]))
    if args.modo == "ajustar":
        g, cfg = ajustar(d, args.destino, args.dispositivo)
        nombre = "calibracion"
    else:
        ck = torch.load(args.destino / "clasificador.pt", map_location="cpu", weights_only=True)
        cfg = ck["config"]
        if cfg["entradas"] != list(ENTRADAS) or cfg["desplazamiento_sigma"] != DESPLAZAMIENTO:
            raise ValueError("El checkpoint no corresponde a estas variables o etiqueta")
        red = Clasificador().to(args.dispositivo)
        red.load_state_dict(ck["modelo"])
        inicio = pd.Timestamp(args.mes + "-01")
        idx = periodo(d, inicio, inicio + pd.offsets.MonthBegin())
        g = puntuar(d, origenes_elegibles(d, idx), red, cfg, args.dispositivo)
        nombre = args.mes
        print(f"{nombre}: {len(idx)} ventanas, {len(g)} elegibles, {g.etiqueta.sum()} positivos; "
              f"{g.regimen_activo.sum()} elegibles pasan el filtro.")
    informar(g, cfg, args.destino, nombre)


if __name__ == "__main__":
    main()
