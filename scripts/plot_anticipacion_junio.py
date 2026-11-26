"""Figuras de junio de 2026 desde los pronósticos congelados a +45/+60/+75/+90 min."""
from pathlib import Path
from datetime import timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd

from btc_forecast.evaluation.metrics import precision_promedio

CARPETA = Path(__file__).resolve().parents[1] / "artifacts/anticipacion_45_90m_s43"
PASO = pd.Timedelta(minutes=15)
PLAZOS = {"rsi": (45,), "macd": (60, 75, 90)}
AZUL, NARANJA, VERDE, GRIS = "#193950", "#AD4521", "#007467", "#526674"


def numero(x, dec=2, signo=False):
    return format(x, f"{'+' if signo else ''}.{dec}f").replace(".", ",").replace("-", "−")


def leer():
    g = pd.read_parquet(CARPETA / "predicciones_2026-06.parquet").sort_index()
    esperados = pd.date_range("2026-06-01 00:14:59.999", periods=2874, freq=PASO, tz="UTC")
    if not g.index.equals(esperados):
        raise ValueError("Faltan orígenes de junio, hay duplicados o difiere el calendario.")
    series = {}
    for indicador, plazos in PLAZOS.items():
        for minutos in plazos:
            nombre = f"{indicador}_{minutos}m"
            fecha = g["fecha_" + nombre]
            if not (fecha - g.index).eq(pd.Timedelta(minutes=minutos)).all():
                raise ValueError(f"Pronóstico mal alineado en {nombre}")
            d = g[[p + "_" + nombre for p in ("real", "pred", "ref")]].copy()
            d.columns = ["real", "pred", "ref"]
            if not np.isfinite(d).all().all():
                raise ValueError(f"Hay niveles sin valor en {nombre}")
            d["origen"] = g.index
            d.index = pd.DatetimeIndex(fecha)
            series[minutos] = d
    # A igual cierre objetivo, el MACD real debe coincidir.
    for minutos in (75, 90):
        a, b = series[60].real.align(series[minutos].real, join="inner")
        np.testing.assert_array_equal(a, b)
    return g, series


def metricas(d):
    errores = d[["pred", "ref"]].sub(d.real, axis=0).abs()
    mae, referencia = errores.mean()
    dias = errores.groupby(d.origen.dt.normalize()).sum().to_numpy()
    indices = np.random.default_rng(43).integers(len(dias), size=(10000, len(dias)))
    red, base = dias[indices].sum(axis=1).T
    ic = np.percentile(100 * (1 - red/base), [2.5, 97.5])
    return mae, referencia, 100 * (1 - mae/referencia), ic


def cabecera(fig, titulo, subtitulo):
    fig.text(.067, .935, titulo, fontsize=26, color=AZUL, weight="bold")
    fig.text(.068, .885, subtitulo, fontsize=14, color=GRIS)


def curvas(arriba, abajo, d, horas):
    arriba.plot(d.index, d.real, color=AZUL, lw=1.8, label="Real")
    arriba.plot(d.index, d.pred, color=NARANJA, lw=1.8, ls="--", label="Transformer")
    arriba.plot(d.index, d.ref, color=VERDE, lw=1.8, ls=":", label="Referencia")
    abajo.axhline(0, lw=.85, color=GRIS)
    abajo.plot(d.index, d.real-d.ref, color=AZUL, lw=1.5, label="Observado − referencia")
    abajo.plot(d.index, d.pred-d.ref, color=NARANJA, lw=1.8, ls="--", label="Pronóstico − referencia")
    for ax in (arriba, abajo):
        ax.grid(axis="y", alpha=.2)
        ax.set_xlim(d.index[0], d.index[-1])
    abajo.xaxis.set_major_locator(md.HourLocator(byhour=horas, tz=timezone.utc))
    abajo.xaxis.set_major_formatter(md.DateFormatter("%d/%m\n%H h", tz=timezone.utc))


def guardar(fig, nombre):
    ruta = CARPETA / nombre
    fig.savefig(ruta, dpi=165, facecolor="white")
    plt.close(fig)
    print(ruta)


def figura_rsi(d, fechas):
    muestra = d.loc[fechas]
    fig, (a, b) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.subplots_adjust(left=.068, right=.721, top=.79, bottom=.20, hspace=.38)
    cabecera(fig, "RSI a 45 minutos", "BTC/USDT   ·   192 cierres objetivo de junio de 2026   ·   Pronósticos emitidos tres velas antes")
    curvas(a, b, muestra, [0, 6, 12, 18])
    a.set_title("Nivel de RSI", loc="left")
    b.set_title("Cambio desde el RSI conocido al emitir", loc="left")
    for ax in (a, b):
        ax.set_ylabel("Puntos de RSI")
    b.set_xlabel("Fecha y hora del cierre objetivo (UTC)")
    a.legend(loc="upper left", bbox_to_anchor=(1.035, 1.05))
    b.legend(loc="upper left", bbox_to_anchor=(1.035, 1.05))
    fig.text(.755, .625, "La referencia mantiene el RSI\ndel origen durante 45 minutos.", color=GRIS, fontsize=13)
    mae, ref, mejora, (lo, hi) = metricas(d)
    fig.text(.755, .365, "MAE de junio · 2.874 ventanas", weight="bold", color=AZUL, fontsize=14)
    fig.text(.755, .317, f"Transformer    {numero(mae, 3)} puntos", color=NARANJA, fontsize=14)
    fig.text(.755, .276, f"Persistencia    {numero(ref, 3)} puntos", color=VERDE, fontsize=14)
    fig.text(.068, .082, f"Mejora del MAE en junio   {numero(mejora, signo=True)} %", fontsize=18, weight="bold", color=AZUL)
    fig.text(.068, .036, f"IC del 95 % de la mejora: {numero(lo, signo=True)} a {numero(hi, signo=True)} %   ·   Bootstrap pareado por días UTC", fontsize=12, color=GRIS)
    guardar(fig, "rsi_junio_2026_45m.png")


def figura_macd(series, fechas):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey="row")
    fig.subplots_adjust(left=.066, right=.978, top=.74, bottom=.29, hspace=.28, wspace=.12)
    cabecera(fig, "MACD a 60, 75 y 90 minutos", "BTC/USDT   ·   Los tres plazos se comparan en los mismos 192 cierres objetivo   ·   Junio de 2026")
    for j, minutos in enumerate(PLAZOS["macd"]):
        a, b = axes[:, j]
        d = series[minutos]
        curvas(a, b, d.loc[fechas], [0,12])
        a.axhline(0, color=GRIS, lw=.6, zorder=0)
        a.set_title(f"Emitido {minutos} min antes", color=AZUL, weight="bold")
        mae, ref, mejora, (lo, hi) = metricas(d)
        x = a.get_position().x0
        fig.text(x, .164, f"MAE   {numero(mae, 2)}  /  {numero(ref, 2)} USDT", fontsize=14, weight="bold", color=AZUL)
        fig.text(x, .117, f"Mejora   {numero(mejora, signo=True)} %", fontsize=14, color=AZUL)
        fig.text(x, .078, f"IC 95 %   [{numero(lo, signo=True)}; {numero(hi, signo=True)}] %", fontsize=12, color=GRIS)
    axes[0,0].set_ylabel("Nivel (USDT)")
    axes[1,0].set_ylabel("Corrección (USDT)")
    lineas = axes[0,0].lines[:3]
    fig.legend(lineas, ["Real", "Transformer", "Referencia de cierre constante"], loc="center left",
               bbox_to_anchor=(.06,.830), ncol=3)
    fig.text(.065,.213,"Junio completo   ·   2.874 ventanas   ·   MAE del Transformer / MAE de la referencia",fontsize=14,color=AZUL)
    fig.text(.065,.023,"Eje horizontal en UTC. Abajo: real − referencia (azul) y pronóstico − referencia (naranja). Intervalos por bloques diarios.",fontsize=12,color=GRIS)
    guardar(fig, "macd_junio_2026_60_75_90m.png")


def figura_cruces(g):
    e = g[g.elegible]
    y = e.primer_cruce_velas.between(4, 6)
    verdaderos = e.loc[y]
    fecha_cruce = verdaderos.index + verdaderos.primer_cruce_velas * PASO
    total_unicos = fecha_cruce.nunique()
    fig, (a, b) = plt.subplots(1, 2, figsize=(16,9))
    fig.subplots_adjust(left=.10, right=.95, top=.71, bottom=.32, wspace=.32)
    cabecera(fig, "Avisos de cruce con 60–90 minutos de antelación",
             f"Junio de 2026   ·   {len(e):,} orígenes elegibles   ·   {y.sum()} orígenes positivos   ·   {total_unicos} cruces distintos alcanzables".replace(",","."))
    a.set_title("¿Cuántos avisos acertaron?", loc="left")
    b.set_title("¿Con cuánto tiempo se detectó cada cruce?", loc="left")
    for j,(clave, nombre, color) in enumerate([("red","Transformer",NARANJA), ("logistica","Logística",VERDE)]):
        aviso = e["aviso_"+clave]
        tp = int((aviso & y).sum())
        fp = int(aviso.sum()) - tp
        al = a.barh(j, tp, height=.5, color=color)
        bl = a.barh(j, fp, left=tp, height=.5, color="#dde3e5")
        a.bar_label(al,labels=[str(tp)],label_type="center",color="white",weight="bold")
        a.bar_label(bl,labels=[str(fp)],label_type="center",color=AZUL)
        antelacion = verdaderos.loc[aviso.loc[verdaderos.index], "primer_cruce_velas"] * 15
        # Para cada cruce, cuenta el primer aviso correcto.
        anticipos = antelacion.groupby(fecha_cruce.loc[antelacion.index]).max()
        cuenta = anticipos.value_counts().reindex([60,75,90],fill_value=0)
        barras = b.bar(np.arange(3) + (j-.5)*.32, cuenta, width=.32,color=color,label=nombre)
        b.bar_label(barras,padding=3,color=AZUL)
        ap = precision_promedio(pd.DataFrame({"etiqueta":y.astype(int),"probabilidad":e["p_"+clave]}))
        x = .09 + .46*j
        fig.text(x,.197,f"{nombre}   ·   {len(anticipos)} de {total_unicos} cruces detectados",fontsize=17,weight="bold",color=color)
        fig.text(x,.139,f"{tp} aciertos en {int(aviso.sum())} avisos   ·   Precisión {numero(100*tp/aviso.sum(),1)} %   ·   AP {numero(ap,3)}",fontsize=13,color=AZUL)
    a.set(yticks=[0,1],yticklabels=["Transformer","Logística"],xlabel="Avisos emitidos (color = acierto, gris = falso aviso)")
    a.invert_yaxis()
    b.set(xticks=[0,1,2],xticklabels=["60 min","75 min","90 min"],ylabel="Cruces distintos",xlabel="Antelación del primer aviso correcto")
    b.set_ylim(0,b.get_ylim()[1]+1)
    b.legend()
    fig.text(.067,.048,"Evento: primer cruce alcista simple en las velas 4 a 6 desde el origen. Un cruce más temprano invalida ese aviso.",fontsize=13,color=GRIS)
    guardar(fig, "cruces_junio_2026_60_90m.png")


def main():
    plt.rcParams.update({"font.size":12, "axes.titlesize":15, "axes.titleweight":"bold",
                         "axes.titlecolor":AZUL, "axes.spines.right":False, "axes.spines.top":False,
                         "legend.frameon":False, "axes.labelcolor":AZUL, "text.color":AZUL})
    g, series = leer()
    fechas = series[45].index
    for d in series.values():
        fechas = fechas.intersection(d.index)
    fechas = fechas.sort_values()[:192]
    assert len(fechas) == 192 and fechas[-1]-fechas[0] == 191*PASO
    figura_rsi(series[45], fechas)
    figura_macd(series, fechas)
    figura_cruces(g)
    print("Primer y último cierre dibujado (UTC):", fechas[0], fechas[-1])


if __name__ == "__main__":
    main()
