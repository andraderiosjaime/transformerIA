"""Entrenamiento con cabeza Student-t o regresión directa de RSI y MACD a uno o seis horizontes."""

import argparse
from copy import deepcopy
from math import isclose, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from btc_forecast.data.dataset import ObjetivosDirectos, construir_muestras
from btc_forecast.data.ingest import leer_rango_local
from btc_forecast.data.splits import corte_con_validacion_interna, corte_cronologico, folds_mensuales_expansivos
from btc_forecast.features.transforms import ajustar_escalado, aplicar_escalado
from btc_forecast.models.baselines import StudentTLineal
from btc_forecast.models.decoder import Decoder
from btc_forecast.models.heads import CabezaRegresionDirecta, CabezaStudentT
from btc_forecast.training.losses import nll_student_t


class _ModeloConDecoder(nn.Module):
    def __init__(self, numero_variables, tipo_cabeza, **opciones):
        super().__init__()
        self.decoder = Decoder(numero_variables)
        self.cabeza = tipo_cabeza(**opciones)

    def forward(self, x):
        return self.cabeza(self.decoder(x)[:, -1, :])


class ModeloStudentT(_ModeloConDecoder):
    def __init__(self, numero_variables):
        super().__init__(numero_variables, CabezaStudentT)


class ModeloRegresionDirecta(_ModeloConDecoder):
    """Devuelve los cambios estandarizados con forma [lote, H, 2]."""

    def __init__(self, numero_variables, horizontes=1):
        super().__init__(numero_variables, CabezaRegresionDirecta, horizontes=horizontes)
        self.escalado_objetivos = None


def crear_modelo(numero_variables, arquitectura="transformer", mu_cero=False, *, horizontes=1):
    if horizontes not in (1, 6) or (horizontes != 1 and arquitectura != "directa"):
        raise ValueError("H=6 requiere regresión directa; el otro horizonte admitido es H=1.")
    if arquitectura == "directa":
        if mu_cero:
            raise ValueError("mu_cero solo se aplica a las cabezas Student-t.")
        return ModeloRegresionDirecta(numero_variables, horizontes)
    clases = {"transformer": ModeloStudentT, "lineal": StudentTLineal}
    if arquitectura not in clases or not isinstance(mu_cero, bool):
        raise ValueError("Se requiere arquitectura transformer, lineal o directa y mu_cero booleano.")
    modelo = clases[arquitectura](numero_variables)
    modelo.cabeza.mu_cero = mu_cero
    return modelo


def _perdida_lote(modelo, x, y):
    salida = modelo(x)
    if isinstance(modelo, ModeloRegresionDirecta):
        if salida.shape != y.shape:
            raise ValueError("Predicciones y objetivos directos deben tener la misma forma [lote, H, 2].")
        return nn.functional.huber_loss(salida, y, reduction="mean", delta=1.0)
    return nll_student_t(y, *salida)


def paso_entrenamiento(modelo, optimizador, x, y):
    """Actualiza los pesos una vez y devuelve la pérdida del lote previa al paso."""
    modelo.train()
    optimizador.zero_grad(set_to_none=True)
    perdida = _perdida_lote(modelo, x, y)
    if not torch.isfinite(perdida):
        raise ValueError("Pérdida no finita antes de la actualización.")
    perdida.backward()
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizador.step()
    return perdida.item()


@torch.inference_mode()
def evaluar_lote(modelo, x, y):
    modelo.eval()
    mu, s, nu = modelo(x)
    perdida = nll_student_t(y, mu, s, nu)
    return perdida.item(), mu[0].item(), s[0].item(), nu.item()


def _columnas_objetivos(horizontes):
    if horizontes == 1:
        return pd.Index(ObjetivosDirectos._fields)
    return pd.MultiIndex.from_product([range(1, horizontes + 1), ObjetivosDirectos._fields],
                                      names=["h", "indicador"])


def _objetivos_directos(muestras, horizontes):
    filas = []
    plazo = pd.Timedelta(hours=horizontes)
    for m in muestras:
        pares = m.objetivos_directos
        if isinstance(pares, ObjetivosDirectos):
            pares = (pares,)
        if pares is None or len(pares) != horizontes or m.fin_etiqueta != m.fecha_prediccion + plazo:
            raise ValueError("Los objetivos directos y el fin de etiqueta deben corresponder a H.")
        filas.append([valor for par in pares for valor in par])
    valores = pd.DataFrame(filas, columns=_columnas_objetivos(horizontes), dtype=float)
    if valores.isna().any().any() or valores.isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("Los dos objetivos directos deben ser finitos.")
    return valores


def ajustar_escalado_objetivos(ajuste, horizontes=1):
    """Media y desviación poblacional por horizonte e indicador. Escala 1 para constantes."""
    valores = _objetivos_directos(ajuste, horizontes)
    constantes = valores.min().eq(valores.max())
    return pd.DataFrame({"media": valores.mean().mask(constantes, valores.iloc[0]),
                         "escala": valores.std(ddof=0).mask(constantes, 1.0)})


def _datos_escalados(muestras, parametros, escalado_objetivos=None):
    ventanas = [torch.from_numpy(aplicar_escalado(m.X, parametros).to_numpy(dtype="float32", copy=True))
                for m in muestras]
    if escalado_objetivos is None:
        y = torch.tensor([m.y for m in muestras], dtype=torch.float32)
    else:
        horizontes = len(escalado_objetivos) // len(ObjetivosDirectos._fields)
        z = (_objetivos_directos(muestras, horizontes) - escalado_objetivos["media"]) / escalado_objetivos["escala"]
        y = torch.from_numpy(z.to_numpy(dtype="float32", copy=True)).reshape(len(muestras), horizontes, 2)
    return TensorDataset(torch.stack(ventanas), y)


@torch.inference_mode()
def perdida_validacion(modelo, lotes, dispositivo):
    modelo.eval()
    suma = 0.0
    for x, y in lotes:
        perdida = _perdida_lote(modelo, x.to(dispositivo), y.to(dispositivo))
        suma += perdida.item() * len(y)
    media = suma / len(lotes.dataset)
    if not isfinite(media):
        raise ValueError("Pérdida interna no finita.")
    return media


nll_validacion = perdida_validacion  # Nombre utilizado por el modo Student-t.


def entrenar(ajuste, interna, *, epocas=10, paciencia=2, tamano_lote=16, semilla=42, dispositivo="cpu",
             arquitectura="transformer", mu_cero=False, horizontes=1):
    """Recibe solo los dos tramos; devuelve modelo restaurado, escalado, historial y mejor época."""
    if not ajuste or not interna or min(epocas, paciencia, tamano_lote) < 1:
        raise ValueError("Se requieren dos tramos con datos y parámetros de entrenamiento positivos.")
    if max(m.fecha_objetivo for m in ajuste) > min(m.fecha_prediccion for m in interna):
        raise ValueError("Una etiqueta de ajuste cruza el inicio de la validación interna.")
    torch.manual_seed(semilla)
    escalado = ajustar_escalado(m.X for m in ajuste)
    modelo = crear_modelo(ajuste[0].X.shape[1], arquitectura, mu_cero, horizontes=horizontes).to(dispositivo)
    escalado_y = None
    if isinstance(modelo, ModeloRegresionDirecta):
        escalado_y = ajustar_escalado_objetivos(ajuste, horizontes)
        modelo.escalado_objetivos = escalado_y
    datos = _datos_escalados(ajuste, escalado, escalado_y)
    lotes_ajuste = DataLoader(datos, batch_size=tamano_lote, shuffle=True,
                             generator=torch.Generator().manual_seed(semilla))
    lotes_internos = DataLoader(_datos_escalados(interna, escalado, escalado_y), batch_size=tamano_lote)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
    minimo_interno = float("inf")
    historial = []
    for epoca in range(1, epocas + 1):
        suma = 0.0
        for x, y in lotes_ajuste:
            perdida = paso_entrenamiento(modelo, optimizador, x.to(dispositivo), y.to(dispositivo))
            suma += perdida * len(y)
        perdida_interna = perdida_validacion(modelo, lotes_internos, dispositivo)
        historial.append((epoca, suma / len(ajuste), perdida_interna))
        if perdida_interna < minimo_interno:
            minimo_interno = perdida_interna
            mejor_epoca = epoca
            mejores_pesos = deepcopy(modelo.state_dict())
        elif epoca - mejor_epoca >= paciencia:
            break

    modelo.load_state_dict(mejores_pesos)
    restaurada = perdida_validacion(modelo, lotes_internos, dispositivo)
    if not isclose(restaurada, minimo_interno, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("La pérdida del modelo restaurado difiere del mínimo interno.")
    return modelo, escalado, historial, mejor_epoca


def _arquitectura(modelo):
    if isinstance(modelo, StudentTLineal):
        return {"tipo": "lineal", "variables": modelo.proyeccion.in_features,
                "dimension": modelo.proyeccion.out_features}
    decoder = modelo.decoder
    bloque = decoder.bloques[0]
    arquitectura = {
        "tipo": "Decoder causal RoPE pre-LayerNorm con normalización final",
        "variables": decoder.proyeccion.in_features, "dimension": decoder.proyeccion.out_features,
        "bloques": len(decoder.bloques), "cabezas": bloque.atencion.cabezas,
        "ffn": bloque.ffn[0].out_features, "activacion": type(bloque.ffn[1]).__name__,
        "dropout": bloque.dropout.p, "rope_base": 10000,
        "salida": "Student-t sobre último token, escala softplus y nu compartido",
    }
    if isinstance(modelo, ModeloRegresionDirecta):
        arquitectura.update(salida="Cambios estandarizados de RSI y MACD sobre cierre de origen",
                            horizontes=modelo.cabeza.horizontes, objetivos=list(ObjetivosDirectos._fields))
    return arquitectura


def guardar_checkpoint(ruta, modelo, escalado, *, ventana, semilla, cortes, epoca):
    """Guarda el modelo restaurado al mínimo interno y las escalas necesarias para inferencia."""
    datos = {
        "pesos": modelo.state_dict(), "arquitectura": _arquitectura(modelo),
        "variables": escalado.index.tolist(), "escalado": escalado.to_dict(orient="list"),
        "ventana": ventana, "semilla": semilla, "epoca": epoca,
        "cortes": {nombre: fecha.isoformat() for nombre, fecha in cortes.items()},
    }
    if isinstance(modelo, ModeloRegresionDirecta):
        if modelo.escalado_objetivos is None:
            raise ValueError("Falta el escalado de objetivos del tramo de ajuste.")
        datos.update(version=3 if modelo.cabeza.horizontes == 1 else 4, modelo="directa", perdida="Huber", delta_huber=1.0,
                     escalado_objetivos=modelo.escalado_objetivos.to_dict(orient="index"))
    else:
        datos.update(version=2, modelo="lineal" if isinstance(modelo, StudentTLineal) else "transformer",
                     mu_cero=modelo.cabeza.mu_cero)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=ruta.parent, delete=False) as archivo:
        temporal = Path(archivo.name)
    try:
        torch.save(datos, temporal)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def cargar_checkpoint(ruta):
    """Reconstruye el modelo en CPU para inferencia y devuelve su escalado y metadatos."""
    datos = torch.load(ruta, map_location="cpu", weights_only=True)
    if datos["version"] == 1:
        datos.update(modelo="transformer", mu_cero=False)
    elif datos["version"] not in (2, 3, 4):
        raise ValueError("Versión de checkpoint incompatible.")
    horizontes = datos["arquitectura"].get("horizontes", 1)
    modelo = crear_modelo(len(datos["variables"]), datos["modelo"], datos.get("mu_cero", False), horizontes=horizontes)
    if datos["arquitectura"] != _arquitectura(modelo):
        raise ValueError("La arquitectura del checkpoint difiere del modelo disponible.")
    modelo.load_state_dict(datos.pop("pesos"), strict=True)
    if isinstance(modelo, ModeloRegresionDirecta):
        objetivos = pd.DataFrame.from_dict(datos["escalado_objetivos"], orient="index")
        columnas = _columnas_objetivos(horizontes)
        modelo.escalado_objetivos = objetivos.loc[columnas, ["media", "escala"]].rename_axis(index=columnas.names)
    escalado = pd.DataFrame(datos.pop("escalado"), index=datos["variables"])
    return modelo.eval(), escalado, datos


@torch.inference_mode()
def _pronosticar_ventana(modelo, ventana, escalado):
    x = aplicar_escalado(ventana, escalado).to_numpy(dtype="float32", copy=True)
    return modelo(torch.from_numpy(x).unsqueeze(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispositivo", default="cpu", help="Por ejemplo cpu, cuda o cuda:0.")
    parser.add_argument("--modo", choices=("lote-fijo", "epocas", "regresion-directa"), default="lote-fijo")
    parser.add_argument("--horizontes", type=int, choices=(1, 6), default=1)
    parser.add_argument("--fold-abril", action="store_true", help="Usa enero a marzo como historia y evalúa abril.")
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--paciencia", type=int, default=2)
    parser.add_argument("--tamano-lote", type=int, default=16)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--arquitectura", choices=("transformer", "lineal"), default="transformer")
    parser.add_argument("--mu-cero", action="store_true", help="Fija la localización de Student-t en cero.")
    args = parser.parse_args()
    directa = args.modo == "regresion-directa"
    if args.horizontes != 1 and not directa:
        parser.error("--horizontes 6 requiere --modo regresion-directa.")
    if args.fold_abril and not directa:
        parser.error("--fold-abril corresponde al experimento de regresión directa.")
    if directa and (args.arquitectura != "transformer" or args.mu_cero):
        parser.error("La regresión directa utiliza el decoder y su cabeza de dos objetivos.")
    dispositivo = torch.device(args.dispositivo)
    if dispositivo.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible en este entorno.")

    torch.manual_seed(args.semilla)
    corte = pd.Timestamp("2025-01-20T11:59:59.999Z")
    if args.fold_abril:
        inicio = pd.Timestamp("2025-01-01", tz="UTC")
        fin = pd.Timestamp("2025-05-01", tz="UTC")
        velas = leer_rango_local(inicio, fin)
    else:
        velas = pd.read_parquet("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    muestras = construir_muestras(velas, regresion_directa=directa, horizontes=args.horizontes)
    if args.modo != "lote-fijo":
        if args.fold_abril:
            corte, ajuste, interna, evaluacion = next(folds_mensuales_expansivos(
                muestras, inicio, meses_historia=3, meses_evaluacion=1))
            for nombre, tramo, limite in (("Ajuste", ajuste, interna[0].fecha_prediccion),
                                          ("Validación interna", interna, corte)):
                ultimo = max(m.fin_etiqueta for m in tramo)
                if ultimo > limite:
                    raise ValueError(f"Hay etiquetas de {nombre} posteriores a {limite}.")
                print(f"{nombre}: última etiqueta {ultimo} <= frontera {limite}", flush=True)
            if evaluacion[0].fecha_prediccion < corte or max(m.fin_etiqueta for m in evaluacion) >= fin:
                raise ValueError("Hay predicciones o etiquetas de evaluación fuera de abril.")
        else:
            ajuste, interna, evaluacion = corte_con_validacion_interna(muestras, corte)
        modelo, escalado, historial, mejor = entrenar(
            ajuste, interna, epocas=args.epocas, paciencia=args.paciencia, tamano_lote=args.tamano_lote,
            semilla=args.semilla, dispositivo=dispositivo,
            arquitectura="directa" if directa else args.arquitectura, mu_cero=args.mu_cero, horizontes=args.horizontes,
        )
        print(f"Dispositivo: {next(modelo.parameters()).device} | Semilla: {args.semilla} | Corte UTC: {corte}")
        print(f"Ajuste: {len(ajuste)} | Validación interna: {len(interna)} | Evaluación reservada: {len(evaluacion)}")
        metrica = "Huber" if directa else "NLL"
        print(f"Época | {metrica} ajuste | {metrica} interna")
        for epoca, perdida_ajuste, perdida_interna in historial:
            print(f"{epoca:5} | {perdida_ajuste:.8f} | {perdida_interna:.8f}")
        print(f"Pesos restaurados y verificados: época {mejor}")
        muestra = evaluacion[0] if args.fold_abril else interna[0]
        if args.fold_abril:
            datos_abril = _datos_escalados(evaluacion, escalado, modelo.escalado_objetivos)
            huber_abril = perdida_validacion(modelo, DataLoader(datos_abril, batch_size=args.tamano_lote), dispositivo)
            print(f"Huber en abril con los pesos restaurados: {huber_abril:.8f}")
        modelo.cpu().eval()
        antes = _pronosticar_ventana(modelo, muestra.X, escalado)
        modo_mu = "mu_cero" if args.mu_cero else "mu_aprendida"
        prefijo = "mejor_student_t"
        if directa:
            prefijo = "mejor_regresion_directa"
        elif args.arquitectura != "transformer" or args.mu_cero:
            prefijo = f"mejor_{args.arquitectura}_{modo_mu}"
        if args.horizontes != 1:
            prefijo += f"_h{args.horizontes}"
        destino = Path("artifacts") / f"{prefijo}_{corte:%Y%m%d}_semilla{args.semilla}.pt"
        guardar_checkpoint(destino, modelo, escalado, ventana=len(muestra.X), semilla=args.semilla,
                           cortes={"inicio_ajuste": ajuste[0].fecha_prediccion,
                                   "inicio_validacion_interna": interna[0].fecha_prediccion,
                                   "inicio_evaluacion": corte}, epoca=mejor)
        cargado, escalado_cpu, metadatos = cargar_checkpoint(destino)
        pd.testing.assert_frame_equal(escalado, escalado_cpu, check_exact=True)
        assert metadatos["ventana"] == len(muestra.X) and metadatos["epoca"] == mejor
        despues = _pronosticar_ventana(cargado, muestra.X, escalado_cpu)
        print(f"Checkpoint: {destino} | Ventana hasta {muestra.fecha_prediccion}")
        if directa:
            pd.testing.assert_frame_equal(modelo.escalado_objetivos, cargado.escalado_objetivos, check_exact=True)
            torch.testing.assert_close(antes, despues, rtol=0, atol=0)
            print(f"Predicción para una ventana {tuple(despues.shape)}, idéntica tras recargar en CPU. Escalado de objetivos:")
            print(cargado.escalado_objetivos.to_string(float_format=lambda v: f"{v:.10g}"))
        else:
            print("Parámetro | Antes | Después (CPU) | Diferencia absoluta")
            for nombre, a, b in zip(("mu", "s", "nu"), antes, despues):
                print(f"{nombre} | {a.item():.9f} | {b.item():.9f} | {(b-a).abs().item():.9g}")
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        print("Repetir desde la raíz del proyecto en PowerShell:")
        print(r".\.venv\Scripts\python.exe -B -m btc_forecast.training.trainer "
              f"--modo {args.modo} "
              f"--epocas {args.epocas} --paciencia {args.paciencia} --tamano-lote {args.tamano_lote} "
              f"--semilla {args.semilla} --dispositivo {dispositivo} --arquitectura {args.arquitectura}"
              + (" --mu-cero" if args.mu_cero else "")
              + (f" --horizontes {args.horizontes}" if directa else "")
              + (" --fold-abril" if args.fold_abril else ""))
        return
    entrenamiento, _ = corte_cronologico(muestras, corte)
    lote = entrenamiento[:8]
    if len(lote) != 8 or any(m.fecha_objetivo > corte for m in lote):
        raise ValueError("Se requieren ocho muestras con objetivo conocido al corte.")
    x = torch.stack([torch.from_numpy(m.X.to_numpy(dtype="float32", copy=True)) for m in lote]).to(dispositivo)
    y = torch.tensor([m.y for m in lote], dtype=torch.float32, device=dispositivo)
    modelo = crear_modelo(x.shape[-1], args.arquitectura, args.mu_cero).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4)

    iniciales = [p.detach().clone() for p in modelo.parameters()]
    antes = evaluar_lote(modelo, x, y)
    pasos = 12
    for _ in range(pasos):
        paso_entrenamiento(modelo, optimizador, x, y)
    despues = evaluar_lote(modelo, x, y)
    if not all(isfinite(valor) for valor in (*antes, *despues)):
        raise ValueError("Resultado no finito al evaluar el lote.")
    cambiados = sum(not torch.equal(a, p) for a, p in zip(iniciales, modelo.parameters()))
    print(f"Dispositivo: {dispositivo} | semilla: {args.semilla} | pasos: {pasos} | lote: {tuple(x.shape)}")
    print(f"Corte UTC: {corte} | último objetivo del lote: {lote[-1].fecha_objetivo}")
    print(f"Primera muestra: {lote[0].fecha_prediccion} | objetivo conocido: {lote[0].fecha_objetivo}")
    print("Evaluación | NLL media del lote | mu[0] | s[0] | nu | retorno observado[0]")
    for etapa, valores in [("Antes", antes), ("Después", despues)]:
        print(etapa, *(f"{v:.8f}" for v in (*valores, y[0].item())), sep=" | ")
    print(f"Tensores de parámetros modificados: {cambiados}/{len(iniciales)}. Pérdidas finitas en todos los pasos.")


if __name__ == "__main__":
    main()
