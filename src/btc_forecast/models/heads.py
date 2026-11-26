"""Cabezas para pronosticar el retorno con Student-t o los cambios de RSI y MACD."""

import torch
from torch import nn
from torch.nn import functional as F


class CabezaStudentT(nn.Module):
    """Ambos modelos pueden configurar modelo.cabeza.mu_cero antes del forward."""

    def __init__(self, *, mu_cero=False):
        super().__init__()
        self.mu_cero = mu_cero
        self.proyeccion = nn.Linear(64, 2)
        self.raw_nu = nn.Parameter(torch.zeros(()))

    def forward(self, ultimo_token):
        """Devuelve dos vectores [batch] y un escalar compartido por todo el lote."""
        if ultimo_token.ndim != 2 or ultimo_token.shape[-1] != 64:
            raise ValueError("Se espera el último token con forma [batch, 64].")
        mu, raw_s = self.proyeccion(ultimo_token).unbind(dim=-1)
        if self.mu_cero:
            mu = torch.zeros_like(mu)
        s = F.softplus(raw_s) + 1e-6
        # Evita que el redondeo deje ν exactamente en 2.
        paso = 2 * torch.finfo(self.raw_nu.dtype).eps
        nu = 2 + F.softplus(self.raw_nu).clamp_min(paso)
        return mu, s, nu


class CabezaRegresionDirecta(nn.Module):
    """Para h = 1, ..., H, predice RSI[t+h] - RSI[t] y (MACD[t+h] - MACD[t]) / C[t]."""

    def __init__(self, horizontes: int = 1):
        super().__init__()
        if type(horizontes) is not int or horizontes < 1:
            raise ValueError("El número de horizontes debe ser un entero positivo.")
        self.horizontes = horizontes
        self.proyeccion = nn.Linear(64, 2 * horizontes)

    def forward(self, ultimo_token):
        """De [batch, 64] a [batch, H, 2], con el par del horizonte h en la posición h-1."""
        if ultimo_token.ndim != 2 or ultimo_token.shape[-1] != self.proyeccion.in_features:
            raise ValueError("Se espera el último token con forma [batch, 64].")
        return self.proyeccion(ultimo_token).reshape(ultimo_token.shape[0], self.horizontes, 2)


def demostrar_mu_cero():
    from btc_forecast.models.baselines import StudentTLineal
    from btc_forecast.training.losses import nll_student_t
    from btc_forecast.training.trainer import ModeloStudentT

    torch.manual_seed(42)
    x = torch.randn(2, 96, 8)
    y = torch.tensor([0.01, -0.02])
    for clase in (StudentTLineal, ModeloStudentT):
        modelo = clase(x.shape[-1]).eval()  # Desactiva dropout para comparar la misma ventana.
        modelo.cabeza.mu_cero = True
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
        mu, s, nu = modelo(x)
        s.retain_grad()
        nll_student_t(y, mu, s, nu).backward()
        assert torch.isfinite(s.grad).all() and (s.grad != 0).all()
        optimizador.step()
        with torch.no_grad():
            mu_nueva, s_nueva, _ = modelo(x)
        assert (mu == 0).all() and (mu_nueva == 0).all()
        print(f"{clase.__name__}: mu {mu.tolist()} -> {mu_nueva.tolist()} | "
              f"dNLL/ds {s.grad.tolist()} | s[0] {s[0].item():.6f} -> {s_nueva[0].item():.6f}")


def main(horizontes=None):
    from itertools import islice
    from pathlib import Path

    import pandas as pd
    from btc_forecast.data.dataset import construir_muestras
    from btc_forecast.models.decoder import Decoder

    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    muestras = list(islice(construir_muestras(pd.read_parquet(ruta)), 2))
    if len(muestras) != 2:
        raise ValueError("El ejemplo requiere dos ventanas completas.")
    x = torch.stack([torch.from_numpy(m.X.to_numpy(dtype="float32", copy=True)) for m in muestras])
    torch.manual_seed(42)
    decoder = Decoder(numero_variables=x.shape[-1]).eval()
    cabeza = CabezaStudentT() if horizontes is None else CabezaRegresionDirecta(horizontes)
    cabeza.eval()
    with torch.inference_mode():
        tokens = decoder(x)
        ultimo = tokens[:, -1, :]
        salida = cabeza(ultimo)
    print("Red sin entrenar. Semilla: 42")
    print(f"Ventanas: {tuple(x.shape)} | Decoder: {tuple(tokens.shape)} | Último token: {tuple(ultimo.shape)}")
    if horizontes is not None:
        print(f"Regresión directa [ventana, índice de horizonte, variable]: {tuple(salida.shape)}")
        print("Posición | Origen UTC | Horizonte | Cambio RSI (puntos) | Cambio MACD / cierre de origen")
        for i, muestra in enumerate(muestras):
            for j, (delta_rsi, delta_macd_sobre_cierre) in enumerate(salida[i].tolist()):
                print(f"[{i}, {j}, :] | {muestra.fecha_prediccion.isoformat()} | t+{j+1} | "
                      f"{delta_rsi:.8f} | {delta_macd_sobre_cierre:.8f}")
        return
    mu, s, nu = salida
    print(f"mu: {tuple(mu.shape)} | s: {tuple(s.shape)} | nu: {tuple(nu.shape)} (compartido)")
    print("Fecha de predicción UTC | mu | s")
    for i, muestra in enumerate(muestras):
        print(f"{muestra.fecha_prediccion.isoformat()} | {mu[i].item():.8f} | {s[i].item():.8f}")
    print(f"nu = {nu.item():.8f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--mu-cero", action="store_true", help="Prueba una actualización con localización fija.")
    modo.add_argument("--regresion-directa", nargs="?", const=1, type=int, metavar="H",
                      help="Demuestra la cabeza directa con H horizontes (por defecto, uno).")
    args = parser.parse_args()
    if args.mu_cero:
        demostrar_mu_cero()
    else:
        main(args.regresion_directa)
