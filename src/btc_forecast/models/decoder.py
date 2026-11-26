"""Decoder causal que devuelve una representación de 64 componentes por vela."""

import torch
from torch import nn
from torch.nn import functional as F


def aplicar_rope(x, coseno, seno):
    """Rota cada par de componentes contiguas según la posición de la vela."""
    pares, impares = x[..., 0::2], x[..., 1::2]
    return torch.stack((pares * coseno - impares * seno,
                        pares * seno + impares * coseno), dim=-1).flatten(-2)


class AtencionCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.cabezas = 4
        self.dim_cabeza = 16
        self.qkv = nn.Linear(64, 3 * 64)
        self.salida = nn.Linear(64, 64)
        self.dropout = 0.1
        exponentes = torch.arange(0, self.dim_cabeza, 2, dtype=torch.float32) / self.dim_cabeza
        self.register_buffer("frecuencias", 10000.0 ** (-exponentes), persistent=False)

    def proyectar_qkv(self, x):
        lote, pasos, _ = x.shape
        qkv = self.qkv(x).reshape(lote, pasos, 3, self.cabezas, self.dim_cabeza)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        posiciones = torch.arange(pasos, device=x.device, dtype=self.frecuencias.dtype)
        angulos = torch.outer(posiciones, self.frecuencias)
        coseno, seno = angulos.cos().to(q.dtype), angulos.sin().to(q.dtype)
        return aplicar_rope(q, coseno, seno), aplicar_rope(k, coseno, seno), v

    def forward(self, x):
        q, k, v = self.proyectar_qkv(x)
        contexto = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.salida(contexto.transpose(1, 2).reshape_as(x))


class BloqueDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_atencion = nn.LayerNorm(64)
        self.atencion = AtencionCausal()
        self.dropout = nn.Dropout(0.1)
        self.norm_ffn = nn.LayerNorm(64)
        self.ffn = nn.Sequential(
            nn.Linear(64, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64)
        )

    def forward(self, x):
        x = x + self.dropout(self.atencion(self.norm_atencion(x)))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


class Decoder(nn.Module):
    def __init__(self, numero_variables: int):
        super().__init__()
        self.proyeccion = nn.Linear(numero_variables, 64)
        self.bloques = nn.Sequential(BloqueDecoder(), BloqueDecoder())
        self.norm_final = nn.LayerNorm(64)

    def forward(self, x):
        """Transforma [lote, posiciones, variables] en [lote, posiciones, 64]."""
        return self.norm_final(self.bloques(self.proyeccion(x)))


def main():
    from itertools import islice
    from pathlib import Path

    import pandas as pd
    from btc_forecast.data.dataset import construir_muestras

    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    muestras = list(islice(construir_muestras(pd.read_parquet(ruta)), 2))
    if len(muestras) != 2:
        raise ValueError("Se requieren dos ventanas completas para la demostración.")
    x = torch.stack([torch.from_numpy(m.X.to_numpy(dtype="float32", copy=True)) for m in muestras])
    torch.manual_seed(42)
    modelo = Decoder(numero_variables=x.shape[-1]).eval()
    with torch.inference_mode():
        salida = modelo(x)
        proyectada = modelo.proyeccion(x)
        bloque = modelo.bloques[0]
        q, k, v = bloque.atencion.proyectar_qkv(bloque.norm_atencion(proyectada))
        print("Entrada:", tuple(x.shape))
        print("Proyección:", tuple(proyectada.shape))
        print("Q, K y V de una cabeza:", [tuple(t[:, 0].shape) for t in (q, k, v)])
        print("Salida:", tuple(salida.shape))

        mitad = x.shape[1] // 2
        alterada = x.clone()
        alterada[:, mitad:] = torch.randn_like(alterada[:, mitad:])
        otra_salida = modelo(alterada)
        torch.testing.assert_close(salida[:, :mitad], otra_salida[:, :mitad], rtol=0, atol=0)
        assert not torch.equal(salida[:, mitad:], otra_salida[:, mitad:])
        print(f"Causalidad verificada: las primeras {mitad} representaciones permanecen iguales.")


if __name__ == "__main__":
    main()
