"""Pérdida distribucional para el retorno logarítmico de la próxima vela."""

import torch
from torch.distributions import StudentT


def nll_student_t(objetivo, mu, s, nu):
    """Media de -log p(y) con y, mu y s de forma [batch] y nu escalar compartido."""
    if objetivo.ndim != 1 or objetivo.numel() == 0:
        raise ValueError("El objetivo debe tener forma [batch] con al menos una muestra.")
    if mu.shape != objetivo.shape or s.shape != objetivo.shape:
        raise ValueError("mu y s deben tener la misma forma [batch] que el objetivo.")
    if nu.ndim != 0:
        raise ValueError("nu debe ser un escalar compartido por el lote.")
    if not all(torch.isfinite(x).all() for x in (objetivo, mu, s, nu)):
        raise ValueError("Los retornos y los parámetros deben ser finitos.")
    if (s <= 0).any() or nu <= 2:
        raise ValueError("La cabeza requiere s > 0 y nu > 2.")
    distribucion = StudentT(df=nu, loc=mu, scale=s)
    return -distribucion.log_prob(objetivo).mean()


def main():
    objetivo = torch.tensor([0.01], dtype=torch.float64)
    perdidas = []
    print("Retorno observado = 0.010 | escala = 0.020 | nu = 5")
    for nombre, centro in [("Cercano", 0.012), ("Lejano", 0.060)]:
        mu = torch.tensor([centro], dtype=objetivo.dtype, requires_grad=True)
        s = torch.tensor([0.02], dtype=objetivo.dtype, requires_grad=True)
        nu = torch.tensor(5.0, dtype=objetivo.dtype, requires_grad=True)
        perdida = nll_student_t(objetivo, mu, s, nu)
        perdida.backward()
        gradientes = [p.grad for p in (mu, s, nu)]
        assert all(g is not None and torch.isfinite(g).all() for g in gradientes)
        print(f"{nombre}: mu={centro:.3f}, NLL={perdida.item():.8f}, "
              f"gradientes (mu, s, nu) = {tuple(round(g.item(), 8) for g in gradientes)}")
        perdidas.append(perdida.item())
    assert perdidas[0] < perdidas[1]

    y = objetivo.repeat(2)
    centros = torch.tensor([0.012, 0.060], dtype=y.dtype)
    escalas = torch.full_like(y, 0.02)
    media = nll_student_t(y, centros, escalas, nu.detach())
    torch.testing.assert_close(media, y.new_tensor(sum(perdidas) / 2))
    try:
        nll_student_t(y[:, None], centros, escalas, nu.detach())
    except ValueError as error:
        print(f"Objetivo [2, 1] rechazado: {error}")
    else:
        raise AssertionError("La pérdida aceptó formas incompatibles.")


if __name__ == "__main__":
    main()
