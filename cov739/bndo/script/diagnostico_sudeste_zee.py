"""
Diagnóstico pontual: investiga faixa por faixa o valor de d(c)/d(S) na
estação Sudeste/ZEE, para descobrir se o pico extremo (-3561) vem de
uma única faixa isolada (provável artefato/salinidade quase constante
localmente) ou de um padrão mais espalhado (possível problema real).

Uso:
    python3 diagnostico_sudeste_zee.py
"""

import numpy as np
import processar_estacoes_ctd as ctd

df = ctd.carregar_arquivo("../dados_ctd_bndo/Oceano Sudeste")
estacao = ctd.processar_estacao_completa(df, -22.27, -39.19, "2002-10-23", rotulo_log="Sudeste/ZEE")

x_bin, deriv = ctd._derivada_por_bins_por_profundidade(
    estacao["Profundidade [m]"].to_numpy(),
    estacao["Salinidade [psu]"].to_numpy(),
    estacao["Velocidade do som [m/s]"].to_numpy(),
)

print("Número de faixas:", len(deriv))
if np.any(np.isfinite(deriv)):
    idx_min = int(np.nanargmin(deriv))
    print("Índice da faixa com o valor mais extremo (mínimo):", idx_min)
    print("Valor nessa faixa:", deriv[idx_min])
    print("Salinidade média nessa faixa:", x_bin[idx_min])

print("\nTodas as faixas (índice: Salinidade | derivada):")
for i in range(len(deriv)):
    marca = "  <<< EXTREMO" if np.isfinite(deriv[i]) and abs(deriv[i]) > 500 else ""
    print(f"  faixa {i:2d}: S={x_bin[i]:.3f}  deriv={deriv[i]:.2f}{marca}")
