"""
Gera um RESUMO NUMÉRICO (sem nenhum gráfico) dos resultados já
calculados pelo pipeline principal — para você poder colar como texto
no chat, sem precisar subir imagem.

Cobre:
  1) Estatísticas da comparação de fórmulas (Mackenzie x dP/dρ x gás
     ideal) por região: diferença média/máxima entre Mackenzie e dP/dρ.
  2) Estatísticas de cada uma das 5 derivadas propriedade x propriedade
     (c×T, S×T, σ×T, c×S, σ×S), por região e por zona: mín, máx, média,
     desvio padrão, e quantos pontos ficaram como NaN (indefinidos).

Requisitos: os mesmos de processar_estacoes_ctd.py (pandas, seawater)
Uso:
    python3 resumo_numerico.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import processar_estacoes_ctd as ctd


def resumir_serie(nome: str, valores: np.ndarray) -> str:
    """Retorna uma linha de texto resumindo min/média/máx/desvio de um array,
    ignorando NaN, e contando quantos NaN existem."""
    validos = valores[np.isfinite(valores)]
    n_nan = len(valores) - len(validos)
    if len(validos) == 0:
        return f"    {nome}: sem valores válidos ({n_nan} NaN de {len(valores)})"
    return (f"    {nome}: min={validos.min():.3f}  méd={validos.mean():.3f}  "
            f"máx={validos.max():.3f}  desvio={validos.std():.3f}  "
            f"(NaN: {n_nan}/{len(valores)})")


def main():
    PASTA_SAIDA = ctd.PASTA_SAIDA
    estacoes_processadas = {regiao: {} for regiao in ctd.ARQUIVOS}

    print("Carregando e processando estações (mesma lógica do pipeline principal)...\n")
    for regiao, caminho in ctd.ARQUIVOS.items():
        caminho_path = Path(caminho)
        if not caminho_path.exists():
            print(f"[ERRO] Arquivo não encontrado: {caminho_path.resolve()}")
            continue
        df = ctd.carregar_arquivo(caminho)
        zonas_da_regiao = ctd.ESTACOES_SELECIONADAS.get(regiao, {})
        for zona, info in zonas_da_regiao.items():
            estacao = ctd.processar_estacao_completa(
                df, info["lat"], info["lon"], info["data"], rotulo_log=f"{regiao} / {zona}"
            )
            if estacao is not None:
                estacoes_processadas[regiao][zona] = estacao

    # -----------------------------------------------------------------
    # 1) Comparação de fórmulas (Mackenzie x dP/dρ)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1) COMPARAÇÃO DE FÓRMULAS — Mackenzie/BNDO vs sqrt(dP/dρ)")
    print("=" * 78)
    for regiao, zonas in estacoes_processadas.items():
        if "ZEE" not in zonas:
            continue
        estacao = zonas["ZEE"]
        c_binado = ctd.calcular_c_diferencas_finitas(estacao)

        # Interpola o valor empírico (Mackenzie) nas profundidades dos bins,
        # para poder comparar ponto a ponto
        prof_emp = estacao["Profundidade [m]"].to_numpy()
        c_emp = estacao["Velocidade do som [m/s]"].to_numpy()
        ordem = np.argsort(prof_emp)
        c_emp_interp = np.interp(c_binado["Profundidade [m]"], prof_emp[ordem], c_emp[ordem])

        diff = c_binado["Velocidade do som (dP/dρ) [m/s]"].to_numpy() - c_emp_interp
        diff_valida = diff[np.isfinite(diff)]

        print(f"\n  Região: {regiao}")
        if len(diff_valida) == 0:
            print("    Sem pontos válidos para comparar.")
            continue
        print(f"    Diferença (dP/dρ − Mackenzie): "
              f"méd={diff_valida.mean():+.1f} m/s  "
              f"desvio={diff_valida.std():.1f} m/s  "
              f"min={diff_valida.min():+.1f}  máx={diff_valida.max():+.1f}")
        print(f"    Pontos usados: {len(diff_valida)} de {len(diff)} bins "
              f"({len(diff) - len(diff_valida)} indefinidos)")

    # -----------------------------------------------------------------
    # 2) Estatísticas das derivadas propriedade x propriedade
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2) DERIVADAS PROPRIEDADE x PROPRIEDADE — por região")
    print("=" * 78)
    for regiao, zonas in estacoes_processadas.items():
        if not zonas:
            continue
        print(f"\n  Região: {regiao}")
        for zona, estacao in zonas.items():
            print(f"   Zona: {zona}")
            for col_x, col_y, _, _, titulo in ctd.PARAMETROS_DISPERSAO:
                if col_x not in estacao.columns or col_y not in estacao.columns:
                    continue
                _, deriv = ctd._derivada_por_bins_por_profundidade(
                    estacao["Profundidade [m]"].to_numpy(),
                    estacao[col_x].to_numpy(),
                    estacao[col_y].to_numpy(),
                )
                print(resumir_serie(f"d({titulo.split(' × ')[0]})/d({titulo.split(' × ')[1]})", deriv))

    print("\n" + "=" * 78)
    print("3) DERIVADAS PROPRIEDADE x PROPRIEDADE — por zona (comparando regiões)")
    print("=" * 78)
    for zona in ["Rasa", "Plataforma", "ZEE"]:
        regioes_com_zona = {r: zonas[zona] for r, zonas in estacoes_processadas.items() if zona in zonas}
        if not regioes_com_zona:
            continue
        print(f"\n  Zona: {zona}")
        for regiao, estacao in regioes_com_zona.items():
            print(f"   Região: {regiao}")
            for col_x, col_y, _, _, titulo in ctd.PARAMETROS_DISPERSAO:
                if col_x not in estacao.columns or col_y not in estacao.columns:
                    continue
                _, deriv = ctd._derivada_por_bins_por_profundidade(
                    estacao["Profundidade [m]"].to_numpy(),
                    estacao[col_x].to_numpy(),
                    estacao[col_y].to_numpy(),
                )
                print(resumir_serie(f"d({titulo.split(' × ')[0]})/d({titulo.split(' × ')[1]})", deriv))

    print("\nFim do resumo.")


if __name__ == "__main__":
    main()
