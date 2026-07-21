"""
Extrai, para cada região (Norte, Nordeste, Sudeste, Sul), a estação de CTD
mais profunda identificada no script de exploração anterior, e gera:

  1) Um CSV limpo por região, só com a estação escolhida (pequeno — pode
     ser reaberto rapidamente depois, ou até compartilhado para inspeção).
  2) Um gráfico comparativo com 3 painéis lado a lado: Temperatura(z),
     Salinidade(z) e Velocidade do som(z), uma curva por região.
  3) Uma tabela-resumo com profundidade máxima, T/S/c mínimo-médio-máximo
     por região.

IMPORTANTE — decisão pendente sobre QC:
  A estação do Norte (2025) ainda não passou por revisão formal do CHM
  (flag QC = "000" na maioria dos campos = "ainda não qualificado", não
  necessariamente "dado ruim"). Por isso, o filtro de QC abaixo
  (QC_MINIMO_ACEITAVEL) vem DESLIGADO (None) por padrão — nada é
  descartado, apenas anotado. Ajuste essa variável quando decidir o
  critério com o professor/você mesmo.

Requisitos: pandas, matplotlib
    pip install pandas matplotlib --break-system-packages

Uso:
    python3 processar_estacoes_ctd.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

ARQUIVOS = {
    "Norte": "../dados_ctd_bndo/Costa Norte 2025 Antares",
    "Nordeste": "../dados_ctd_bndo/Nordeste Cruzeiro do Sul",
    "Sudeste": "../dados_ctd_bndo/Oceano Sudeste",
    "Sul": "../dados_ctd_bndo/Oceano Sul V",
}

# Estações escolhidas (as mais profundas de cada região, identificadas no
# script de exploração anterior).
ESTACOES_SELECIONADAS = {
    "Norte":    {"lat": 6.987,   "lon": -44.172, "data": "2025-10-12"},
    "Nordeste": {"lat": -7.973,  "lon": -33.482, "data": "2025-05-01"},
    "Sudeste":  {"lat": -23.084, "lon": -39.466, "data": "2002-11-08"},
    "Sul":      {"lat": -30.241, "lon": -37.008, "data": "2018-05-05"},
}

CASAS_DECIMAIS_COORD = 3

# Pasta onde salvar os CSVs limpos (uma estação por região) e os gráficos
PASTA_SAIDA = Path("../resultados_ctd")

# Filtro de QC (ver aviso no topo do arquivo). None = não filtra, só anota.
# Se quiser filtrar, defina como um inteiro (ex: 2 para manter só "correto"),
# ou uma lista (ex: [2, 3] para aceitar "correto" e "inconsistente").
QC_MINIMO_ACEITAVEL = None

# ---------------------------------------------------------------------------
# FUNÇÕES DE LEITURA E EXTRAÇÃO
# ---------------------------------------------------------------------------


def carregar_arquivo(caminho: str) -> pd.DataFrame:
    """Lê um arquivo CTD do BNDO (CSV separado por ';'), acessando colunas
    por nome (a ordem varia entre arquivos)."""
    df = pd.read_csv(
        caminho,
        sep=";",
        na_values=["None", "none", ""],
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def extrair_estacao(df: pd.DataFrame, lat: float, lon: float, data_str: str) -> pd.DataFrame:
    """Filtra o DataFrame completo para manter só as linhas da estação
    (lat/lon/data) especificada, ordenadas por profundidade crescente."""
    df = df.copy()
    df["_data"] = pd.to_datetime(df["Data-Hora"], errors="coerce").dt.date
    df["_lat_r"] = df["Latitude [deg]"].round(CASAS_DECIMAIS_COORD)
    df["_lon_r"] = df["Longitude [deg]"].round(CASAS_DECIMAIS_COORD)

    alvo_data = pd.to_datetime(data_str).date()
    alvo_lat = round(lat, CASAS_DECIMAIS_COORD)
    alvo_lon = round(lon, CASAS_DECIMAIS_COORD)

    mask = (
        (df["_lat_r"] == alvo_lat)
        & (df["_lon_r"] == alvo_lon)
        & (df["_data"] == alvo_data)
    )
    estacao = df[mask].sort_values("Profundidade [m]").reset_index(drop=True)
    estacao = estacao.drop(columns=["_data", "_lat_r", "_lon_r"])
    return estacao


def extrair_digito_qc(flag) -> "int | None":
    """
    Extrai o dígito de qualidade principal de uma flag de QC.

    As flags observadas nos dados vêm em duas formas:
      - Um único dígito (ex: '9'), usado tipicamente quando o valor está
        ausente ('9' = sem valor).
      - Um código composto de 3 dígitos (ex: '202'), onde o PRIMEIRO
        dígito é o QC geral do BNDO (na mesma escala 0-9: 2=correto,
        3=inconsistente, 4=duvidoso, etc. — ver arquivo Leia-me do BNDO).

    Retorna o primeiro dígito como inteiro, ou None se não for possível
    interpretar (valor ausente/malformado).
    """
    if pd.isna(flag):
        return None
    s = str(flag).strip()
    if not s:
        return None
    try:
        return int(s[0])
    except ValueError:
        return None


def anotar_qualidade(estacao: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas com o dígito de qualidade interpretado para
    Temperatura, Salinidade e Velocidade do som, para inspeção/filtragem."""
    estacao = estacao.copy()
    mapeamento = {
        "QC_Temperatura [Flag]": "qc_temperatura",
        "QC_Salinidade [Flag]": "qc_salinidade",
        "QC_Velocidade do som [Flag]": "qc_velocidade_som",
    }
    for col_original, col_nova in mapeamento.items():
        if col_original in estacao.columns:
            estacao[col_nova] = estacao[col_original].apply(extrair_digito_qc)
    return estacao


def aplicar_filtro_qc(estacao: pd.DataFrame, qc_minimo) -> pd.DataFrame:
    """
    Se QC_MINIMO_ACEITAVEL não for None, filtra as linhas mantendo apenas
    aquelas cujo dígito de qualidade (temperatura E salinidade) esteja no
    conjunto aceitável. Caso contrário, retorna o DataFrame sem alteração.
    """
    if qc_minimo is None:
        return estacao

    aceitos = qc_minimo if isinstance(qc_minimo, (list, set, tuple)) else [qc_minimo]

    mask = pd.Series(True, index=estacao.index)
    for col in ("qc_temperatura", "qc_salinidade"):
        if col in estacao.columns:
            mask &= estacao[col].isin(aceitos)

    return estacao[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    perfis = {}

    for regiao, caminho in ARQUIVOS.items():
        print(f"Processando região: {regiao} ...")
        caminho_path = Path(caminho)
        if not caminho_path.exists():
            print(f"  [ERRO] Arquivo não encontrado: {caminho_path.resolve()}")
            continue

        df = carregar_arquivo(caminho)

        info = ESTACOES_SELECIONADAS[regiao]
        estacao = extrair_estacao(df, info["lat"], info["lon"], info["data"])

        if estacao.empty:
            print(f"  [AVISO] Nenhum ponto encontrado para a estação "
                  f"especificada em {regiao}. Confira lat/lon/data.")
            continue

        estacao = anotar_qualidade(estacao)
        estacao_filtrada = aplicar_filtro_qc(estacao, QC_MINIMO_ACEITAVEL)

        n_antes = len(estacao)
        n_depois = len(estacao_filtrada)
        print(f"  Pontos na estação: {n_antes} "
              f"(após filtro de QC: {n_depois})")

        # Salva o CSV limpo (pequeno, fácil de reabrir depois ou compartilhar)
        caminho_csv = PASTA_SAIDA / f"estacao_{regiao}.csv"
        estacao_filtrada.to_csv(caminho_csv, sep=";", index=False)
        print(f"  Salvo em: {caminho_csv.resolve()}")

        perfis[regiao] = estacao_filtrada

    if not perfis:
        print("\n[ERRO] Nenhuma região foi processada com sucesso. "
              "Confira os caminhos e as coordenadas das estações.")
        return

    # -----------------------------------------------------------------
    # Gráfico comparativo: Temperatura, Salinidade e Velocidade do som
    # -----------------------------------------------------------------
    fig, eixos = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    cores = {"Norte": "tab:red", "Nordeste": "tab:orange",
             "Sudeste": "tab:green", "Sul": "tab:blue"}

    for regiao, estacao in perfis.items():
        cor = cores.get(regiao, None)
        prof = estacao["Profundidade [m]"]

        if "Temperatura [°c]" in estacao.columns:
            eixos[0].plot(estacao["Temperatura [°c]"], prof, label=regiao, color=cor)
        if "Salinidade [psu]" in estacao.columns:
            eixos[1].plot(estacao["Salinidade [psu]"], prof, label=regiao, color=cor)
        if "Velocidade do som [m/s]" in estacao.columns:
            eixos[2].plot(estacao["Velocidade do som [m/s]"], prof, label=regiao, color=cor)

    eixos[0].set_xlabel("Temperatura [°C]")
    eixos[0].set_ylabel("Profundidade [m]")
    eixos[0].invert_yaxis()
    eixos[0].set_title("Perfil de Temperatura")
    eixos[0].legend()
    eixos[0].grid(alpha=0.3)

    eixos[1].set_xlabel("Salinidade [psu]")
    eixos[1].set_title("Perfil de Salinidade")
    eixos[1].grid(alpha=0.3)

    eixos[2].set_xlabel("Velocidade do som [m/s]")
    eixos[2].set_title("Perfil de Velocidade do Som")
    eixos[2].grid(alpha=0.3)

    fig.suptitle("Comparação de Perfis CTD por Região — Costa Brasileira", fontsize=14)
    fig.tight_layout()

    caminho_figura = PASTA_SAIDA / "comparacao_perfis_ctd.png"
    fig.savefig(caminho_figura, dpi=150)
    print(f"\nGráfico comparativo salvo em: {caminho_figura.resolve()}")

    # -----------------------------------------------------------------
    # Tabela-resumo
    # -----------------------------------------------------------------
    print("\nResumo por região:")
    linhas_resumo = []
    for regiao, estacao in perfis.items():
        linhas_resumo.append({
            "Região": regiao,
            "N pontos": len(estacao),
            "Prof. máx [m]": estacao["Profundidade [m]"].max(),
            "T mín/méd/máx [°C]": (
                f"{estacao['Temperatura [°c]'].min():.2f} / "
                f"{estacao['Temperatura [°c]'].mean():.2f} / "
                f"{estacao['Temperatura [°c]'].max():.2f}"
            ) if "Temperatura [°c]" in estacao.columns else "N/A",
            "c mín/méd/máx [m/s]": (
                f"{estacao['Velocidade do som [m/s]'].min():.1f} / "
                f"{estacao['Velocidade do som [m/s]'].mean():.1f} / "
                f"{estacao['Velocidade do som [m/s]'].max():.1f}"
            ) if "Velocidade do som [m/s]" in estacao.columns else "N/A",
        })
    resumo = pd.DataFrame(linhas_resumo)
    print(resumo.to_string(index=False))

    caminho_resumo = PASTA_SAIDA / "resumo_por_regiao.csv"
    resumo.to_csv(caminho_resumo, sep=";", index=False)
    print(f"\nTabela-resumo salva em: {caminho_resumo.resolve()}")


if __name__ == "__main__":
    main()
