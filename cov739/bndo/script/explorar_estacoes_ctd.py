"""
Explora os arquivos de dados CTD do BNDO (um por região) para entender:
  1) Quais colunas cada arquivo tem (a ordem varia entre arquivos!)
  2) Quantas estações/lances distintos existem dentro de cada arquivo
  3) A profundidade máxima e o número de pontos de cada estação

Isso é um passo de DIAGNÓSTICO, antes de decidirmos qual estação usar por
região no trabalho final.

Requisitos: pandas
    pip install pandas --break-system-packages   (se ainda não tiver)

Uso:
    python3 explorar_estacoes_ctd.py
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste os caminhos conforme necessário
# ---------------------------------------------------------------------------

ARQUIVOS = {
    "Norte": "../dados_ctd_bndo/Costa Norte 2025 Antares",
    "Nordeste": "../dados_ctd_bndo/Nordeste Cruzeiro do Sul",
    "Sudeste": "../dados_ctd_bndo/Oceano Sudeste",
    "Sul": "../dados_ctd_bndo/Oceano Sul V",
}

# Quantas casas decimais usar para agrupar Latitude/Longitude ao identificar
# uma "estação" (evita que pequenas variações de ponto flutuante quebrem o
# agrupamento de pontos que são, na prática, a mesma estação/lance).
CASAS_DECIMAIS_COORD = 3

# ---------------------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------------------


def carregar_arquivo(caminho: str) -> pd.DataFrame:
    """
    Lê um arquivo CTD do BNDO como CSV separado por ';', tratando 'None'
    como valor ausente (NaN). Não assume nenhuma ordem fixa de colunas —
    tudo é acessado pelo NOME da coluna depois de carregado.
    """
    df = pd.read_csv(
        caminho,
        sep=";",
        na_values=["None", "none", ""],
        encoding="utf-8",
        engine="python",  # mais tolerante a campos de texto livre com ';' internos problemáticos
        on_bad_lines="warn",
    )
    # Remove espaços acidentais nos nomes de coluna (ex: " CTQC_Profundidade")
    df.columns = [c.strip() for c in df.columns]
    return df


def identificar_estacoes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa as linhas do DataFrame por estação, usando Latitude/Longitude
    (arredondadas) + a data (sem hora) como identificador de uma estação.

    Retorna um resumo com: lat/lon, data, profundidade máxima, número de
    pontos e a plataforma/comissão registrada.
    """
    df = df.copy()

    # Extrai só a data (sem hora), assumindo que uma mesma estação pode ter
    # pequenas variações de horário entre a primeira e a última medição.
    df["_data"] = pd.to_datetime(df["Data-Hora"], errors="coerce").dt.date

    df["_lat_r"] = df["Latitude [deg]"].round(CASAS_DECIMAIS_COORD)
    df["_lon_r"] = df["Longitude [deg]"].round(CASAS_DECIMAIS_COORD)

    coluna_profundidade = "Profundidade [m]"

    agrupado = (
        df.groupby(["_lat_r", "_lon_r", "_data"])
        .agg(
            n_pontos=(coluna_profundidade, "count"),
            profundidade_max=(coluna_profundidade, "max"),
            profundidade_min=(coluna_profundidade, "min"),
        )
        .reset_index()
        .rename(columns={"_lat_r": "Latitude", "_lon_r": "Longitude", "_data": "Data"})
        .sort_values("profundidade_max", ascending=False)
    )

    return agrupado


def main():
    for regiao, caminho in ARQUIVOS.items():
        caminho_path = Path(caminho)
        print("=" * 70)
        print(f"REGIÃO: {regiao}  ({caminho})")
        print("=" * 70)

        if not caminho_path.exists():
            print(f"  [ERRO] Arquivo não encontrado: {caminho_path.resolve()}")
            continue

        try:
            df = carregar_arquivo(caminho)
        except Exception as e:
            print(f"  [ERRO] Falha ao ler o arquivo: {e}")
            continue

        print(f"  Total de linhas no arquivo: {len(df)}")
        print(f"  Colunas encontradas ({len(df.columns)}):")
        for c in df.columns:
            print(f"    - {c}")

        colunas_essenciais = [
            "Data-Hora", "Latitude [deg]", "Longitude [deg]",
            "Profundidade [m]", "Temperatura [°c]", "Salinidade [psu]",
            "Velocidade do som [m/s]", "Densidade ro [kg/m³]",
        ]
        faltando = [c for c in colunas_essenciais if c not in df.columns]
        if faltando:
            print(f"  [AVISO] Colunas essenciais ausentes neste arquivo: {faltando}")

        try:
            estacoes = identificar_estacoes(df)
            print(f"\n  Estações/lances distintos identificados: {len(estacoes)}")
            print("  Top 5 mais profundas:")
            print(estacoes.head(5).to_string(index=False))
        except Exception as e:
            print(f"  [ERRO] Falha ao identificar estações: {e}")

        print()


if __name__ == "__main__":
    main()
