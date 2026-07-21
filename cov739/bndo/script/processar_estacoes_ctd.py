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

Requisitos: pandas, matplotlib, seawater
    pip install pandas matplotlib seawater --break-system-packages

Uso:
    python3 processar_estacoes_ctd.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seawater as sw
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


def completar_profundidade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Profundidade [m]' vier vazia mas 'Pressão [db]' estiver
    disponível, calcula a profundidade a partir da pressão (fórmula de
    Saunders & Fofonoff, via seawater.dpth(), que leva em conta a
    latitude). Marca a origem em 'origem_profundidade'.

    IMPORTANTE: esta função deve ser chamada ANTES de
    colapsar_duplicatas_por_profundidade — como o colapso agrupa as
    linhas pela coluna de profundidade, qualquer linha com profundidade
    ausente seria descartada silenciosamente nesse agrupamento, mesmo
    tendo pressão e outros dados válidos.
    """
    estacao = estacao.copy()

    if "Pressão [db]" not in estacao.columns:
        estacao["origem_profundidade"] = "BNDO"
        return estacao

    tem_bndo = estacao["Profundidade [m]"].notna()
    pode_calcular = (~tem_bndo) & estacao["Pressão [db]"].notna()

    estacao["origem_profundidade"] = "ausente"
    estacao.loc[tem_bndo, "origem_profundidade"] = "BNDO"

    if pode_calcular.any():
        estacao.loc[pode_calcular, "Profundidade [m]"] = sw.dpth(
            estacao.loc[pode_calcular, "Pressão [db]"].values, latitude
        )
        estacao.loc[pode_calcular, "origem_profundidade"] = "calculada (dpth/Saunders-Fofonoff)"

    return estacao


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


def _primeiro_valido(serie: pd.Series):
    """Retorna o primeiro valor não-nulo de uma Series, ou NaN se não houver nenhum."""
    validos = serie.dropna()
    return validos.iloc[0] if len(validos) > 0 else pd.NA


def colapsar_duplicatas_por_profundidade(estacao: pd.DataFrame, casas_decimais_prof: int = 1) -> pd.DataFrame:
    """
    O BNDO frequentemente grava, para uma mesma profundidade nominal, duas
    (ou mais) linhas com timestamps próximos: uma com Pressão/Condutividade/
    Densidade/Velocidade do som, outra com Temperatura/Salinidade. Sem
    juntar essas linhas, cada parâmetro fica com "buracos" e os perfis saem
    incompletos.

    Esta função agrupa por profundidade (arredondada) e, para cada coluna
    de interesse, pega o primeiro valor NÃO nulo entre as linhas do grupo —
    reconstituindo uma única linha por profundidade com o máximo de
    informação disponível.
    """
    estacao = estacao.copy()
    estacao["Profundidade [m]"] = estacao["Profundidade [m]"].round(casas_decimais_prof)

    colunas_numericas = [
        "Temperatura [°c]", "Salinidade [psu]", "Condutividade [S/m]",
        "Velocidade do som [m/s]", "Densidade ro [kg/m³]",
        "Pressão [db]", "Temperatura Potencial [°C]",
    ]
    colunas_qc_brutas = [
        "QC_Temperatura [Flag]", "QC_Salinidade [Flag]",
        "QC_Velocidade do som [Flag]", "QC_Condutividade [Flag]",
        "QC_Densidade ro [Flag]", "QC_Pressão [Flag]",
    ]
    colunas_para_combinar = [
        c for c in (colunas_numericas + colunas_qc_brutas + ["Data-Hora"])
        if c in estacao.columns
    ]

    agregadores = {c: _primeiro_valido for c in colunas_para_combinar}

    colapsada = (
        estacao.groupby("Profundidade [m]", as_index=False)
        .agg(agregadores)
        .sort_values("Profundidade [m]")
        .reset_index(drop=True)
    )

    # Garante que as colunas numéricas fiquem com NaN "de verdade" (float),
    # e não com o tipo pd.NA (que o matplotlib não sabe interpretar).
    for c in colunas_numericas:
        if c in colapsada.columns:
            colapsada[c] = pd.to_numeric(colapsada[c], errors="coerce")

    return colapsada


def celeridade_mackenzie(temperatura, salinidade, profundidade):
    """
    Equação de Mackenzie (1981) para a velocidade do som na água do mar,
    válida para T entre 2-30°C, S entre 25-40 psu, profundidade até 8000m.

    c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
        + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D^2
        - 1.025e-2*T*(S-35) - 7.139e-13*T*D^3

    onde T = temperatura [°C], S = salinidade [psu], D = profundidade [m].
    """
    T, S, D = temperatura, salinidade, profundidade
    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D**3
    )


def completar_velocidade_som(estacao: pd.DataFrame) -> pd.DataFrame:
    """
    Sempre que 'Velocidade do som [m/s]' vier vazia do BNDO, calcula o
    valor usando a equação de Mackenzie a partir de T, S e profundidade
    (quando esses três estiverem disponíveis), e marca a origem do dado
    numa nova coluna 'origem_c'.
    """
    estacao = estacao.copy()
    tem_bndo = estacao["Velocidade do som [m/s]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Profundidade [m]"].notna()
    )

    estacao["origem_c"] = "ausente"
    estacao.loc[tem_bndo, "origem_c"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    estacao.loc[calcular_aqui, "Velocidade do som [m/s]"] = celeridade_mackenzie(
        estacao.loc[calcular_aqui, "Temperatura [°c]"],
        estacao.loc[calcular_aqui, "Salinidade [psu]"],
        estacao.loc[calcular_aqui, "Profundidade [m]"],
    )
    estacao.loc[calcular_aqui, "origem_c"] = "Mackenzie (calculado)"

    return estacao


def completar_pressao(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Pressão [db]' vier vazia do BNDO, calcula o valor a
    partir da profundidade (fórmula de Saunders & Fofonoff, via
    seawater.pres(), que leva em conta a latitude). Marca a origem em
    'origem_pressao'.

    Deve ser chamada depois de completar_profundidade() (para garantir
    que 'Profundidade [m]' já esteja completa) e antes de
    completar_densidade()/completar_condutividade() (que dependem da
    pressão já preenchida).
    """
    estacao = estacao.copy()
    if "Pressão [db]" not in estacao.columns:
        estacao["Pressão [db]"] = pd.NA

    tem_bndo = estacao["Pressão [db]"].notna()
    pode_calcular = (~tem_bndo) & estacao["Profundidade [m]"].notna()

    estacao["origem_pressao"] = "ausente"
    estacao.loc[tem_bndo, "origem_pressao"] = "BNDO"

    if pode_calcular.any():
        estacao.loc[pode_calcular, "Pressão [db]"] = sw.pres(
            estacao.loc[pode_calcular, "Profundidade [m]"].values, latitude
        )
        estacao.loc[pode_calcular, "origem_pressao"] = "calculada (pres/Saunders-Fofonoff)"

    estacao["Pressão [db]"] = pd.to_numeric(estacao["Pressão [db]"], errors="coerce")
    return estacao


def completar_densidade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Densidade ro [kg/m³]' vier vazia do BNDO, calcula o valor
    pela equação de estado da água do mar EOS-80 (UNESCO 1980), a partir
    de Salinidade, Temperatura e Pressão — via seawater.dens(). Marca a
    origem em 'origem_rho'.

    Pressupõe que 'Pressão [db]' já foi completada por completar_pressao().
    """
    estacao = estacao.copy()
    tem_bndo = estacao["Densidade ro [kg/m³]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Pressão [db]"].notna()
    )

    estacao["origem_rho"] = "ausente"
    estacao.loc[tem_bndo, "origem_rho"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    if calcular_aqui.any():
        estacao.loc[calcular_aqui, "Densidade ro [kg/m³]"] = sw.dens(
            estacao.loc[calcular_aqui, "Salinidade [psu]"].values,
            estacao.loc[calcular_aqui, "Temperatura [°c]"].values,
            estacao.loc[calcular_aqui, "Pressão [db]"].values,
        )
        estacao.loc[calcular_aqui, "origem_rho"] = "EOS-80 (calculado)"

    return estacao


def completar_condutividade(estacao: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """
    Sempre que 'Condutividade [S/m]' vier vazia do BNDO, calcula o valor a
    partir de Salinidade, Temperatura e Pressão, usando a escala prática
    de salinidade PSS-78 (via seawater.cndr(), que retorna a razão de
    condutividade R = C(S,T,P)/C(35,15,0)).

    A condutividade de referência C(35,15,0) = 42.914 mS/cm = 4.2914 S/m
    é a constante padrão da escala PSS-78. Marca a origem em 'origem_sigma'.

    Pressupõe que 'Pressão [db]' já foi completada por completar_pressao().
    """
    estacao = estacao.copy()
    C_REFERENCIA_S_POR_M = 4.2914  # 42.914 mS/cm convertido para S/m

    tem_bndo = estacao["Condutividade [S/m]"].notna()

    pode_calcular = (
        estacao["Temperatura [°c]"].notna()
        & estacao["Salinidade [psu]"].notna()
        & estacao["Pressão [db]"].notna()
    )

    estacao["origem_sigma"] = "ausente"
    estacao.loc[tem_bndo, "origem_sigma"] = "BNDO"

    calcular_aqui = (~tem_bndo) & pode_calcular
    if calcular_aqui.any():
        razao = sw.cndr(
            estacao.loc[calcular_aqui, "Salinidade [psu]"].values,
            estacao.loc[calcular_aqui, "Temperatura [°c]"].values,
            estacao.loc[calcular_aqui, "Pressão [db]"].values,
        )
        estacao.loc[calcular_aqui, "Condutividade [S/m]"] = razao * C_REFERENCIA_S_POR_M
        estacao.loc[calcular_aqui, "origem_sigma"] = "PSS-78 (calculado)"

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

        n_antes_colapso = len(estacao)
        estacao = completar_profundidade(estacao, info["lat"])
        n_prof_calc = (estacao["origem_profundidade"] == "calculada (dpth/Saunders-Fofonoff)").sum()
        n_prof_ausente = (estacao["origem_profundidade"] == "ausente").sum()
        if n_prof_calc or n_prof_ausente:
            print(f"  Profundidade — {n_prof_calc} calculadas a partir da pressão, "
                  f"{n_prof_ausente} linhas descartadas por falta de profundidade E pressão")

        estacao = estacao.dropna(subset=["Profundidade [m]"]).reset_index(drop=True)

        estacao = colapsar_duplicatas_por_profundidade(estacao)
        print(f"  Linhas antes/depois de colapsar duplicatas por profundidade: "
              f"{n_antes_colapso} -> {len(estacao)}")

        estacao = anotar_qualidade(estacao)
        estacao = completar_velocidade_som(estacao)
        n_bndo = (estacao["origem_c"] == "BNDO").sum()
        n_calc = (estacao["origem_c"] == "Mackenzie (calculado)").sum()
        n_ausente = (estacao["origem_c"] == "ausente").sum()
        print(f"  Velocidade do som — origem: {n_bndo} do BNDO, "
              f"{n_calc} calculadas (Mackenzie), {n_ausente} ainda ausentes")

        estacao = completar_pressao(estacao, info["lat"])
        n_bndo_p = (estacao["origem_pressao"] == "BNDO").sum()
        n_calc_p = (estacao["origem_pressao"] == "calculada (pres/Saunders-Fofonoff)").sum()
        n_ausente_p = (estacao["origem_pressao"] == "ausente").sum()
        print(f"  Pressão — origem: {n_bndo_p} do BNDO, "
              f"{n_calc_p} calculadas (Saunders-Fofonoff), {n_ausente_p} ainda ausentes")

        estacao = completar_densidade(estacao, info["lat"])
        n_bndo_rho = (estacao["origem_rho"] == "BNDO").sum()
        n_calc_rho = (estacao["origem_rho"] == "EOS-80 (calculado)").sum()
        n_ausente_rho = (estacao["origem_rho"] == "ausente").sum()
        print(f"  Densidade — origem: {n_bndo_rho} do BNDO, "
              f"{n_calc_rho} calculadas (EOS-80), {n_ausente_rho} ainda ausentes")

        estacao = completar_condutividade(estacao, info["lat"])
        n_bndo_sigma = (estacao["origem_sigma"] == "BNDO").sum()
        n_calc_sigma = (estacao["origem_sigma"] == "PSS-78 (calculado)").sum()
        n_ausente_sigma = (estacao["origem_sigma"] == "ausente").sum()
        print(f"  Condutividade — origem: {n_bndo_sigma} do BNDO, "
              f"{n_calc_sigma} calculadas (PSS-78), {n_ausente_sigma} ainda ausentes")

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
    # Gráfico comparativo: Temperatura, Salinidade, Velocidade do som,
    # Densidade e Condutividade — todos os parâmetros pedidos ao BNDO,
    # exceto profundidade/pressão (que já é o eixo vertical comum).
    # -----------------------------------------------------------------
    parametros = [
        ("Temperatura [°c]", "Temperatura [°C]", "Perfil de Temperatura"),
        ("Salinidade [psu]", "Salinidade [psu]", "Perfil de Salinidade"),
        ("Velocidade do som [m/s]", "Velocidade do som [m/s]", "Perfil de Velocidade do Som"),
        ("Densidade ro [kg/m³]", "Densidade [kg/m³]", "Perfil de Densidade"),
        ("Condutividade [S/m]", "Condutividade [S/m]", "Perfil de Condutividade"),
    ]

    cores = {"Norte": "tab:red", "Nordeste": "tab:orange",
             "Sudeste": "tab:green", "Sul": "tab:blue"}

    fig = plt.figure(figsize=(18, 11))
    gs_principal = fig.add_gridspec(2, 3)

    # Os 5 primeiros painéis (T, S, c, ρ, σ) — todas as regiões sobrepostas,
    # compartilhando o eixo de profundidade.
    eixos = []
    for i, (col, rotulo_x, titulo) in enumerate(parametros):
        linha, coluna = divmod(i, 3)
        ax = fig.add_subplot(gs_principal[linha, coluna],
                              sharey=eixos[0] if eixos else None)
        for regiao, estacao in perfis.items():
            if col in estacao.columns:
                ax.plot(estacao[col], estacao["Profundidade [m]"],
                        label=regiao, color=cores.get(regiao))
        ax.set_xlabel(rotulo_x)
        ax.set_title(titulo)
        ax.grid(alpha=0.3)
        eixos.append(ax)

    eixos[0].set_ylabel("Profundidade [m]")
    eixos[0].invert_yaxis()  # afeta todos os eixos, pois compartilham o y
    eixos[0].legend()

    # 6º painel (posição [1,2]) subdividido em um mini-grid 2x2: um
    # gráfico de Profundidade x Pressão por região, com escala própria
    # (já que as 4 curvas são quase coincidentes num único gráfico
    # compartilhado, tornando a comparação pouco útil).
    gs_pressao = gs_principal[1, 2].subgridspec(2, 2, hspace=0.6, wspace=0.5)
    for idx, regiao in enumerate(["Norte", "Nordeste", "Sudeste", "Sul"]):
        linha, coluna = divmod(idx, 2)
        ax_mini = fig.add_subplot(gs_pressao[linha, coluna])
        if regiao in perfis and "Pressão [db]" in perfis[regiao].columns:
            estacao = perfis[regiao]
            ax_mini.plot(estacao["Pressão [db]"], estacao["Profundidade [m]"],
                         color=cores.get(regiao))
        ax_mini.invert_yaxis()
        ax_mini.set_title(regiao, fontsize=9)
        ax_mini.tick_params(labelsize=7)
        ax_mini.grid(alpha=0.3)

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
        def faixa(col):
            if col not in estacao.columns or estacao[col].dropna().empty:
                return "N/A"
            return (f"{estacao[col].min():.2f} / "
                    f"{estacao[col].mean():.2f} / "
                    f"{estacao[col].max():.2f}")

        linhas_resumo.append({
            "Região": regiao,
            "N pontos": len(estacao),
            "Prof. máx [m]": estacao["Profundidade [m]"].max(),
            "T mín/méd/máx [°C]": faixa("Temperatura [°c]"),
            "S mín/méd/máx [psu]": faixa("Salinidade [psu]"),
            "c mín/méd/máx [m/s]": faixa("Velocidade do som [m/s]"),
            "ρ mín/méd/máx [kg/m³]": faixa("Densidade ro [kg/m³]"),
            "σ mín/méd/máx [S/m]": faixa("Condutividade [S/m]"),
            "P mín/méd/máx [db]": faixa("Pressão [db]"),
        })
    resumo = pd.DataFrame(linhas_resumo)
    print(resumo.to_string(index=False))

    caminho_resumo = PASTA_SAIDA / "resumo_por_regiao.csv"
    resumo.to_csv(caminho_resumo, sep=";", index=False)
    print(f"\nTabela-resumo salva em: {caminho_resumo.resolve()}")


if __name__ == "__main__":
    main()