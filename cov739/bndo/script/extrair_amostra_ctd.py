"""
Extrai as primeiras N linhas de cada arquivo de dados CTD encontrado em uma
pasta de origem (busca recursiva em subpastas) e salva cada amostra em um
arquivo separado dentro de uma pasta de destino.

Objetivo: gerar amostras pequenas o suficiente para inspecionar a estrutura
dos arquivos (colunas, separador, cabeçalho) sem precisar mexer nos arquivos
originais de 360MB.

Uso:
    python extrair_amostra_ctd.py

Antes de rodar, ajuste as três variáveis abaixo (PASTA_ORIGEM, PASTA_DESTINO,
N_LINHAS) conforme necessário.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste aqui antes de rodar
# ---------------------------------------------------------------------------

# Pasta onde estão os arquivos originais do BNDO (pode ter subpastas dentro)
PASTA_ORIGEM = Path("./../dados_ctd_bndo")

# Pasta onde as amostras (primeiras N linhas) serão salvas
PASTA_DESTINO = Path("./../amostras_ctd")

# Quantas linhas extrair de cada arquivo
N_LINHAS = 50

# Extensões de arquivo consideradas "texto" (serão processadas).
# Se os arquivos do BNDO tiverem outra extensão (ex: .dat, .cnv, .odv),
# adicione aqui. Deixe como None para tentar processar TODOS os arquivos
# encontrados (não recomendado se houver arquivos binários misturados).
EXTENSOES_VALIDAS = {".txt", ".csv", ".dat", ".cnv", ".odv", ".asc", ".tsv"}

# ---------------------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------------------


def eh_arquivo_valido(caminho: Path) -> bool:
    """Decide se um arquivo deve ser processado, com base na extensão."""
    if EXTENSOES_VALIDAS is None:
        return True
    return caminho.suffix.lower() in EXTENSOES_VALIDAS


def ler_primeiras_linhas(caminho: Path, n: int) -> list[str]:
    """
    Lê as primeiras n linhas de um arquivo de texto, sem carregar o arquivo
    inteiro na memória (importante para arquivos grandes).

    Tenta primeiro UTF-8; se falhar (erro de decodificação), tenta Latin-1,
    que é comum em arquivos gerados por sistemas mais antigos/institucionais.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            linhas = []
            with open(caminho, "r", encoding=encoding, errors="strict") as f:
                for i, linha in enumerate(f):
                    if i >= n:
                        break
                    linhas.append(linha)
            return linhas
        except UnicodeDecodeError:
            continue
    # Se nem utf-8 nem latin-1 funcionarem, força leitura ignorando erros
    linhas = []
    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        for i, linha in enumerate(f):
            if i >= n:
                break
            linhas.append(linha)
    return linhas


def nome_amostra_unico(caminho_original: Path, pasta_origem: Path) -> str:
    """
    Gera um nome de arquivo de saída que preserva a origem (pasta + nome),
    para não perder a rastreabilidade quando há arquivos de mesmo nome em
    subpastas diferentes.

    Exemplo:
        dados_ctd_bndo/Costa Norte 2025 Antares/estacao01.txt
        -> Costa_Norte_2025_Antares__estacao01.txt
    """
    caminho_relativo = caminho_original.relative_to(pasta_origem)
    partes = caminho_relativo.parts  # tupla com pasta(s) + nome do arquivo
    nome_seguro = "__".join(partes).replace(" ", "_")
    return nome_seguro


def main():
    if not PASTA_ORIGEM.exists():
        print(f"[ERRO] Pasta de origem não encontrada: {PASTA_ORIGEM.resolve()}")
        print("Ajuste a variável PASTA_ORIGEM no início do script.")
        return

    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)

    arquivos_encontrados = [
        p for p in PASTA_ORIGEM.rglob("*") if p.is_file() and eh_arquivo_valido(p)
    ]

    if not arquivos_encontrados:
        print(f"[AVISO] Nenhum arquivo com extensão válida encontrado em "
              f"{PASTA_ORIGEM.resolve()}")
        print(f"Extensões aceitas: {EXTENSOES_VALIDAS}")
        return

    print(f"Encontrados {len(arquivos_encontrados)} arquivo(s) em "
          f"{PASTA_ORIGEM.resolve()}\n")

    processados = 0
    erros = 0

    for caminho in arquivos_encontrados:
        try:
            linhas = ler_primeiras_linhas(caminho, N_LINHAS)
            nome_saida = nome_amostra_unico(caminho, PASTA_ORIGEM)
            caminho_saida = PASTA_DESTINO / nome_saida

            with open(caminho_saida, "w", encoding="utf-8") as f_out:
                f_out.writelines(linhas)

            tamanho_original_mb = caminho.stat().st_size / (1024 * 1024)
            print(f"  OK  {caminho.relative_to(PASTA_ORIGEM)}  "
                  f"({tamanho_original_mb:.1f} MB original -> "
                  f"{len(linhas)} linhas extraídas)")
            processados += 1

        except Exception as e:
            print(f"  ERRO ao processar {caminho}: {e}")
            erros += 1

    print(f"\nConcluído: {processados} arquivo(s) processado(s), "
          f"{erros} erro(s).")
    print(f"Amostras salvas em: {PASTA_DESTINO.resolve()}")


if __name__ == "__main__":
    main()