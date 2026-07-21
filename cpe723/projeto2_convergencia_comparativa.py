# projeto2_convergencia_comparativa.py
# Gera o gráfico 2x2 de convergência comparativa ES vs Híbrido
# com mesma seed e mesma população inicial para NC=4 e NC=8.
#
# NÃO faz grid search — roda apenas 1 execução por algoritmo/NC.
# Dependências: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# =========================================================
# Configuração visual
# =========================================================
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})

COR_ES  = "#d62728"   # vermelho
COR_HIB = "#2ca02c"   # verde
COR_REF = "#7f7f7f"   # cinza
ESTILO  = {4: "-", 8: "--"}

output_path = os.path.expanduser("~/cpe723_analise")
os.makedirs(output_path, exist_ok=True)

# =========================================================
# 1. Dados sintéticos
# =========================================================
def generate_data_r3(P=100, NC=8, sigma=0.1, seed=1):
    np.random.seed(seed)
    cluster_centers = np.random.normal(0, 1, (3, NC))
    data_vectors = []
    for k in range(NC):
        centro = cluster_centers[:, k:k+1]
        cluster = sigma * np.random.normal(0, 1, (3, P)) + np.tile(centro, (1, P))
        data_vectors.append(cluster)
    return np.concatenate(data_vectors, axis=1), cluster_centers

def J_hard(X, Y):
    diff  = X[:, :, None] - Y[:, None, :]
    dist2 = np.sum(diff**2, axis=0)
    return np.mean(np.min(dist2, axis=0))

# =========================================================
# 2. Passo DA
# =========================================================
def passo_da(X, Y, T):
    diff = X[:, :, None] - Y[:, None, :]
    d    = np.sum(diff**2, axis=0).T
    p    = np.exp(-d / T)
    p   /= np.sum(p, axis=0)
    weights = np.sum(p, axis=1)
    return (X @ p.T) / weights

# =========================================================
# 3. Execução única — ES ou Híbrido
#    hibrido=False → ES puro (T_da e alpha_da ignorados)
#    hibrido=True  → ES+DA
# =========================================================
def run_uma_execucao(
    data_vectors, NC,
    Nind, Npais, Nfilhos, Nsob,
    Nger=800,
    epson0=1e-8,
    sigma0_low=0.1, sigma0_high=0.5,
    clip_val=None,
    patience=20,
    prec=1.0, pmut=1.0,
    T_da=0.5, alpha_da=1.0,
    tol=1e-2, J_ref=None,
    init_seed=0,
    hibrido=False,
):
    rng = np.random.default_rng(init_seed)
    Nd  = 3 * NC
    X   = data_vectors

    tau1 = 1.0 / np.sqrt(2 * Nd)
    tau2 = 1.0 / np.sqrt(2 * np.sqrt(Nd))

    x_ini     = rng.standard_normal((Nind, Nd))
    sigma_ini = sigma0_low + (sigma0_high - sigma0_low) * rng.random((Nind, Nd))
    ind = np.hstack([x_ini, sigma_ini])

    def custo_batch(pop):
        n_pop = len(pop)
        C     = pop[:, :Nd].reshape(n_pop, 3, NC).transpose(0, 2, 1)
        dv    = data_vectors.T
        diff  = C[:, :, None, :] - dv[None, None, :, :]
        dist2 = np.sum(diff**2, axis=3)
        return np.mean(np.min(dist2, axis=1), axis=1)

    J_ini  = custo_batch(ind)
    D_best = J_ini.min()

    # g=0: população inicial
    history_D     = [float(np.mean(J_ini))]
    history_Dbest = [D_best]
    history_time  = [0.0]

    sem_melhora         = 0
    sem_melhora_pos_suc = 0
    isuc = 0
    t0   = time.perf_counter()

    for g in range(Nger):
        idx_pais = rng.integers(len(ind), size=Npais)
        pais     = ind[idx_pais]

        i1s      = rng.integers(Npais, size=Nfilhos)
        i2s      = rng.integers(Npais, size=Nfilhos)
        mask_rec = rng.random(Nfilhos) < prec
        mask_var = rng.random((Nfilhos, Nd)) < 0.5

        filhos = np.zeros((Nfilhos, 2 * Nd))
        filhos[mask_rec,  :Nd] = np.where(
            mask_var[mask_rec],
            pais[i1s[mask_rec], :Nd],
            pais[i2s[mask_rec], :Nd]
        )
        filhos[mask_rec,  Nd:] = 0.5 * (
            pais[i1s[mask_rec], Nd:] + pais[i2s[mask_rec], Nd:]
        )
        filhos[~mask_rec, :Nd] = pais[i1s[~mask_rec], :Nd]
        filhos[~mask_rec, Nd:] = pais[i1s[~mask_rec], Nd:]

        # Passo DA — só no híbrido
        if hibrido and alpha_da > 0.0:
            for j in range(Nfilhos):
                Y_atual  = filhos[j, :Nd].reshape(3, NC)
                Y_da     = passo_da(X, Y_atual, T_da)
                filhos[j, :Nd] = (alpha_da * Y_da
                                  + (1.0 - alpha_da) * Y_atual).flatten()

        mask_mut = rng.random(Nfilhos) < pmut
        if mask_mut.any():
            r_comum = rng.standard_normal((Nfilhos, 1))
            r_comp  = rng.standard_normal((Nfilhos, Nd))
            r_mut   = rng.standard_normal((Nfilhos, Nd))
            shat = filhos[:, Nd:] * np.exp(tau1 * r_comum + tau2 * r_comp)
            shat = np.maximum(shat, epson0)
            filhos[mask_mut, :Nd] += shat[mask_mut] * r_mut[mask_mut]
            filhos[mask_mut, Nd:]  = shat[mask_mut]

        if clip_val is not None:
            filhos[:, :Nd] = np.clip(filhos[:, :Nd], -clip_val, clip_val)

        J_fil    = custo_batch(filhos)
        idx_sort = np.argsort(J_fil)
        ind      = filhos[idx_sort[:Nsob]]

        if J_fil[idx_sort[0]] < D_best:
            D_best      = J_fil[idx_sort[0]]
            sem_melhora = 0
            if isuc == 0:
                sem_melhora_pos_suc = 0
        else:
            sem_melhora += 1
        if isuc == 1:
            sem_melhora_pos_suc += 1

        history_D.append(float(np.mean(J_fil)))
        history_Dbest.append(D_best)
        history_time.append(time.perf_counter() - t0)

        if isuc == 0 and J_ref is not None:
            if abs(D_best - J_ref) <= tol:
                isuc = 1

        if sem_melhora >= patience:
            break
        if isuc == 1 and sem_melhora_pos_suc >= patience:
            break

    label = "Híbrido" if hibrido else "ES"
    print(f"  {label} NC={NC}: {len(history_Dbest)-1} gerações | "
          f"J_best={D_best:.6f} | isuc={isuc}")

    return (np.array(history_D),
            np.array(history_Dbest),
            np.array(history_time))

# =========================================================
# 4. Configurações finais por NC
# =========================================================
CONFIGS = {
    4: dict(
        Nind=50, Npais=50, Nfilhos=700, Nsob=50,
        patience=10, prec=0.7, pmut=1.0,
        T_da=0.5, alpha_da=1.0,
    ),
    8: dict(
        Nind=100, Npais=100, Nfilhos=700, Nsob=100,
        patience=20, prec=1.0, pmut=1.0,
        T_da=0.1, alpha_da=0.7,
    ),
}

J_REF = {4: 0.028944, 8: 0.030216}
SEED  = 0   # <- mesma seed para ES e Híbrido em cada NC

# =========================================================
# 5. Roda as execuções
# =========================================================
resultados = {}
for nc in [4, 8]:
    print(f"\n=== NC={nc} ===")
    cfg = CONFIGS[nc]
    X, centers = generate_data_r3(NC=nc)
    J_ref = J_REF[nc]

    kwargs = dict(
        data_vectors=X, NC=nc,
        Nger=800, epson0=1e-8,
        sigma0_low=0.1, sigma0_high=0.5,
        clip_val=None,
        tol=1e-2, J_ref=J_ref,
        init_seed=SEED,
        **cfg,
    )

    hD_es,  hDb_es,  ht_es  = run_uma_execucao(**kwargs, hibrido=False)
    hD_hib, hDb_hib, ht_hib = run_uma_execucao(**kwargs, hibrido=True)

    resultados[nc] = dict(
        es  = (hD_es,  hDb_es,  ht_es),
        hib = (hD_hib, hDb_hib, ht_hib),
    )

# =========================================================
# 6. Gráfico 1x2
#    [0] por geração — NC=4 sólido, NC=8 tracejado
#    [1] por tempo   — NC=4 sólido, NC=8 tracejado
# =========================================================
ESTILO_NC = {4: "-", 8: "--"}
LW_BEST   = 2.0
LW_MEDIO  = 0.9

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Convergência Comparativa ES vs Híbrido ES+DA\n"
    "(mesma seed, mesma população inicial)",
    fontweight="bold", fontsize=13
)

for col, xlabel in enumerate(["Geração", "Tempo (s)"]):
    ax = axes[col]
    primeiro_nc4 = True  # controla label do ponto de início

    for nc in [4, 8]:
        r     = resultados[nc]
        J_ref = J_REF[nc]
        cfg   = CONFIGS[nc]
        ls    = ESTILO_NC[nc]

        hD_es,  hDb_es,  ht_es  = r["es"]
        hD_hib, hDb_hib, ht_hib = r["hib"]

        xs_es  = np.arange(len(hDb_es))  if col == 0 else ht_es
        xs_hib = np.arange(len(hDb_hib)) if col == 0 else ht_hib

        # # Custo médio — linha fina semitransparente
        # ax.plot(xs_es,  hD_es,  color=COR_ES,  linestyle=ls,
        #         linewidth=LW_MEDIO, alpha=0.35)
        # ax.plot(xs_hib, hD_hib, color=COR_HIB, linestyle=ls,
        #         linewidth=LW_MEDIO, alpha=0.35)

        # D_best — linha principal
        ax.plot(xs_es,  hDb_es,  color=COR_ES,  linestyle=ls,
                linewidth=LW_BEST,
                label=f"ES  NC={nc}")
        ax.plot(xs_hib, hDb_hib, color=COR_HIB, linestyle=ls,
                linewidth=LW_BEST,
                label=f"Híbrido  NC={nc}")

        # Referência por NC
        ax.axhline(J_ref, color=COR_REF, linestyle=ls, linewidth=0.9,
                   alpha=0.8,
                   label=f"$Custo_{{ref}}$ NC={nc}={J_ref:.4f}")

        # Ponto de início comum (só NC=4 para não duplicar)
        if nc == 4:
            ax.scatter([xs_es[0]], [hDb_es[0]], color="black",
                       zorder=5, s=60,
                       label=f"Início comum (seed={SEED})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("$Custo_{best}$")
    ax.set_title(f"Por {xlabel.split(' ')[0]}   NC=4 (sólido), NC=8 (tracejado)")
    ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
out = os.path.join(output_path, "convergencia_es_vs_hibrido.png")
plt.savefig(out)
plt.show()
print(f"\nSalvo: {out}")