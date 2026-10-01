import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
       # /// script
       # requires-python = ">=3.11"
       # dependencies = ["marimo", "numpy", "matplotlib"]
       # ///
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Débruiter un signal grâce à son spectre
    **Objectif** : lire un spectre et choisir la fréquence de coupure *fc* d'un filtre RC passe-bas.
    """)
    return


@app.cell
def _(mo):
    bruit = mo.ui.slider(0.0, 2.0, step=0.1, value=0.8, label="Niveau de bruit")
    R = mo.ui.slider(100, 10000, step=100, value=1000, label="R (Ω)")
    C = mo.ui.slider(0.05, 10.0, step=0.05, value=0.5, label="C (µF)")
    reveler = mo.ui.checkbox(label="Révéler le spectre (après vos prédictions)")
    mo.vstack([bruit, R, C, reveler])
    return C, R, bruit, reveler


@app.cell
def _(bruit, np):
    # Signal : 50 Hz + 120 Hz, plus bruit blanc (graine fixe pour reproductibilité)
    fs = 5000
    N = 5000
    t = np.arange(N) / fs
    propre = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    rng = np.random.default_rng(42)
    x = propre + bruit.value * rng.normal(size=N)
    f = np.fft.rfftfreq(N, 1 / fs)
    X = np.fft.rfft(x)
    return N, X, f, propre, t, x


@app.cell
def _(C, N, R, X, f, mo, np):
    # Filtre RC passe-bas : H(f) = 1 / (1 + j f/fc), fc = 1 / (2 pi R C)
    fc = 1 / (2 * np.pi * R.value * C.value * 1e-6)
    H = 1 / (1 + 1j * f / fc)
    y = np.fft.irfft(X * H, n=N)
    mo.md(f"**Fréquence de coupure : fc = {fc:.0f} Hz**")
    return H, fc, y


@app.cell
def _(plt, propre, t, x, y):
    # Domaine temporel (premières 100 ms)
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    m = t < 0.1
    ax1.plot(t[m] * 1000, x[m], color="lightgray", label="Signal bruité")
    ax1.plot(t[m] * 1000, y[m], color="tab:red", label="Signal filtré")
    ax1.plot(t[m] * 1000, propre[m], "--", color="tab:blue", label="Signal propre")
    ax1.set_xlabel("Temps (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(H, N, X, f, fc, mo, np, plt, reveler):
    mo.stop(
        not reveler.value,
        mo.md("*Prédisez : quelles fréquences voyez-vous dans le signal ? Où placer fc ?*"),
    )
    amp = 2 * np.abs(X) / N
    amp_f = 2 * np.abs(X * H) / N
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    k = f <= 1000
    ax2.plot(f[k], amp[k], color="lightgray", label="Spectre bruité")
    ax2.plot(f[k], amp_f[k], color="tab:red", label="Spectre filtré")
    ax2.axvline(fc, color="black", linestyle=":", label="fc")
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2
    return


if __name__ == "__main__":
    app.run()
