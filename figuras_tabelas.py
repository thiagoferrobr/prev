import importlib.util
import io
import logging
import os
import pickle
import sys
import types
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from scipy.signal import lombscargle
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import acf, kpss, pacf


def _fallback_pv():
    ns = types.SimpleNamespace()
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    drive = "/content/drive/MyDrive"
    ns.DATA_DIR = os.environ.get("PREV_DATA_DIR", os.path.join(drive, "dados_inmet") if in_colab else "dados_inmet")
    ns.OUT_DIR = os.environ.get("PREV_OUT_DIR", os.path.join(drive, "prev_v9") if in_colab else "prev_v9")
    ns.STATIONS = {"F501": dict(name="Cercadinho", lat=-19.98, lon=-43.95861, tz=-3),
                   "A703": dict(name="Ponta Pora", lat=-22.5525, lon=-55.7164, tz=-4)}
    ns.YEARS = (2023, 2024, 2025)
    ns.ISC, ns.ACC_H, ns.SUBSTEPS, ns.MU0_MIN, ns.KT_MAX = 1361.0, 1.0, 60, 0.10, 1.0
    ns.HORIZONS, ns.MAX_SCAN_LAG = (1, 3, 6), 72
    ns.BMA_KAPPA, ns.BMA_KAPPAS = 5.0, (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf)
    ns.PERS = "Pers: Smart"
    ns.BASE = ["ML: ANN", "ML: SVM", "ML: XGBoost", "DL: LSTM+Att", "Hybrid: SVM+LSTM-res"]
    ns.EXOG = ["temp", "temp_dew", "rh", "pres", "wind_speed", "wind_gust", "precip",
               "amplitude_termica", "deficit_orvalho", "pressao_vapor", "tendencia_pressao",
               "vento_sin", "vento_cos", "chuva_ocorrencia"]
    cols = {"GHI": ("RADIACAO GLOBAL",), "precip": ("PRECIPITA",),
            "pres": ("PRESSAO ATMOSFERICA AO NIVEL",), "temp": ("TEMPERATURA DO AR",),
            "temp_dew": ("TEMPERATURA DO PONTO DE ORVALHO",),
            "temp_max": ("TEMPERATURA MÁXIMA", "TEMPERATURA MAXIMA"),
            "temp_min": ("TEMPERATURA MÍNIMA", "TEMPERATURA MINIMA"),
            "rh": ("UMIDADE RELATIVA DO AR",), "wind_dir": ("VENTO, DIRE",),
            "wind_gust": ("VENTO, RAJADA",), "wind_speed": ("VENTO, VELOCIDADE",)}
    ranges = {"precip": (0, 200), "pres": (600, 1100), "temp": (-20, 55), "temp_dew": (-30, 40),
              "temp_max": (-20, 55), "temp_min": (-20, 55), "rh": (0, 100), "wind_dir": (0, 360),
              "wind_gust": (0, 75), "wind_speed": (0, 60)}

    def parse_inmet(src):
        raw = pd.read_csv(src, sep=";", skiprows=8, encoding="latin-1", on_bad_lines="skip")
        raw.columns = [c.strip().upper() for c in raw.columns]
        date = raw[next(c for c in raw.columns if c.startswith("DATA"))].astype(str).str.replace("/", "-")
        hour = (raw[next(c for c in raw.columns if c.startswith("HORA"))].astype(str)
                .str.replace(":", "").str.replace("UTC", "").str.strip().str.zfill(4).str[:2])
        out = pd.DataFrame({"t": pd.to_datetime(date + " " + hour + ":00", errors="coerce")})
        for key, frags in cols.items():
            col = next((c for f in frags for c in raw.columns if f in c), None)
            out[key] = (pd.to_numeric(raw[col].astype(str).str.replace(",", "."), errors="coerce")
                        if col else np.nan)
        out["GHI"] = out["GHI"] / 3.6
        return out.dropna(subset=["t"])

    def header_coords(src):
        meta = {}
        with open(src, encoding="latin-1") as f:
            for _ in range(8):
                key, _, val = f.readline().partition(";")
                meta[key.strip().rstrip(":").upper()] = val.strip().replace(",", ".")
        return float(meta["LATITUDE"]), float(meta["LONGITUDE"])

    def station_files(sid):
        names = os.listdir(ns.DATA_DIR) if os.path.isdir(ns.DATA_DIR) else []
        return sorted(os.path.join(ns.DATA_DIR, f) for f in names
                      if f.upper().endswith(".CSV") and f"_{sid}_" in f.upper() and any(str(y) in f for y in ns.YEARS))

    def fetch_portal():
        os.makedirs(ns.DATA_DIR, exist_ok=True)
        for y in ns.YEARS:
            r = requests.get(f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{y}.zip", timeout=1800)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for n in z.namelist():
                    if n.upper().endswith(".CSV") and any(f"_{sid}_" in n.upper() for sid in ns.STATIONS):
                        with open(os.path.join(ns.DATA_DIR, os.path.basename(n)), "wb") as f:
                            f.write(z.read(n))

    def read_station(sid):
        if not station_files(sid):
            fetch_portal()
        files = station_files(sid)
        d = pd.concat(parse_inmet(f) for f in files).drop_duplicates("t").set_index("t").sort_index()
        d = d[(d.index.year >= min(ns.YEARS)) & (d.index.year <= max(ns.YEARS))]
        return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h")), header_coords(files[0])

    def exog_qc(d):
        d, rows = d.copy(), []
        for c, (lo, hi) in ranges.items():
            bad = d[c].notna() & ((d[c] < lo) | (d[c] > hi))
            rows.append(dict(Variavel=c, N_presente=int(d[c].notna().sum()), N_fora_da_faixa=int(bad.sum())))
            d.loc[bad, c] = np.nan
            rows[-1]["Frac_disponivel"] = float(d[c].notna().mean())
        return d, pd.DataFrame(rows)

    def exog_features(d):
        d = d.copy()
        d["amplitude_termica"] = d["temp_max"] - d["temp_min"]
        d["deficit_orvalho"] = d["temp"] - d["temp_dew"]
        d["pressao_vapor"] = 6.112 * np.exp(17.67 * d["temp_dew"] / (d["temp_dew"] + 243.5))
        d["tendencia_pressao"] = d["pres"].diff()
        a = np.deg2rad(d["wind_dir"])
        d["vento_sin"], d["vento_cos"] = np.sin(a), np.cos(a)
        d["chuva_ocorrencia"] = (d["precip"] > 0).astype(float).where(d["precip"].notna())
        for c in ns.EXOG:
            d[f"{c}_lag1"] = d[c].shift(1)
        return d

    def equation_of_time(g):
        return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                         - 0.014615 * np.cos(2 * g) - 0.04089 * np.sin(2 * g))

    def spencer(doy, hour_utc, lat, lon):
        g = 2 * np.pi * (doy - 1) / 365
        decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
                + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.001480 * np.sin(3 * g))
        omega = np.deg2rad(15 * (hour_utc + lon / 15 + equation_of_time(g) / 60 - 12))
        phi = np.deg2rad(lat)
        mu = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega), 0, 1)
        e0 = (1.00011 + 0.034221 * np.cos(g) + 0.00128 * np.sin(g)
              + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))
        return mu, e0

    def geometry(ts, lat, lon):
        doy = ts.dayofyear.values.astype(float)
        end = (ts.hour + ts.minute / 60).values.astype(float)
        mu_mid, e0 = spencer(doy, end - ns.ACC_H / 2, lat, lon)
        frac = (np.arange(ns.SUBSTEPS) + 0.5) / ns.SUBSTEPS
        mu_s, e_s = spencer(doy[None, :], end[None, :] - ns.ACC_H * (1 - frac[:, None]), lat, lon)
        mu_end, e_end = spencer(doy, end, lat, lon)
        return mu_mid, ns.ISC * e0, ns.ISC * (e_s * mu_s).mean(axis=0), ns.ISC * e_end * mu_end

    def quality_control(raw, lat, lon, mu_min=ns.MU0_MIN):
        d = raw.dropna(subset=["GHI"]).copy()
        d["mu0"], d["Sa"], d["H0h"], d["H0h_carimbo"] = geometry(d.index, lat, lon)
        n = {"N_bruto": len(d)}
        d = d[d["mu0"] > mu_min]
        n["N_mascara_diurna"] = len(d)
        d = d[(d["GHI"] >= -4) & (d["GHI"] <= 1.5 * d["Sa"] * d["mu0"] ** 1.2 + 100)]
        n["N_limites_fisicos"] = len(d)
        d = d[(d["GHI"] >= -2) & (d["GHI"] <= 1.2 * d["Sa"] * d["mu0"] ** 1.2 + 50)].copy()
        n["N_eventos_raros"] = len(d)
        d.loc[d["GHI"] < 0, "GHI"] = 0.0
        d["Kt"] = d["GHI"] / d["H0h"]
        n["N_Kt_acima_1"] = int((d["Kt"] > 1).sum())
        n["N_Kt_carimbo_acima_1"] = int((d["GHI"] / d["H0h_carimbo"] > 1).sum())
        d = d[d["Kt"] <= ns.KT_MAX]
        n.update(N_final=len(d), Kt_medio=float(d["Kt"].mean()), CV=float(d["Kt"].std() / d["Kt"].mean()))
        return d, n

    ns.read_station, ns.exog_qc, ns.exog_features, ns.quality_control = (
        read_station, exog_qc, exog_features, quality_control
    )
    return ns


def pipeline():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "STATIONS") and hasattr(main_mod, "quality_control"):
        return main_mod
    try:
        import prev_v9
        if hasattr(prev_v9, "STATIONS"):
            return prev_v9
        sys.modules.pop("prev_v9", None)
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ""
    for c in (here, os.getcwd(), os.environ.get("PREV_OUT_DIR", ""), "/content/drive/MyDrive/prev_v9", "/content"):
        p = os.path.join(c, "prev_v9.py") if c else ""
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("prev_v9_file", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return _fallback_pv()


pv = pipeline()

for ttf in Path("/usr/share/fonts/truetype/liberation").glob("Liberation*.ttf"):
    font_manager.fontManager.addfont(str(ttf))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#009E73", "#1f1f1e", "#6b6a66", "#e4e3df"
SITES = [(sid, st["name"]) for sid, st in pv.STATIONS.items()]
NAMES = [n for _, n in SITES]
COLOR = dict(zip(NAMES, (BLUE, ORANGE)))
LABEL = {"Cercadinho": "Cercadinho (MG)", "Ponta Pora": "Ponta Porã (MS)"}
SITE = {"Cercadinho": "Cercadinho", "Ponta Pora": "Ponta Porã"}
UTC = {st["name"]: -st["tz"] for st in pv.STATIONS.values()}
PERS, BASE, POLY = pv.PERS, pv.BASE, "Emp: 3_polynomial"
ORDER = [POLY, "Emp: sigmoid_4p", "Emp: logistic_3p", "Emp: logistic_4p", "Emp: rational",
         "ML: ANN", "ML: SVM", "ML: XGBoost", "DL: LSTM+Att", "Hybrid: SVM+LSTM-res",
         "Ens: HetDS", "Ens: DSAG", "Ens: BMA", PERS]
NAME = {POLY: "Polinômio de terceiro grau", "Emp: sigmoid_4p": "Sigmoide de 4 parâmetros",
        "Emp: logistic_3p": "Logística de 3 parâmetros", "Emp: logistic_4p": "Logística de 4 parâmetros",
        "Emp: rational": "Racional", "ML: ANN": "MLP", "ML: SVM": "SVM", "ML: XGBoost": "XGBoost",
        "DL: LSTM+Att": "LSTM-Att", "Hybrid: SVM+LSTM-res": "SVM+LSTM-res", "Ens: HetDS": "HetDS",
        "Ens: DSAG": "DSAG", "Ens: BMA": "BMA", PERS: "Persistência"}
MCOL = {"ML: ANN": "#CC79A7", "ML: SVM": "#009E73", "ML: XGBoost": "#0072B2", "DL: LSTM+Att": "#D55E00",
        "Hybrid: SVM+LSTM-res": "#E69F00", "Ens: BMA": "#000000"}
MMARK = {"ML: ANN": "v", "ML: SVM": "s", "ML: XGBoost": "o", "DL: LSTM+Att": "^", "Hybrid: SVM+LSTM-res": "D",
         "Ens: BMA": "P"}
VNAME = {"temp": "Temperatura do ar", "temp_dew": "Ponto de orvalho", "rh": "Umidade relativa",
         "pres": "Pressão da estação", "wind_speed": "Velocidade do vento", "wind_gust": "Rajada máxima",
         "precip": "Precipitação", "amplitude_termica": "Amplitude térmica", "deficit_orvalho": "Déficit de orvalho",
         "pressao_vapor": "Pressão de vapor", "tendencia_pressao": "Tendência de pressão", "vento_sin": "Direção, seno",
         "vento_cos": "Direção, cosseno", "chuva_ocorrencia": "Ocorrência de chuva", "nenhuma": "Nenhuma"}
ROT = {"AMI x1": "AMI, 1\\textsuperscript{o} múltiplo", "AMI x2": "AMI, 2\\textsuperscript{o} múltiplo",
       "AMI x3": "AMI, 3\\textsuperscript{o} múltiplo", "PACF": "PACF", "ACF": "ACF",
       "referencia diurna": "referência diurna", "exploratoria": "exploratória"}
REG = ["Baixa", "Intermediária", "Alta"]
BINS = [-np.inf, 0.35, 0.65, np.inf]
HOR = pv.HORIZONS
WEEK = ("2025-11-17", "2025-11-24")


def rd(name, **kw):
    return pd.read_csv(os.path.join(D, name), sep=";", decimal=",", **kw)


def f(x, d=2):
    if pd.isna(x):
        return "--"
    s = f"{abs(x):.{d}f}".replace(".", "{,}")
    return f"$-{s}$" if x < 0 and float(f"{abs(x):.{d}f}") != 0 else s


def fh(x, d):
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{abs(v) if v == 0 else v:.{d}f}".replace(".", "{,}")
    return f"$-{s[1:]}$" if s.startswith("-") else s


def th(n):
    return f"{int(n):,d}".replace(",", ".")


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.1f}".replace(".", "{,}") + f"\\times10^{{{e}}}$"


def pk(p):
    return "$< 0{,}01$" if p <= 0.01 else "$> 0{,}10$" if p >= 0.1 else f(p, 3)


def save(name, rows):
    with open(os.path.join(OUT, "tabelas", name), "w", encoding="utf-8") as fh_:
        fh_.write("\n".join(rows) + "\n")


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def comma(fmt):
    return ticker.FuncFormatter(lambda x, _: fmt.format(x).replace(".", ",").replace("-", "−"))


def title(ax, txt):
    ax.set_title(txt, loc="left", fontsize=10, color=INK)


def savefig(fig, name, **kw):
    fig.savefig(os.path.join(OUT, "figuras", name), **kw)
    plt.close(fig)


def series():
    path = os.path.join(D, "phase1.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f_:
            return pickle.load(f_)["data"]
    out = {}
    for sid, name in SITES:
        raw, _ = pv.read_station(sid)
        raw = pv.exog_features(pv.exog_qc(raw)[0])
        out[name] = pv.quality_control(raw, pv.STATIONS[sid]["lat"], pv.STATIONS[sid]["lon"])[0]
    return out


def locate():
    args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
    d = args[0] if args and os.path.isdir(args[0]) else pv.OUT_DIR
    if d.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    return d, args[1] if len(args) > 1 else os.path.join(d, "figuras_tabelas")


def ls(t, y, w, chunk=1500):
    return np.concatenate([lombscargle(t, y, w[i:i + chunk], normalize=True) for i in range(0, len(w), chunk)])


def tab_mu0(mu):
    rows = []
    for e in NAMES:
        for _, x in mu[mu.Estacao == e].iterrows():
            rows.append(f"{SITE[e]} & {fh(x.mu0_min, 2)} & {fh(x.Zenital_max_graus, 1)} & {th(x.N_final)} & "
                        f"{fh(x.Kt_medio, 3)} & {fh(x.CV, 3)} \\\\")
        rows.append("\\midrule")
    save("tab_mu0.tex", rows[:-1])


def tab_caracterizacao(stats, train):
    s = stats.set_index("Estacao").loc[NAMES]
    c = lambda fn: " & ".join(fn(s.loc[e]) for e in NAMES)
    kp = {e: {k: kpss(train[e].Kt.values, regression=k, nlags="auto") for k in ("c", "ct")} for e in NAMES}
    rows = ["\\multicolumn{4}{l}{Descritores} \\\\",
            f"Registros & & {c(lambda r: th(r.N_treino))} \\\\",
            f"$K_t$ médio & & {c(lambda r: f(r.Kt_medio, 3))} \\\\",
            f"CV & & {c(lambda r: f(r.CV, 3))} \\\\", "\\midrule",
            "\\multicolumn{4}{l}{Estacionariedade, valor-p} \\\\",
            f"ADF & raiz unitária & {c(lambda r: sci(r.ADF_p))} \\\\",
            f"KPSS & nível constante & {c(lambda r: pk(r.KPSS_p_nivel))} \\\\",
            f"KPSS & tendência determinística & {c(lambda r: pk(r.KPSS_p_tendencia))} \\\\", "\\midrule",
            "\\multicolumn{4}{l}{Lomb-Scargle, potência normalizada} \\\\"]
    rows += [f"{p} h & eixo temporal real & {c(lambda r, p=p: f(r[f'LS_tempo_{p}h'], 3))} \\\\" for p in (24, 12, 8, 6)]
    rows += [f"{p} amostras & eixo amostral & {c(lambda r, p=p: sci(r[f'LS_amostral_{p}h']))} \\\\" for p in (24, 12)]
    rows += ["\\midrule", "\\multicolumn{4}{l}{Critérios de dimensionamento da janela} \\\\",
             f"AMI & primeiro mínimo local & {c(lambda r: str(int(r.AMI_primeiro_minimo)))} \\\\",
             f"PACF & corte pela banda de 95\\% & {c(lambda r: str(int(r.PACF_corte)))} \\\\",
             f"ACF & corte pela banda de 95\\% & {c(lambda r: str(int(r.ACF_corte)))} \\\\",
             f"Banda & limiar de 95\\% & {c(lambda r: f(r.Banda_95, 4))} \\\\",
             f"Referência diurna & mediana de amostras por dia & {c(lambda r: str(int(r.Amostras_por_dia)))} \\\\"]
    rows += [f"% KPSS {k}, {SITE[e]}: estatística {kp[e][k][0]:.3f}; críticos 10% {kp[e][k][3]['10%']:.3f}, "
             f"1% {kp[e][k][3]['1%']:.3f}" for e in NAMES for k in ("c", "ct")]
    save("tab_caracterizacao.tex", rows)


def tab_grade(lg):
    lab = lg.set_index(["Estacao", "Lag"]).Rotulos
    lags = sorted(lg.Lag.unique())
    rows = [f"{L} & {ROT[lab[(NAMES[0], L)]]} & {ROT[lab[(NAMES[1], L)]]} \\\\" for L in lags] + ["\\midrule"]
    cnt = lambda e, fn: str(sum(fn(lab[(e, L)]) for L in lags))
    for txt, fn in (("Ancoradas na AMI", lambda r: r.startswith("AMI")), ("Indicadas por critério linear", lambda r: r in ("PACF", "ACF")),
                    ("Referência diurna", lambda r: r == "referencia diurna"), ("Exploratórias", lambda r: r == "exploratoria")):
        rows.append(f"{txt} & {cnt(NAMES[0], fn)} & {cnt(NAMES[1], fn)} \\\\")
    save("tab_grade.tex", rows)


def tab_exog(scr, sel):
    for e, tag in zip(NAMES, ("cerc", "pp")):
        g = scr[(scr.Estacao == e) & (scr.Horizonte == 1)].set_index("Variavel")
        rows = [f"{VNAME[v]} & {fh(x.Pearson, 3)} & {fh(x.MI, 3)} & {fh(x.MI_excesso, 1)} & {fh(x.MI_residuo, 3)} & "
                f"{fh(x.MI_residuo_excesso, 1)} \\\\" for v, x in ((v, g.loc[v]) for v in pv.EXOG)]
        save(f"tab_exog_{tag}.tex", rows)
        rows = []
        for _, x in sel[sel.Estacao == e].iterrows():
            if x.Ordem == 0:
                rows.append(f"0 & Nenhuma & -- & -- & -- & {fh(1000 * x.MSE_val_interna, 3)} \\\\")
            else:
                rows.append(f"{int(x.Ordem)} & {VNAME[x.Variavel]} & {fh(x.Relevancia, 4)} & {fh(x.Redundancia, 4)} & "
                            f"{fh(x.Escore, 4)} & {fh(1000 * x.MSE_val_interna, 3)} \\\\")
        save(f"tab_trilha_{tag}.tex", rows)


def tab_particoes(m):
    rows = []
    for e in NAMES:
        for h in HOR:
            r = m[(m.Estacao == e) & (m.Horizonte == h)].iloc[0]
            rows.append(f"{SITE[e]} & {h}~h & {th(r.N_treino)} & {th(r.N_val_base)} & {th(r.N_val_meta)} & {th(r.N_teste)} \\\\")
        rows.append("\\midrule")
    save("tab_particoes.tex", rows[:-1])


def tab_ancoradas(m, anc):
    rows = []
    for e in NAMES:
        rows.append(f"\\multicolumn{{7}}{{l}}{{{SITE[e]}, defasagem de {anc[e]} amostras}} \\\\")
        g = m[(m.Estacao == e) & (m.Lag == anc[e]) & (m.Horizonte == 1)].set_index("Modelo")
        for k in ORDER:
            r = g.loc[k]
            rows.append(f"{NAME[k]} & {f(r.MBD)} & {f(r.MAD)} & {f(r.RMSD)} & {f(r.R2, 3)} & {f(r.dm, 3)} & {f(r.SS, 3)} \\\\")
        rows.append("\\midrule")
    save("tab_ancoradas_h1.tex", rows[:-1])
    rows = []
    for k in ORDER:
        cells = []
        for e in NAMES:
            for h in HOR[1:]:
                r = m[(m.Estacao == e) & (m.Lag == anc[e]) & (m.Horizonte == h) & (m.Modelo == k)].iloc[0]
                cells += [f(r.RMSD), f(r.SS, 3)]
        rows.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    save("tab_ancoradas_h36.tex", rows)


def tab_criterios(best, lg):
    lab = lg.set_index(["Estacao", "Lag"]).Rotulos.to_dict()
    rows = []
    for e in NAMES:
        rows.append(f"\\multicolumn{{5}}{{l}}{{{SITE[e]}}} \\\\")
        b = best[best.Estacao == e]
        mins = {h: b[b.Horizonte == h].RMSD.min() for h in HOR}
        mark = lambda v, h: "\\textsuperscript{a}" if np.isclose(v, mins[h]) else ""
        for key in ("AMI x1", "AMI x2", "AMI x3", "PACF", "ACF", "referencia diurna"):
            L = next(l for (s, l), v in lab.items() if s == e and v == key)
            cells = [f(v) + mark(v, h) for h in HOR for v in [b[(b.Lag == L) & (b.Horizonte == h)].RMSD.iloc[0]]]
            rows.append(f"{ROT[key][0].upper()}{ROT[key][1:]} & {L} & " + " & ".join(cells) + " \\\\")
        cells = []
        for h in HOR:
            bb = b[(b.Horizonte == h) & (b.Lag.map(lambda l: lab[(e, l)]) == "exploratoria")]
            r = bb.loc[bb.RMSD.idxmin()]
            cells.append(f"{f(r.RMSD)}{mark(r.RMSD, h)} ({int(r.Lag)})")
        rows.append("Melhor exploratória & & " + " & ".join(cells) + " \\\\")
        cells = []
        for h in HOR:
            s = b[b.Horizonte == h].RMSD
            cells.append(f"{f(s.max() - s.min())} ({f(100 * (s.max() - s.min()) / s.min(), 1)}\\%)")
        rows.append("Amplitude na grade & & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    save("tab_criterios.tex", rows[:-1])


def tab_vitorias(m):
    def wins(col, absval=False):
        d = m[m.Modelo != PERS].copy()
        d["c"] = d[col].abs() if absval else d[col]
        w = d.loc[d.groupby(["Estacao", "Lag", "Horizonte"]).c.idxmin()]
        return pd.crosstab(w.Modelo, w.Horizonte)
    W = {"RMSD": wins("RMSD"), "MAD": wins("MAD"), "MBD": wins("MBD", True)}
    rows = []
    for k in [k for k in ORDER if any(k in W[c].index for c in W)]:
        cells = []
        for c in ("RMSD", "MAD", "MBD"):
            v = [int(W[c].loc[k, h]) if (k in W[c].index and h in W[c].columns) else 0 for h in HOR]
            cells += [str(x) for x in v] + [str(sum(v))]
        rows.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    save("tab_vitorias.tex", rows)


def tab_mbd(m):
    rows = [f"{NAME[k]} & " + " & ".join(f(m[(m.Estacao == e) & (m.Horizonte == h) & (m.Modelo == k)].MBD.mean())
                                           for e in NAMES for h in HOR) + " \\\\" for k in ORDER]
    save("tab_mbd.tex", rows)


def tab_regimes(P, anc, bm):
    rows, cal = [], []
    for e in NAMES:
        rows.append(f"\\multicolumn{{9}}{{l}}{{{SITE[e]}, defasagem de {anc[e]} amostras, melhor modelo: {NAME[bm[e]]}}} \\\\")
        cal.append(f"\\multicolumn{{8}}{{l}}{{{SITE[e]}, defasagem de {anc[e]} amostras, melhor modelo: {NAME[bm[e]]}}} \\\\")
        mods = (bm[e], POLY, PERS)
        for h in HOR:
            g = P[(P.Estacao == e) & (P.Lag == anc[e]) & (P.Horizonte == h)]
            obs = pd.cut(g.Kt_real, bins=BINS, labels=[0, 1, 2], right=False).astype(int)
            for ri in range(3):
                s = g[obs == ri]
                err = {k: s[f"Prev: {k}"] - s.GHI_real for k in mods}
                hl = f"{h}~h" if ri == 0 else ""
                rows.append(f"{hl} & {REG[ri]} & {f(100 * len(s) / len(g), 1)} & "
                            + " & ".join(f(err[k].mean()) for k in mods) + " & "
                            + " & ".join(f(np.sqrt((err[k] ** 2).mean())) for k in mods) + " \\\\")
                frac, mbd = [], []
                for k in mods:
                    pred = pd.cut(g[f"Prev: {k}"] / g.H0h_alvo, bins=BINS, labels=[0, 1, 2], right=False).astype(int)
                    sp = g[pred == ri]
                    frac.append(f(100 * len(sp) / len(g), 1))
                    mbd.append(f((sp[f"Prev: {k}"] - sp.GHI_real).mean()) if len(sp) else "--")
                cal.append(f"{hl} & {REG[ri]} & " + " & ".join(frac + mbd) + " \\\\")
            if h != HOR[-1]:
                rows.append("\\addlinespace")
                cal.append("\\addlinespace")
        rows.append("\\midrule")
        cal.append("\\midrule")
    save("tab_regimes.tex", rows[:-1])
    save("tab_calibracao.tex", cal[:-1])


def tab_origem_movel(m, mr, anc):
    rows = []
    for e in NAMES:
        rows.append(f"\\multicolumn{{7}}{{l}}{{{SITE[e]}, defasagem de {anc[e]} amostras}} \\\\")
        tabs = {}
        for h in HOR:
            g = mr[(mr.Estacao == e) & (mr.Lag == anc[e]) & (mr.Horizonte == h)]
            t = g.pivot_table(index="Modelo", columns="Dobra", values="RMSD")
            t["mean"], t["sd"] = t.mean(axis=1), t.std(axis=1, ddof=1)
            t["fixed"] = m[(m.Estacao == e) & (m.Lag == anc[e]) & (m.Horizonte == h)].set_index("Modelo").RMSD
            tabs[h] = t
        for k in ORDER:
            cells = [c for h in HOR for c in (f(tabs[h].loc[k, "fixed"]), f"{f(tabs[h].loc[k, 'mean'], 1)} ({f(tabs[h].loc[k, 'sd'], 1)})")]
            rows.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
        rho = [spearmanr(tabs[h].drop(PERS)["fixed"], tabs[h].drop(PERS)["mean"]).statistic for h in HOR]
        rows.append("\\addlinespace")
        rows.append(f"Spearman, {len(ORDER) - 1} modelos & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{f(r, 3)}}}" for r in rho) + " \\\\")
        rows.append("\\midrule")
    save("tab_origem_movel.tex", rows[:-1])


def fig_fase(ser, phase):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for c, e in enumerate(NAMES):
        s = ser[e]
        prof = s.assign(c=s.index.hour + s.index.minute / 60 - pv.ACC_H / 2).groupby("c")[["GHI", "H0h", "H0h_carimbo"]].mean()
        cen = {k: float((prof.index.values * prof[k].values).sum() / prof[k].sum()) for k in prof}
        ax = axes[c]
        ax.plot(prof.index, prof.GHI, "-o", color=INK, ms=3, lw=1.2, label="GHI medida")
        ax.plot(prof.index, prof.H0h, "-s", color=BLUE, ms=3, lw=1.2, label="$\\overline{H}_{0h}$, integrada na hora")
        ax.plot(prof.index, prof.H0h_carimbo, "--^", color=ORANGE, ms=3, lw=1.0, label="$H_{0h}$ na marca temporal")
        for k, col in (("GHI", INK), ("H0h", BLUE), ("H0h_carimbo", ORANGE)):
            ax.axvline(cen[k], color=col, lw=0.8, ls=":")
        ax.axvline(phase.loc[e, "Meio_dia_solar_UTC"], color=GREEN, lw=1.0, ls="-.", label="meio-dia solar")
        ax.set_xlabel("Hora do dia (UTC, centro da hora de acumulação)")
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        ax.set_xlim(8, 23)
        ax.set_xticks(range(9, 23, 2))
        title(ax, f"({'ab'[c]}) {LABEL[e]}")
    axes[0].set_ylabel("Irradiância média (W/m$^{2}$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "fig_fase.pdf")


def fig_periodograma(train):
    marks = np.array([24.0, 12.0, 8.0, 6.0])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for c, e in enumerate(NAMES):
        tr = train[e]
        v0 = tr.Kt.values - tr.Kt.values.mean()
        hours = (tr.index - tr.index[0]).total_seconds().values / 3600.0
        freqs = np.sort(np.concatenate([np.linspace(1 / 30, 1 / 4, 24000), 1 / marks]))
        pw = ls(hours, v0, 2 * np.pi * freqs)
        chk = lombscargle(hours, v0, 2 * np.pi / marks, normalize=True)
        ax = axes[0, c]
        ax.plot(1 / freqs, pw, color=COLOR[e], lw=1.1)
        for p in marks:
            ax.axvline(p, color=MUTED, lw=0.7, ls=":")
        ax.plot(marks, chk, "o", ms=3.2, mfc="white", mec=INK, mew=0.9, zorder=5)
        lines = [f"{int(p)} h: {val:.3f}".replace(".", ",") for p, val in zip(marks, chk)]
        ax.text(0.98, 0.97, "\n".join(["Potência nos harmônicos"] + lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, linewidth=0.6))
        ax.set_ylim(0, pw.max() * 1.18)
        ax.set_xlabel("Período (h de relógio)")
        ax.set_ylabel("Potência normalizada")
        ax.yaxis.set_major_formatter(comma("{:.2f}"))
        ax.set_xticks([6, 8, 12, 18, 24, 30])
        title(ax, f"({'ab'[c]}) {LABEL[e]}, eixo temporal real")
        freqs_s = np.linspace(1 / 30, 1 / 4, 24000)
        pws = ls(np.arange(len(v0), dtype=float), v0, 2 * np.pi * freqs_s)
        ax = axes[1, c]
        ax.plot(1 / freqs_s, pws, color=COLOR[e], lw=1.1)
        for p in (24, 12, 11):
            ax.axvline(p, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("Período (amostras diurnas)")
        ax.set_ylabel("Potência normalizada")
        ax.yaxis.set_major_formatter(comma("{:.3f}"))
        ax.set_xticks([6, 8, 11, 12, 18, 24, 30])
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment({"11": "right", "12": "left"}.get(lab.get_text(), "center"))
        title(ax, f"({'cd'[c]}) {LABEL[e]}, eixo amostral")
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    savefig(fig, "fig_periodograma.pdf")


def fig_memoria(train, ami, stats):
    s = stats.set_index("Estacao")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for r, e in enumerate(NAMES):
        v = train[e].Kt.values
        band = 1.96 / np.sqrt(len(v))
        lags = np.arange(1, pv.MAX_SCAN_LAG + 1)
        a = ami[ami.Estacao == e].sort_values("Defasagem")
        lstar, cut_p, cut_a = (int(s.loc[e, k]) for k in ("AMI_primeiro_minimo", "PACF_corte", "ACF_corte"))
        ax = axes[r, 0]
        ax.plot(a.Defasagem, a.AMI, color=COLOR[e], lw=1.2)
        y0 = a[a.Defasagem == lstar].AMI.iloc[0]
        ax.plot([lstar], [y0], "o", color=INK, ms=4, zorder=5)
        ax.axvline(lstar, color=INK, lw=0.8, ls=":")
        ax.annotate(f"$\\ell^{{*}}={lstar}$", (lstar, 0.6 * a.AMI.max()), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)
        ax.set_ylabel("AMI (nats)")
        ax.yaxis.set_major_formatter(comma("{:.1f}"))
        title(ax, f"({'abcdef'[3 * r]}) {LABEL[e]}, AMI")
        for j, (coef, cut, lab) in enumerate(((acf(v, nlags=pv.MAX_SCAN_LAG, fft=True)[1:], cut_a, "ACF"),
                                             (pacf(v, nlags=pv.MAX_SCAN_LAG, method="ywm")[1:], cut_p, "PACF")), 1):
            ax = axes[r, j]
            ax.plot(lags, coef, color=COLOR[e], lw=1.2)
            ax.axhline(band, color=MUTED, lw=0.8, ls="--")
            ax.axhline(-band, color=MUTED, lw=0.8, ls="--")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.axvline(cut, color=INK, lw=0.8, ls=":")
            right = cut < pv.MAX_SCAN_LAG / 2
            ax.annotate(f"corte = {cut}", (cut, 0.6), xytext=(4 if right else -4, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center", fontsize=9)
            ax.set_ylabel(lab)
            ax.yaxis.set_major_formatter(comma("{:.1f}"))
            title(ax, f"({'abcdef'[3 * r + j]}) {LABEL[e]}, {lab}")
        axes[r, 2].set_ylim(-0.15, 0.75)
    for ax in axes[1]:
        ax.set_xlabel("Defasagem (amostras diurnas)")
        ax.set_xlim(0, pv.MAX_SCAN_LAG + 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    savefig(fig, "fig_memoria.pdf")


def fig_kt_distribuicao(ser, splits):
    with plt.rc_context({"xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5}):
        fig, ax = plt.subplots(figsize=(5.0, 3.1))
        bins = np.arange(0.0, 1.0 + 1e-9, 0.025)
        for e in NAMES:
            t0 = pd.Timestamp(splits[(splits.Estacao == e) & (splits.Desenho == "fixa")].test_start.iloc[0])
            kt = ser[e].loc[ser[e].index >= t0, "Kt"].values
            ax.hist(kt, bins=bins, density=True, histtype="stepfilled", color=COLOR[e], alpha=0.18, linewidth=0)
            ax.hist(kt, bins=bins, density=True, histtype="step", color=COLOR[e], linewidth=1.1,
                    label=f"{LABEL[e]}, n = {th(len(kt))}")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("índice de claridade $K_t$")
        ax.set_ylabel("densidade")
        ax.xaxis.set_major_formatter(comma("{:.1f}"))
        ax.yaxis.set_major_formatter(comma("{:.1f}"))
        ax.legend(loc="upper left", handlelength=1.6)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        savefig(fig, "fig_kt_distribuicao.pdf")


def fig_superficie(best, lg, lags):
    lab = lg.set_index(["Estacao", "Lag"]).Rotulos.to_dict()
    mk = {"AMI": ("o", 6.0), "PACF": ("^", 6.5), "ACF": ("s", 5.5), "referencia diurna": ("D", 5.0), "exploratoria": ("o", 3.0)}
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.55), sharey=True)
    x = np.arange(len(lags))
    for ax, h, tag in zip(axes, HOR, "abc"):
        for e in NAMES:
            s = best[(best.Estacao == e) & (best.Horizonte == h)].set_index("Lag").RMSD.reindex(lags)
            y = 100 * (s - s.min()) / s.min()
            ax.plot(x, y.values, color=COLOR[e], lw=1.1, zorder=2)
            for xi, L in zip(x, lags):
                key = "AMI" if lab[(e, L)].startswith("AMI") else lab[(e, L)]
                ax.plot(xi, y.loc[L], marker=mk[key][0], ms=mk[key][1], mfc=COLOR[e] if key != "exploratoria" else "white",
                        mec=COLOR[e], mew=1.0, zorder=3, ls="none")
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in lags], fontsize=7.5)
        ax.set_xlabel("defasagem (amostras)")
        title(ax, f"({tag}) {h} h")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.2, 4.0)
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
    axes[0].set_ylabel("excesso de RMSD (%)")
    handles = [Line2D([], [], color=COLOR[e], lw=1.1, label=LABEL[e]) for e in NAMES]
    handles += [Line2D([], [], color=MUTED, marker=m, ms=s, mfc=c, ls="none", label=t) for m, s, c, t in
                (("o", 6, MUTED, "múltiplo da AMI"), ("^", 6.5, MUTED, "PACF"), ("s", 5.5, MUTED, "ACF"),
                 ("D", 5, MUTED, "referência diurna"), ("o", 3, "white", "exploratória"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14), handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout()
    savefig(fig, "fig_superficie.pdf")


def fig_superficie_modelos(m, lags):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    x = np.arange(len(lags))
    for ax, e, tag in zip(axes, NAMES, "ab"):
        g = m[(m.Estacao == e) & (m.Horizonte == 1)]
        for k in MCOL:
            s = g[g.Modelo == k].set_index("Lag").RMSD.reindex(lags)
            ax.plot(x, s.values, color=MCOL[k], lw=1.0, marker=MMARK[k], ms=3.8, label=NAME[k])
        for v, lst, txt in ((g[g.Modelo == PERS].RMSD.iloc[0], "--", "persistência"), (g[g.Modelo.str.startswith("Emp")].RMSD.min(), ":", "empírico")):
            ax.axhline(v, color=MUTED, lw=1.0, ls=lst)
            ax.text(x[-1] + 0.35, v, txt, va="center", ha="left", fontsize=8, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in lags], fontsize=7.5)
        ax.set_xlabel("defasagem (amostras)")
        title(ax, f"({tag}) {LABEL[e]}")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        ax.set_xlim(-0.4, len(lags) - 0.6)
    axes[0].set_ylabel("RMSD (W/m²)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.09), handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(w_pad=3.0)
    savefig(fig, "fig_superficie_modelos.pdf")


def fig_previsto_real(P, anc, bm):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), gridspec_kw=dict(width_ratios=[1.75, 1.0]))
    for row, e in enumerate(NAMES):
        g = P[(P.Estacao == e) & (P.Lag == anc[e]) & (P.Horizonte == 1)].copy()
        g["loc"] = pd.to_datetime(g.Alvo) - pd.Timedelta(hours=UTC[e])
        w = g[(g["loc"] >= WEEK[0]) & (g["loc"] < WEEK[1])].sort_values("loc")
        t = w["loc"].values
        gap = np.r_[False, np.diff(t).astype("timedelta64[m]").astype(float) > 90]

        def broken(v):
            tt, vv = [], []
            for ti, vi, gi in zip(t, v.astype(float), gap):
                if gi:
                    tt.append(ti)
                    vv.append(np.nan)
                tt.append(ti)
                vv.append(vi)
            return np.array(tt), np.array(vv)
        ax = axes[row, 0]
        for col, c, lw, lst, lab in ((w.GHI_real, INK, 1.2, "-", "medido"), (w[f"Prev: {bm[e]}"], COLOR[e], 1.1, "-", NAME[bm[e]]),
                                     (w[f"Prev: {PERS}"], MUTED, 0.9, "--", "persistência")):
            ax.plot(*broken(col.values), color=c, lw=lw, ls=lst, label=lab)
        ax.set_xlim(pd.Timestamp(WEEK[0]), pd.Timestamp(WEEK[1]))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.set_ylabel("GHI (W/m²)")
        ax.set_xlabel("dia de novembro de 2025")
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        ax.set_ylim(0, 1400)
        ax.set_yticks([0, 250, 500, 750, 1000])
        ax.grid(axis="x", visible=False)
        title(ax, f"({'ac'[row]}) {LABEL[e]}")
        ax.legend(loc="upper center", ncol=3, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
        ax = axes[row, 1]
        ax.scatter(g.GHI_real, g[f"Prev: {bm[e]}"], s=2.5, color=COLOR[e], alpha=0.25, linewidths=0, rasterized=True)
        ax.plot([0, 1150], [0, 1150], color=INK, lw=0.8)
        ax.set_xlim(0, 1150)
        ax.set_ylim(0, 1150)
        ax.set_xlabel("medido (W/m²)")
        ax.set_ylabel(f"previsto, {NAME[bm[e]]} (W/m²)")
        ax.xaxis.set_major_formatter(comma("{:.0f}"))
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        ax.set_aspect("equal")
        title(ax, f"({'bd'[row]}) {LABEL[e]}")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    savefig(fig, "fig_previsto_real.pdf", dpi=300)


def fig_kappa(bk, anc):
    bk = bk[bk.Desenho == "fixa"]
    kap = list(pv.BMA_KAPPAS)
    ls_ = {1: ("-", "o"), 3: ("--", "s"), 6: (":", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    x = np.arange(len(kap))
    for ax, e, tag in zip(axes, NAMES, "ab"):
        for h in HOR:
            g = bk[(bk.Estacao == e) & (bk.Lag == anc[e]) & (bk.Horizonte == h)].set_index("kappa").RMSD
            ax.plot(x, (g.reindex(kap) - g.loc[pv.BMA_KAPPA]).values, color=COLOR[e], ls=ls_[h][0], marker=ls_[h][1], ms=4, lw=1.1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["∞" if np.isinf(k) else f"{k:g}" for k in kap])
        ax.set_xlabel("fator de escala κ")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        title(ax, f"({tag}) {LABEL[e]}")
    axes[0].set_ylabel(f"RMSD − RMSD(κ = {pv.BMA_KAPPA:g}) (W/m²)")
    handles = [Line2D([], [], color=MUTED, ls=ls_[h][0], marker=ls_[h][1], ms=4, lw=1.1, label=f"{h} h") for h in HOR]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handlelength=2.4)
    fig.tight_layout()
    savefig(fig, "fig_kappa.pdf")


def fig_origem_movel(mr, anc):
    mods = [("Ens: BMA", "#000000", "-", "P"), ("ML: XGBoost", "#0072B2", "-", "o"), (PERS, MUTED, "--", "s")]
    folds = sorted(mr.Dobra.unique())
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.3), sharex=True)
    for i, e in enumerate(NAMES):
        for j, h in enumerate(HOR):
            ax = axes[i, j]
            g = mr[(mr.Estacao == e) & (mr.Lag == anc[e]) & (mr.Horizonte == h)]
            for k, c, lst, mk in mods:
                s = g[g.Modelo == k].set_index("Dobra").RMSD.reindex(folds)
                ax.plot(s.index, s.values, color=c, ls=lst, marker=mk, ms=3.8, lw=1.0, label=NAME[k].lower() if k == PERS else NAME[k])
            ax.set_xticks(folds)
            ax.yaxis.set_major_formatter(comma("{:.0f}"))
            ax.grid(axis="x", visible=False)
            title(ax, f"({'abcdef'[3 * i + j]}) {SITE[e]}, {h} h")
            if i == 1:
                ax.set_xlabel("dobra")
            if j == 0:
                ax.set_ylabel("RMSD (W/m²)")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savefig(fig, "fig_origem_movel.pdf")


def main():
    global D, OUT
    D, OUT = locate()
    for sub in ("tabelas", "figuras"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    style()
    stats, splits, lg, ami = rd("timeseries_stats.csv"), rd("splits.csv"), rd("lag_grid.csv"), rd("ami_curve.csv")
    m, mr, bk = rd("metrics_fixed.csv"), rd("metrics_rolling.csv"), rd("bma_kappa.csv")
    anc = stats.set_index("Estacao").AMI_primeiro_minimo.astype(int).to_dict()
    lags = sorted(lg.Lag.unique())
    best = m[m.Modelo != PERS].loc[lambda d: d.groupby(["Estacao", "Lag", "Horizonte"]).RMSD.idxmin()]
    bm = {e: best[(best.Estacao == e) & (best.Lag == anc[e]) & (best.Horizonte == 1)].Modelo.iloc[0] for e in NAMES}
    cols = ["Estacao", "Lag", "Horizonte", "Alvo", "GHI_real", "Kt_real", "H0h_alvo"] + [f"Prev: {k}" for k in {*bm.values(), POLY, PERS}]
    P = rd("predictions_fixed.csv", usecols=cols)
    ser = series()
    train = {e: ser[e][ser[e].index < pd.Timestamp(splits[(splits.Estacao == e) & (splits.Desenho == "fixa")].train_end.iloc[0])]
             for e in NAMES}
    tab_mu0(rd("mu0_sensitivity.csv"))
    tab_caracterizacao(stats, train)
    tab_grade(lg)
    tab_exog(rd("exog_screening.csv"), rd("exog_selection.csv"))
    tab_particoes(m)
    tab_ancoradas(m, anc)
    tab_criterios(best, lg)
    tab_vitorias(m)
    tab_mbd(m)
    tab_regimes(P, anc, bm)
    tab_origem_movel(m, mr, anc)
    fig_fase(ser, rd("phase_check.csv").set_index("Estacao"))
    fig_periodograma(train)
    fig_memoria(train, ami, stats)
    fig_kt_distribuicao(ser, splits)
    fig_superficie(best, lg, lags)
    fig_superficie_modelos(m, lags)
    fig_previsto_real(P, anc, bm)
    fig_kappa(bk, anc)
    fig_origem_movel(mr, anc)
    print(f"Concluído: 9 figuras em {os.path.join(OUT, 'figuras')} e 16 tabelas em {os.path.join(OUT, 'tabelas')}")


if __name__ == "__main__":
    main()
