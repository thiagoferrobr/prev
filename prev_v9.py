"""
Previsão horária de GHI nas estações INMET F501 (Cercadinho) e A703 (Ponta Porã).

Colab: Ambiente de execução > Alterar o tipo de ambiente > GPU. Coloque os CSV anuais do INMET (2023 a 2025)
em MyDrive/dados_inmet ou deixe a pasta ausente para baixá-los do portal. Execute com %run prev_v9.py.
As saídas ficam em MyDrive/prev_v9; ao reexecutar, as células concluídas são retomadas.
"""
import io
import os
import pickle
import shutil
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.signal import lombscargle
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

try:
    from google.colab import drive as gdrive
    IN_COLAB = True
except ImportError:
    gdrive, IN_COLAB = None, False

SMOKE_TEST = False
RUN_TAG = "v9_teste" if SMOKE_TEST else "v9"

DRIVE = "/content/drive/MyDrive"
DATA_DIR = os.environ.get("PREV_DATA_DIR", os.path.join(DRIVE, "dados_inmet") if IN_COLAB else "dados_inmet")
OUT_DIR = os.environ.get("PREV_OUT_DIR", os.path.join(DRIVE, f"prev_{RUN_TAG}") if IN_COLAB else f"prev_{RUN_TAG}")
WORK_DIR = f"/content/prev_{RUN_TAG}" if IN_COLAB else f"{OUT_DIR}_work"

STATIONS = {"F501": dict(name="Cercadinho", lat=-19.98, lon=-43.95861, tz=-3),
            "A703": dict(name="Ponta Pora", lat=-22.5525, lon=-55.7164, tz=-4)}
YEARS = (2023, 2024, 2025)

ISC = 1361.0
ACC_H = 1.0
SUBSTEPS = 60
MU0_MIN = 0.10
MU0_GRID = (0.01, 0.05, 0.10, 0.15, 0.20)
KT_MAX = 1.0
PHASE_TOL_ANALYTIC = 0.10
PHASE_TOL_EMPIRICAL = 0.25
TEST_LEVEL = 0.05

HORIZONS = (1, 3, 6)
SHORT_LAGS = ()
AMI_MULTIPLES = 3
AMI_BINS = 16
MAX_SCAN_LAG = 72
N_CAL = 5

TRAIN_END, VAL_BASE_END, TEST_START = 0.50, 0.50 + 1.0 / 12.0, 2.0 / 3.0
ROLL_FOLDS, ROLL_STEP, ROLL_WINDOW, ROLL_TEST = 5, 0.10, 0.50, 0.10

EXOG = ["temp", "temp_dew", "rh", "pres", "wind_speed", "wind_gust", "precip",
        "amplitude_termica", "deficit_orvalho", "pressao_vapor", "tendencia_pressao",
        "vento_sin", "vento_cos", "chuva_ocorrencia"]
EXOG_PERM, EXOG_MIN_Z, EXOG_MIN_AVAIL, EXOG_MAX, EXOG_W, EXOG_INNER = 20, 3.0, 0.98, 5, 1.0, 0.75

SVM_GRID = {"C": [1, 10, 100], "gamma": ["scale", 0.05, 0.2], "epsilon": [0.001, 0.01]}
MLP_GRID = {"hidden_layer_sizes": [(50,), (100,), (50, 25)], "alpha": [1e-4, 1e-3]}
XGB_GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1]}
SVM_TUNE_N, OOF_FOLDS = 2500, 3
EPOCHS, MLP_EPOCHS, BATCH, PATIENCE = 80, 300, 64, 12
LSTM_UNITS, DENSE_UNITS, DROPOUT, LR = 32, 16, 0.2, 1e-3
HETDS_K, BMA_KAPPA, BMA_KAPPAS = 7, 5.0, (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf)
R2_MIN, PERS_RATIO_MAX, FAIL_FRAC = 0.40, 1.10, 0.50
SEED = 42
FLUSH_EVERY = 5

if SMOKE_TEST:
    HORIZONS, SHORT_LAGS, ROLL_FOLDS, EPOCHS, MLP_EPOCHS, EXOG_PERM = (1, 6), (1,), 2, 3, 5, 3

PERS = "Pers: Smart"
BASE = ["ML: ANN", "ML: SVM", "ML: XGBoost", "DL: LSTM+Att", "Hybrid: SVM+LSTM-res"]
CSV = dict(sep=";", decimal=",", index=False)

COLS = {"GHI": ("RADIACAO GLOBAL",), "precip": ("PRECIPITA",),
        "pres": ("PRESSAO ATMOSFERICA AO NIVEL",), "temp": ("TEMPERATURA DO AR",),
        "temp_dew": ("TEMPERATURA DO PONTO DE ORVALHO",),
        "temp_max": ("TEMPERATURA MÁXIMA", "TEMPERATURA MAXIMA"),
        "temp_min": ("TEMPERATURA MÍNIMA", "TEMPERATURA MINIMA"),
        "rh": ("UMIDADE RELATIVA DO AR",), "wind_dir": ("VENTO, DIRE",),
        "wind_gust": ("VENTO, RAJADA",), "wind_speed": ("VENTO, VELOCIDADE",)}
RANGES = {"precip": (0, 200), "pres": (600, 1100), "temp": (-20, 55), "temp_dew": (-30, 40),
          "temp_max": (-20, 55), "temp_min": (-20, 55), "rh": (0, 100), "wind_dir": (0, 360),
          "wind_gust": (0, 75), "wind_speed": (0, 60)}


def mount():
    if IN_COLAB and not os.path.isdir(DRIVE):
        gdrive.mount("/content/drive")
    for d in (OUT_DIR, os.path.join(OUT_DIR, "cells"), os.path.join(WORK_DIR, "cells")):
        os.makedirs(d, exist_ok=True)
    for f in os.listdir(os.path.join(OUT_DIR, "cells")):
        dst = os.path.join(WORK_DIR, "cells", f)
        if f.endswith(".pkl") and not os.path.isfile(dst):
            shutil.copy2(os.path.join(OUT_DIR, "cells", f), dst)


def flush():
    if IN_COLAB:
        gdrive.flush_and_unmount()
        gdrive.mount("/content/drive")


def to_drive(src, dst):
    for _ in range(3):
        try:
            shutil.copy2(src, dst)
            return
        except OSError:
            time.sleep(15)
            if IN_COLAB and not os.path.isdir(DRIVE):
                gdrive.mount("/content/drive")
    shutil.copy2(src, dst)


def save_cell(job, obj):
    path = os.path.join(WORK_DIR, "cells", f"{job}.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)
    to_drive(path, os.path.join(OUT_DIR, "cells", f"{job}.pkl"))


def done(job):
    return os.path.isfile(os.path.join(WORK_DIR, "cells", f"{job}.pkl"))


def parse_inmet(src):
    raw = pd.read_csv(src, sep=";", skiprows=8, encoding="latin-1", on_bad_lines="skip")
    raw.columns = [c.strip().upper() for c in raw.columns]
    date = raw[next(c for c in raw.columns if c.startswith("DATA"))].astype(str).str.replace("/", "-")
    hour = (raw[next(c for c in raw.columns if c.startswith("HORA"))].astype(str)
            .str.replace(":", "").str.replace("UTC", "").str.strip().str.zfill(4).str[:2])
    out = pd.DataFrame({"t": pd.to_datetime(date + " " + hour + ":00", errors="coerce")})
    for key, frags in COLS.items():
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
    names = os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []
    return sorted(os.path.join(DATA_DIR, f) for f in names
                  if f.upper().endswith(".CSV") and f"_{sid}_" in f.upper() and any(str(y) in f for y in YEARS))


def fetch_portal():
    os.makedirs(DATA_DIR, exist_ok=True)
    for y in YEARS:
        r = requests.get(f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{y}.zip", timeout=1800)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for n in z.namelist():
                if n.upper().endswith(".CSV") and any(f"_{sid}_" in n.upper() for sid in STATIONS):
                    with open(os.path.join(DATA_DIR, os.path.basename(n)), "wb") as f:
                        f.write(z.read(n))


def read_station(sid):
    if not station_files(sid):
        fetch_portal()
    files = station_files(sid)
    d = pd.concat(parse_inmet(f) for f in files).drop_duplicates("t").set_index("t").sort_index()
    d = d[(d.index.year >= min(YEARS)) & (d.index.year <= max(YEARS))]
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h")), header_coords(files[0])


def exog_qc(d):
    d, rows = d.copy(), []
    for c, (lo, hi) in RANGES.items():
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
    for c in EXOG:
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
    mu_mid, e0 = spencer(doy, end - ACC_H / 2, lat, lon)
    frac = (np.arange(SUBSTEPS) + 0.5) / SUBSTEPS
    mu_s, e_s = spencer(doy[None, :], end[None, :] - ACC_H * (1 - frac[:, None]), lat, lon)
    mu_end, e_end = spencer(doy, end, lat, lon)
    return mu_mid, ISC * e0, ISC * (e_s * mu_s).mean(axis=0), ISC * e_end * mu_end


def quality_control(raw, lat, lon, mu_min=MU0_MIN):
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
    d = d[d["Kt"] <= KT_MAX]
    n.update(N_final=len(d), Kt_medio=float(d["Kt"].mean()), CV=float(d["Kt"].std() / d["Kt"].mean()))
    return d, n


def phase_check(d, lon, name):
    coord = d.index.hour + d.index.minute / 60 - ACC_H / 2
    prof = d.assign(c=coord).groupby("c")[["GHI", "H0h", "H0h_carimbo"]].mean()
    cen = {k: float((prof.index.values * prof[k].values).sum() / prof[k].sum()) for k in prof}
    eot = equation_of_time(2 * np.pi * (d.index.dayofyear.values - 1) / 365)
    noon = float(np.median(12 - lon / 15 - eot / 60))
    row = dict(Estacao=name, Centroide_GHI_UTC=cen["GHI"], Centroide_H0h_UTC=cen["H0h"],
               Meio_dia_solar_UTC=noon, Desvio_analitico_h=cen["H0h"] - noon,
               Desvio_empirico_h=cen["GHI"] - cen["H0h"],
               Desvio_empirico_carimbo_h=cen["GHI"] - cen["H0h_carimbo"])
    if abs(row["Desvio_analitico_h"]) > PHASE_TOL_ANALYTIC or abs(row["Desvio_empirico_h"]) > PHASE_TOL_EMPIRICAL:
        raise AssertionError(f"{name}: fase inconsistente {row}")
    return row


def mi(x, y):
    return float(mutual_info_regression(np.asarray(x, float).reshape(-1, 1), np.asarray(y, float),
                                        random_state=SEED)[0])


def ami(x, y):
    q = lambda v: np.searchsorted(np.quantile(v, np.linspace(0, 1, AMI_BINS + 1)[1:-1]), v, side="right")
    j = np.zeros((AMI_BINS, AMI_BINS))
    np.add.at(j, (q(x), q(y)), 1)
    j /= j.sum()
    p, m = np.outer(j.sum(1), j.sum(0)), j > 0
    return float((j[m] * np.log(j[m] / p[m])).sum())


def cutoff(coef, band):
    last = None
    for k in range(1, len(coef)):
        if abs(coef[k]) > band:
            last = k
        else:
            break
    return last, last == len(coef) - 1


def characterize(tr, tz, name):
    v = tr["Kt"].values
    per = np.array([24.0, 12.0, 8.0, 6.0])
    hours = (tr.index - tr.index[0]).total_seconds().values / 3600
    ls_t = lombscargle(hours, v - v.mean(), 2 * np.pi / per, normalize=True)
    ls_s = lombscargle(np.arange(len(v), dtype=float), v - v.mean(), 2 * np.pi / per, normalize=True)
    curve = [ami(v[:-k], v[k:]) for k in range(1, MAX_SCAN_LAG + 1)]
    l_star = next((i + 1 for i in range(1, len(curve) - 1) if curve[i - 1] > curve[i] < curve[i + 1]), MAX_SCAN_LAG)
    band = 1.96 / np.sqrt(len(v))
    pc, ps = cutoff(pacf(v, nlags=MAX_SCAN_LAG, method="ywm"), band)
    ac, as_ = cutoff(acf(v, nlags=MAX_SCAN_LAG, fft=True), band)
    adf_p = adfuller(v, autolag="AIC")[1]
    kpss_c, kpss_ct = kpss(v, regression="c", nlags="auto")[1], kpss(v, regression="ct", nlags="auto")[1]
    daily = int(np.median(tr.groupby((tr.index + pd.Timedelta(hours=tz)).normalize()).size()))
    stats = dict(Estacao=name, N_treino=len(v), Inicio=tr.index[0], Fim=tr.index[-1],
                 Kt_medio=float(v.mean()), CV=float(v.std() / v.mean()), ADF_p=adf_p,
                 KPSS_p_nivel=kpss_c, KPSS_p_tendencia=kpss_ct,
                 Estacionaria_em_nivel=bool(adf_p < TEST_LEVEL and kpss_c >= TEST_LEVEL),
                 **{f"LS_tempo_{int(p)}h": float(x) for p, x in zip(per, ls_t)},
                 **{f"LS_amostral_{int(p)}h": float(x) for p, x in zip(per, ls_s)},
                 AMI_primeiro_minimo=l_star, PACF_corte=pc, PACF_saturada=ps, ACF_corte=ac,
                 ACF_saturada=as_, Banda_95=band, Amostras_por_dia=daily)
    return stats, pd.DataFrame({"Estacao": name, "Defasagem": np.arange(1, MAX_SCAN_LAG + 1), "AMI": curve})


def screen(tr, l_star, name):
    kt = tr["Kt"]
    lags = pd.concat({i: kt.shift(i) for i in range(l_star)}, axis=1).values
    rng, rows = np.random.default_rng(SEED), []
    for h in HORIZONS:
        y = kt.reindex(tr.index + pd.Timedelta(hours=h)).values
        ok = np.isfinite(lags).all(axis=1) & np.isfinite(y)
        res = np.full(len(y), np.nan)
        res[ok] = y[ok] - Ridge(alpha=1.0).fit(lags[ok], y[ok]).predict(lags[ok])
        for c in EXOG:
            x = tr[c].values
            m = ok & np.isfinite(x)
            row = dict(Estacao=name, Horizonte=h, Variavel=c, N=int(m.sum()), Frac_disponivel=float(np.isfinite(x).mean()))
            if m.sum() >= 100 and np.std(x[m]) > 0:
                row["Pearson"] = pearsonr(x[m], y[m])[0]
                for tag, target in (("MI", y), ("MI_residuo", res)):
                    obs = mi(x[m], target[m])
                    null = np.array([mi(x[m], rng.permutation(target[m])) for _ in range(EXOG_PERM)])
                    row[tag], row[f"{tag}_excesso"] = obs, (obs - null.mean()) / null.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def make_dataset(d, lag, h, tz, exog, align):
    kt = d["Kt"]
    loc = d.index + pd.Timedelta(hours=tz)
    hr, doy = loc.hour.values.astype(float), loc.dayofyear.values.astype(float)
    feats = pd.concat([pd.DataFrame({f"kt_lag_{i}": kt.shift(i - 1) for i in range(1, lag + 1)}),
                       pd.DataFrame({"sin_hora": np.sin(2 * np.pi * hr / 24), "cos_hora": np.cos(2 * np.pi * hr / 24),
                                     "sin_dia": np.sin(2 * np.pi * doy / 365.25),
                                     "cos_dia": np.cos(2 * np.pi * doy / 365.25), "mu0": d["mu0"].values},
                                    index=d.index),
                       d[[c for x in exog for c in (x, f"{x}_lag1")]]], axis=1)
    tgt = d.index + pd.Timedelta(hours=h)
    y, h0, ghi = (d[c].reindex(tgt).values for c in ("Kt", "H0h", "GHI"))
    ok = kt.shift(align - 1).notna().values & feats.notna().all(axis=1).values & np.isfinite(y)
    return dict(X=feats.values[ok], y=y[ok], h0=h0[ok], ghi=ghi[ok], ts=d.index[ok], lag=lag, h=h)


def select(tr, scr, l_star, tz, name):
    s = scr[(scr.Horizonte == HORIZONS[0]) & (scr.MI_residuo_excesso >= EXOG_MIN_Z)
            & (scr.Frac_disponivel >= EXOG_MIN_AVAIL)]
    relev = s.set_index("Variavel").MI_residuo.to_dict()
    pool, base, order, cache = list(relev), tr[list(relev)].dropna(), [], {}
    rows = [dict(Estacao=name, Ordem=0, Variavel="nenhuma")]
    while pool and len(order) < EXOG_MAX:
        for c in pool:
            for j in order:
                cache.setdefault((c, j), mi(base[c], base[j]))
        red = {c: np.mean([cache[(c, j)] for j in order]) if order else 0.0 for c in pool}
        best = max(pool, key=lambda c: relev[c] - EXOG_W * red[c])
        order.append(best)
        pool.remove(best)
        rows.append(dict(Estacao=name, Ordem=len(order), Variavel=best, Relevancia=relev[best],
                         Redundancia=red[best], Escore=relev[best] - EXOG_W * red[best]))
    ds = make_dataset(tr, l_star, HORIZONS[0], tz, order, l_star)
    cut = tr.index[int(EXOG_INNER * len(tr))]
    fit, val = ds["ts"] + pd.Timedelta(hours=ds["h"]) < cut, ds["ts"] >= cut
    for r in rows:
        cols = l_star + N_CAL + 2 * r["Ordem"]
        m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0)
        m.fit(ds["X"][fit][:, :cols], ds["y"][fit])
        r["MSE_val_interna"] = float(np.mean((m.predict(ds["X"][val][:, :cols]) - ds["y"][val]) ** 2))
    k = int(np.argmin([r["MSE_val_interna"] for r in rows]))
    for r in rows:
        r["Selecionada"] = 0 < r["Ordem"] <= k
    return order[:k], pd.DataFrame(rows)


def bounds(idx, start, train, vbase, test, end):
    n = len(idx)
    pos = lambda f: idx[min(int(round(f * n)), n - 1)]
    return dict(start=idx[0] if start <= 0 else pos(start), train_end=pos(train), vb_end=pos(vbase),
                test_start=pos(test), test_end=idx[-1] + pd.Timedelta(hours=1) if end >= 1 - 1e-9 else pos(end))


def build_grid(crit):
    grid = set(SHORT_LAGS)
    for c in crit.values():
        grid |= {k * c["AMI_primeiro_minimo"] for k in range(1, AMI_MULTIPLES + 1)
                 if k * c["AMI_primeiro_minimo"] <= MAX_SCAN_LAG}
        grid |= {c[f"{k}_corte"] for k in ("PACF", "ACF") if c[f"{k}_corte"] and not c[f"{k}_saturada"]}
        grid.add(c["Amostras_por_dia"])
    grid = sorted(grid)
    rows = []
    for name, c in crit.items():
        a = c["AMI_primeiro_minimo"]
        for lag in grid:
            mult = lag // a if lag % a == 0 and lag // a <= AMI_MULTIPLES else None
            lab = (["curta"] if lag in SHORT_LAGS else []) + ([f"AMI x{mult}"] if mult else [])
            lab += [k for k in ("PACF", "ACF") if lag == c[f"{k}_corte"] and not c[f"{k}_saturada"]]
            lab += ["referencia diurna"] if lag == c["Amostras_por_dia"] else []
            rows.append(dict(Estacao=name, Lag=lag, Multiplo_AMI=mult, Rotulos=" + ".join(lab) or "exploratoria"))
    return grid, pd.DataFrame(rows)


def phase1():
    path = os.path.join(OUT_DIR, "phase1.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data, splits, crit, sel = {}, {}, {}, {}
    tabs = {k: [] for k in ("qc_summary", "mu0_sensitivity", "phase_check", "timeseries_stats", "ami_curve",
                            "exog_qc_summary", "exog_screening", "exog_selection", "splits")}
    for sid, st in STATIONS.items():
        name = st["name"]
        raw, header = read_station(sid)
        lat, lon = st["lat"], st["lon"]
        raw, eqc = exog_qc(raw)
        raw = exog_features(raw)
        tabs["exog_qc_summary"].append(eqc.assign(Estacao=name))
        for m in MU0_GRID:
            _, n = quality_control(raw, lat, lon, m)
            tabs["mu0_sensitivity"].append(dict(Estacao=name, mu0_min=m,
                                                Zenital_max_graus=float(np.degrees(np.arccos(m))), **n))
        d, n = quality_control(raw, lat, lon)
        tabs["qc_summary"].append(dict(Estacao=name, Latitude=lat, Longitude=lon, Latitude_cabecalho=header[0],
                                       Longitude_cabecalho=header[1], **n))
        tabs["phase_check"].append(phase_check(d, lon, name))
        b = bounds(d.index, 0, TRAIN_END, VAL_BASE_END, TEST_START, 1)
        splits[name] = {"fixa": b}
        for f in range(ROLL_FOLDS):
            s0 = f * ROLL_STEP
            splits[name][f"dobra{f + 1}"] = bounds(d.index, s0, s0 + 0.75 * ROLL_WINDOW, s0 + 0.875 * ROLL_WINDOW,
                                                   s0 + ROLL_WINDOW, s0 + ROLL_WINDOW + ROLL_TEST)
        tabs["splits"] += [dict(Estacao=name, Desenho=k, **v) for k, v in splits[name].items()]
        tr = d[d.index < b["train_end"]]
        stats, curve = characterize(tr, st["tz"], name)
        crit[name] = stats
        tabs["timeseries_stats"].append(stats)
        tabs["ami_curve"].append(curve)
        scr = screen(tr, stats["AMI_primeiro_minimo"], name)
        sel[name], trail = select(tr, scr, stats["AMI_primeiro_minimo"], st["tz"], name)
        tabs["exog_screening"].append(scr)
        tabs["exog_selection"].append(trail)
        data[name] = d
        print(f"{name}: lat={lat} lon={lon} cabecalho={header} N={len(d)} AMI={stats['AMI_primeiro_minimo']} "
              f"PACF={stats['PACF_corte']} ACF={stats['ACF_corte']}{' (saturada)' if stats['ACF_saturada'] else ''} "
              f"dia={stats['Amostras_por_dia']} exogenas={sel[name]}", flush=True)
    grid, labels = build_grid(crit)
    out = dict(data=data, splits=splits, crit=crit, sel=sel, grid=grid,
               tables={k: (pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v))
                       for k, v in tabs.items()})
    out["tables"]["lag_grid"] = labels
    with open(path, "wb") as f:
        pickle.dump(out, f)
    for k, t in out["tables"].items():
        t.to_csv(os.path.join(OUT_DIR, f"{k}.csv"), **CSV)
    print(f"grade de defasagens (amostras diurnas): {grid}", flush=True)
    return out


def partition(ds, b):
    ts = ds["ts"]
    tt = ts + pd.Timedelta(hours=ds["h"])
    masks = {"train": (ts >= b["start"]) & (tt < b["train_end"]), "vb": (ts >= b["train_end"]) & (tt < b["vb_end"]),
             "vm": (ts >= b["vb_end"]) & (tt < b["test_start"]), "test": (ts >= b["test_start"]) & (tt < b["test_end"])}
    P = {k: {"X": ds["X"][m], "y": ds["y"][m], "h0": ds["h0"][m], "ghi": ds["ghi"][m], "ts": ts[m]}
         for k, m in masks.items()}
    sc = StandardScaler().fit(P["train"]["X"])
    for p in P.values():
        p["Xs"] = sc.transform(p["X"])
    return P


def kt_to_ghi(kt, h0):
    return np.clip(np.asarray(kt, float), 0, KT_MAX) * h0


def metrics(obs, pred, pers):
    e = pred - obs
    sse, sst = np.sum(e ** 2), np.sum((obs - obs.mean()) ** 2)
    den = np.sum((np.abs(pred - obs.mean()) + np.abs(obs - obs.mean())) ** 2)
    rmsd, rmsd_p = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean((pers - obs) ** 2))
    return dict(MBD=e.mean(), MAD=np.abs(e).mean(), RMSD=rmsd, R2=1 - sse / sst, dm=1 - sse / den,
                SS=1 - rmsd / rmsd_p)


def _p3(x, a, b, c, d): return a * x + b * x ** 2 + c * x ** 3 + d
def _sig4(x, a, b, c, d): return d + a / (1 + np.exp(-(x - c) / np.where(b == 0, 1e-6, b)))
def _log3(x, a, b, c): return a / (1 + (np.abs(x) / max(abs(c), 1e-6)) ** b)
def _log4(x, a, b, c, d): return d + (a - d) / (1 + (np.abs(x) / max(abs(c), 1e-6)) ** b)
def _rat(x, a, b, c, d): return (a + b * x) / (1 + c * x + d * x ** 2 + 1e-9)


EMPIRICAL = {"Emp: 3_polynomial": (_p3, [0.5, 0, 0, 0.1]), "Emp: sigmoid_4p": (_sig4, [0.5, 0.2, 0.5, 0.1]),
             "Emp: logistic_3p": (_log3, [0.5, 1, 0.5]), "Emp: logistic_4p": (_log4, [0.5, 1, 0.5, 0.1]),
             "Emp: rational": (_rat, [0.1, 0.5, 0.1, 0.1])}


def fit_empirical(x, y, x_te):
    out = {}
    for k, (fun, p0) in EMPIRICAL.items():
        try:
            popt, _ = curve_fit(fun, x, y, p0=p0, method="lm", maxfev=20000)
            out[k] = np.nan_to_num(fun(x_te, *popt), nan=0.0, posinf=KT_MAX, neginf=0.0)
        except Exception:
            out[k] = np.full(len(x_te), float(np.mean(y)))
    return out


def svr(hp):
    return SVR(kernel="rbf", cache_size=1000, **hp)


def svr_mse(X, y, Xv, yv, hp):
    return float(np.mean((svr(hp).fit(X, y).predict(Xv) - yv) ** 2))


def svr_part(X, y, keep, Xp, hp):
    return svr(hp).fit(X[keep], y[keep]).predict(Xp)


def fit_svm(P):
    grid = list(ParameterGrid(SVM_GRID))
    X, y = P["train"]["Xs"], P["train"]["y"]
    err = Parallel(n_jobs=-1)(delayed(svr_mse)(X[-SVM_TUNE_N:], y[-SVM_TUNE_N:], P["vb"]["Xs"], P["vb"]["y"], hp)
                              for hp in grid)
    hp = grid[int(np.argmin(err))]
    edges = np.linspace(0, len(y), OOF_FOLDS + 1).astype(int)
    full = np.arange(len(y))
    evals = np.vstack([P[s]["Xs"] for s in ("vb", "vm", "test")])
    jobs = [(np.r_[0:edges[k], edges[k + 1]:len(y)], X[edges[k]:edges[k + 1]]) for k in range(OOF_FOLDS)]
    out = Parallel(n_jobs=-1)(delayed(svr_part)(X, y, keep, Xp, hp) for keep, Xp in jobs + [(full, evals)])
    cuts = np.cumsum([len(P[s]["y"]) for s in ("vb", "vm")])
    pred = dict(zip(("vb", "vm", "test"), np.split(out[-1], cuts)))
    return hp, pred, y - np.concatenate(out[:-1])


def mlp_fit(X, y, Xv, yv, hp):
    m = MLPRegressor(activation="relu", solver="adam", learning_rate_init=LR, random_state=SEED, **hp)
    best, wait, state = np.inf, 0, None
    for ep in range(1, MLP_EPOCHS + 1):
        m.partial_fit(X, y)
        e = float(np.mean((m.predict(Xv) - yv) ** 2))
        if e < best:
            best, wait, state = e, 0, ([w.copy() for w in m.coefs_], [b.copy() for b in m.intercepts_], ep)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.coefs_, m.intercepts_ = state[0], state[1]
    return best, state[2], m


def fit_mlp(P):
    grid = list(ParameterGrid(MLP_GRID))
    fits = Parallel(n_jobs=-1)(delayed(mlp_fit)(P["train"]["Xs"], P["train"]["y"], P["vb"]["Xs"], P["vb"]["y"], hp)
                               for hp in grid)
    i = int(np.argmin([f[0] for f in fits]))
    hp = dict(grid[i], epocas=fits[i][1])
    return hp, {s: fits[i][2].predict(P[s]["Xs"]) for s in ("vm", "test")}


def fit_xgb(P):
    best = None
    for hp in ParameterGrid(XGB_GRID):
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=30, reg_alpha=0.1, reg_lambda=1.0,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, tree_method="hist",
                             verbosity=0, **hp)
        m.fit(P["train"]["Xs"], P["train"]["y"], eval_set=[(P["vb"]["Xs"], P["vb"]["y"])], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (m.best_score, dict(hp, arvores=m.best_iteration + 1), m)
    return best[1], {s: best[2].predict(P[s]["Xs"]) for s in ("vm", "test")}


def seq(Xs, lag):
    return [Xs[:, :lag][:, ::-1, None], Xs[:, lag:]]


def fit_lstm(P, lag, y_tr, y_vb, seed):
    keras.utils.set_random_seed(seed)
    s_in, x_in = keras.Input((lag, 1)), keras.Input((P["train"]["Xs"].shape[1] - lag,))
    h = layers.LSTM(LSTM_UNITS, return_sequences=True)(s_in)
    a = layers.Softmax(axis=1)(layers.Dense(1, activation="tanh")(h))
    c = layers.Flatten()(layers.Dot(axes=1)([a, h]))
    z = layers.Dense(DENSE_UNITS, activation="relu")(layers.Dropout(DROPOUT)(layers.Concatenate()([c, x_in])))
    m = keras.Model([s_in, x_in], layers.Dense(1)(z))
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    hist = m.fit(seq(P["train"]["Xs"], lag), y_tr, validation_data=(seq(P["vb"]["Xs"], lag), y_vb), epochs=EPOCHS,
                 batch_size=BATCH, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    best = int(np.argmin(hist.history["val_loss"])) + 1
    return best, {s: m.predict(seq(P[s]["Xs"], lag), batch_size=1024, verbose=0).ravel() for s in ("vm", "test")}


def hetds(P, pv, pt):
    nn = NearestNeighbors(n_neighbors=min(HETDS_K, len(P["vm"]["y"]))).fit(P["vm"]["Xs"])
    idx = nn.kneighbors(P["test"]["Xs"], return_distance=False)
    err = np.column_stack([np.abs(pv[k] - P["vm"]["y"]) for k in BASE])
    pick = err[idx].mean(axis=1).argmin(axis=1)
    return np.column_stack([pt[k] for k in BASE])[np.arange(len(pick)), pick]


def bma_weights(mse, kappa):
    if np.isinf(kappa):
        return (mse == mse.min()).astype(float) / (mse == mse.min()).sum()
    z = -kappa * mse / mse.min()
    w = np.exp(z - z.max())
    return w / w.sum()


def dsag(P, pv, pt, seed):
    keras.utils.set_random_seed(seed)
    f_in, p_in = keras.Input((P["vm"]["Xs"].shape[1],)), keras.Input((len(BASE),))
    w = layers.Dense(len(BASE), activation="softmax")(layers.Dense(32, activation="relu")(f_in))
    m = keras.Model([f_in, p_in], layers.Dot(axes=1)([w, p_in]))
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    m.fit([P["vm"]["Xs"], np.column_stack([pv[k] for k in BASE])], P["vm"]["y"], epochs=60, batch_size=BATCH,
          verbose=0, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return m.predict([P["test"]["Xs"], np.column_stack([pt[k] for k in BASE])], batch_size=1024, verbose=0).ravel()


def run_job(ds, b, name, design, fold):
    P = partition(ds, b)
    sizes = {k: len(P[k]["y"]) for k in ("train", "vb", "vm", "test")}
    if min(sizes.values()) < 30:
        return None
    lag, te = ds["lag"], P["test"]
    hp_svm, svm, svm_res = fit_svm(P)
    hp_mlp, mlp = fit_mlp(P)
    hp_xgb, xg = fit_xgb(P)
    ep_lstm, lstm = fit_lstm(P, lag, P["train"]["y"], P["vb"]["y"], SEED)
    ep_res, res = fit_lstm(P, lag, svm_res, P["vb"]["y"] - svm["vb"], SEED + 1)
    pv = {"ML: ANN": mlp["vm"], "ML: SVM": svm["vm"], "ML: XGBoost": xg["vm"], "DL: LSTM+Att": lstm["vm"],
          "Hybrid: SVM+LSTM-res": svm["vm"] + res["vm"]}
    pt = {"ML: ANN": mlp["test"], "ML: SVM": svm["test"], "ML: XGBoost": xg["test"], "DL: LSTM+Att": lstm["test"],
          "Hybrid: SVM+LSTM-res": svm["test"] + res["test"]}
    mse_vm = np.array([np.mean((pv[k] - P["vm"]["y"]) ** 2) for k in BASE])
    stack = np.column_stack([pt[k] for k in BASE])
    kt = dict(pt)
    kt["Ens: HetDS"] = hetds(P, pv, pt)
    kt["Ens: BMA"] = stack @ bma_weights(mse_vm, BMA_KAPPA)
    kt["Ens: DSAG"] = dsag(P, pv, pt, SEED)
    kt.update(fit_empirical(P["train"]["X"][:, 0], P["train"]["y"], te["X"][:, 0]))
    pers = kt_to_ghi(te["X"][:, 0], te["h0"])
    ghi = {PERS: pers, **{k: kt_to_ghi(v, te["h0"]) for k, v in kt.items()}}
    key = dict(Estacao=name, Lag=lag, Horizonte=ds["h"], Desenho=design, Dobra=fold)
    n = dict(N_treino=sizes["train"], N_val_base=sizes["vb"], N_val_meta=sizes["vm"], N_teste=sizes["test"])
    out = dict(rows=[dict(key, Modelo=k, **n, **metrics(te["ghi"], v, pers)) for k, v in ghi.items()],
               hp=dict(key, **{f"SVM_{k}": v for k, v in hp_svm.items()}, **{f"MLP_{k}": str(v) for k, v in hp_mlp.items()},
                       **{f"XGB_{k}": v for k, v in hp_xgb.items()}, LSTM_epocas=ep_lstm, LSTM_res_epocas=ep_res),
               weights=dict(key, **{f"w: {k}": w for k, w in zip(BASE, bma_weights(mse_vm, BMA_KAPPA))},
                            **{f"MSE_val_meta: {k}": e for k, e in zip(BASE, mse_vm)}),
               kappa=[dict(key, kappa=kap, **{f"w: {k}": w for k, w in zip(BASE, bma_weights(mse_vm, kap))},
                           **metrics(te["ghi"], kt_to_ghi(stack @ bma_weights(mse_vm, kap), te["h0"]), pers))
                      for kap in BMA_KAPPAS],
               pred=pd.DataFrame(dict(key, Origem=te["ts"], Alvo=te["ts"] + pd.Timedelta(hours=ds["h"]),
                                      GHI_real=te["ghi"], Kt_real=te["y"], H0h_alvo=te["h0"], Kt_origem=te["X"][:, 0],
                                      **{f"Prev: {k}": v for k, v in ghi.items()})))
    keras.backend.clear_session()
    return out


def audit(m):
    fixed = m[m.Desenho == "fixa"]
    aud = fixed.assign(Pass_R2=~fixed.Falha_R2, Pass_vs_pers=~fixed.Falha_pers)
    aud["Flagged_Failed"] = ~(aud.Pass_R2 & aud.Pass_vs_pers)
    aud = aud[["Estacao", "Lag", "Horizonte", "Modelo", "R2", "RMSD", "RMSD_pers", "Pass_R2", "Pass_vs_pers",
               "Flagged_Failed"]]
    frac = aud[aud.Modelo != PERS].groupby("Modelo").Flagged_Failed.mean()
    failed = sorted(frac[frac > FAIL_FRAC].index)
    roll = m[m.Desenho == "movel"].groupby(["Estacao", "Lag", "Horizonte", "Modelo"]).RMSD.agg(["mean", "std"])
    cmp = (fixed.set_index(["Estacao", "Lag", "Horizonte", "Modelo"])[["RMSD"]].rename(columns={"RMSD": "RMSD_janela_fixa"})
           .join(roll.rename(columns={"mean": "RMSD_deslizante_media", "std": "RMSD_deslizante_dp_dobras"}), how="inner")
           .reset_index())
    cmp["Diferenca_abs"] = (cmp.RMSD_deslizante_media - cmp.RMSD_janela_fixa).abs()
    cmp["Diferenca_rel"] = 100 * cmp.Diferenca_abs / cmp.RMSD_janela_fixa
    return {"stage2_audit.csv": aud, "stage2_failed.csv": pd.DataFrame({"Modelo": failed}), "rolling_vs_fixed.csv": cmp}


def export(final=False):
    res = []
    for f in sorted(os.listdir(os.path.join(WORK_DIR, "cells"))):
        if f.endswith(".pkl"):
            with open(os.path.join(WORK_DIR, "cells", f), "rb") as fh:
                r = pickle.load(fh)
            if r:
                res.append(r)
    if not res:
        return
    key = ["Estacao", "Lag", "Horizonte", "Desenho", "Dobra"]
    m = pd.DataFrame([row for r in res for row in r["rows"]])
    m = m.join(m[m.Modelo == PERS].set_index(key).RMSD.rename("RMSD_pers"), on=key)
    m["Falha_R2"] = m.R2 < R2_MIN
    m["Falha_pers"] = m.RMSD > PERS_RATIO_MAX * m.RMSD_pers
    m = m.sort_values(key + ["Modelo"])
    m[m.Desenho == "fixa"].drop(columns=["Desenho", "Dobra"]).to_csv(os.path.join(OUT_DIR, "metrics_fixed.csv"), **CSV)
    m[m.Desenho == "movel"].to_csv(os.path.join(OUT_DIR, "metrics_rolling.csv"), **CSV)
    pd.DataFrame([r["hp"] for r in res]).sort_values(key).to_csv(os.path.join(OUT_DIR, "hyperparameters.csv"), **CSV)
    pd.DataFrame([r["weights"] for r in res]).sort_values(key).to_csv(os.path.join(OUT_DIR, "bma_weights.csv"), **CSV)
    pd.DataFrame([k for r in res for k in r["kappa"]]).sort_values(key + ["kappa"]).to_csv(
        os.path.join(OUT_DIR, "bma_kappa.csv"), **CSV)
    if final:
        preds = pd.concat([r["pred"] for r in res], ignore_index=True).sort_values(key + ["Origem"])
        for design, name in (("fixa", "predictions_fixed.csv"), ("movel", "predictions_rolling.csv")):
            t = preds[preds.Desenho == design]
            if len(t):
                t.to_csv(os.path.join(OUT_DIR, name), **CSV)
        for name, t in audit(m).items():
            t.to_csv(os.path.join(OUT_DIR, name), **CSV)


def main():
    t0 = time.time()
    mount()
    p1 = phase1()
    flush()
    grid = p1["grid"]
    anchors = {n: c["AMI_primeiro_minimo"] for n, c in p1["crit"].items()}
    lags = grid if not SMOKE_TEST else sorted(set(SHORT_LAGS) | set(anchors.values()))
    tasks = [(sid, st["name"], lag, h, "fixa", 0) for sid, st in STATIONS.items() for lag in lags for h in HORIZONS]
    tasks += [(sid, st["name"], anchors[st["name"]], h, "movel", f + 1)
              for sid, st in STATIONS.items() for h in HORIZONS for f in range(ROLL_FOLDS)]
    rank = lambda t: (0 if t[4] == "fixa" and t[2] == anchors[t[1]] else 1 if t[4] == "fixa" and t[3] == 1
                      else 2 if t[4] == "movel" else 3, t[3], t[2], t[5], t[0])
    tasks.sort(key=rank)
    job = lambda t: f"{t[1].replace(' ', '_')}_{t[4]}{t[5]}_L{t[2]:03d}_h{t[3]}"
    pending = [t for t in tasks if not done(job(t))]
    print(f"{len(tasks)} células, {len(pending)} pendentes", flush=True)
    align, spent = max(grid), []
    for i, t in enumerate(pending, 1):
        sid, name, lag, h, design, fold = t
        t1 = time.time()
        ds = make_dataset(p1["data"][name], lag, h, STATIONS[sid]["tz"], p1["sel"][name], align)
        b = p1["splits"][name]["fixa" if design == "fixa" else f"dobra{fold}"]
        save_cell(job(t), run_job(ds, b, name, design, fold))
        spent.append((time.time() - t1) / 60)
        print(f"[{i}/{len(pending)}] {job(t)} {spent[-1]:.1f} min, restam ~{np.mean(spent) * (len(pending) - i) / 60:.1f} h",
              flush=True)
        if i % FLUSH_EVERY == 0:
            export()
            flush()
    export(final=True)
    flush()
    print(f"concluído em {(time.time() - t0) / 3600:.2f} h; saídas em {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
