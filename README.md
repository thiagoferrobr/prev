# Hourly global horizontal irradiance forecasting: Cercadinho (MG) and Ponta Porã (MS)

Reproducibility package for the master's dissertation *Previsão horária de irradiância global horizontal por combinação heterogênea de modelos com dimensionamento informacional da janela autorregressiva*, Programa de Pós-Graduação em Energias Renováveis, Universidade Federal de Alagoas, 2026.

The package has two scripts. `prev_v9.py` runs the five stages of the method, from the INMET annual files to the CSV outputs. `figuras_tabelas.py` reads those outputs and regenerates every figure and every data table of Chapters 4 and 5 of the dissertation. The quantities quoted in the text are read from the same CSVs under `outputs/`. Thresholds, grids, partition fractions and every other setting are declared in the configuration block at the top of `prev_v9.py`.

## Data

Hourly records of the INMET automatic stations F501 (Belo Horizonte, Cercadinho, MG) and A703 (Ponta Porã, MS) for 2023 to 2025. The annual files are read from the directory `PREV_DATA_DIR` (default `dados_inmet`, or `MyDrive/dados_inmet` in Colab). When the directory is absent, the annual archives are downloaded from the INMET historical data service. The hour column published by INMET is expressed in UTC and is kept on that axis, so solar geometry is evaluated at the timestamp it belongs to. In the CSVs the second station is named Ponta Pora, without the diacritic.

## Stages of `prev_v9.py`

1. **Acquisition and quality control.** Exogenous variables are checked against physical ranges. The diurnal mask keeps records with cosine of the solar zenith angle above 0.10 at the middle of the accumulation hour. Physically possible and rare-event limits follow the BSRN protocol in a GHI-only setting. The clearness index is the ratio between the measured irradiance and the extraterrestrial irradiance integrated over the accumulation hour, with 60 sub-steps, truncated at 1. A phase check compares the centroids of the measured irradiance and of the integrated denominator with the analytical solar noon and aborts the run when they disagree beyond tolerance. The sensitivity of the retained set to the mask threshold is tabulated.
2. **Stochastic characterization and lag grid.** Computed on the training partition only. Coefficient of variation, ADF and KPSS tests, Lomb-Scargle periodogram at 24, 12, 8 and 6 h over the real time axis and over the sample axis, auto mutual information up to 72 lags with 16 equiprobable bins, and PACF and ACF cutoffs against the 95 percent band. The lag grid is the union of the first three multiples of the AMI first minimum, the PACF and ACF cutoffs and the diurnal reference, evaluated in both stations. Each station-lag pair receives a label.
3. **Exogenous screening and selection.** Candidates are ranked by mutual information with the target and with the residual of a ridge autoregressive fit, against a permutation null with 20 draws. A candidate passes with availability of at least 0.98 and incremental score of at least 3. Selection is greedy by maximum relevance and minimum redundancy, up to five variables, with an internal XGBoost validation. The set selected at 1 h is kept at 3 and 6 h.
4. **Modelling.** Fourteen models per cell: smart persistence; five empirical forms fitted by nonlinear least squares (third-degree polynomial, four-parameter sigmoid, three- and four-parameter logistic, rational); MLP, SVM and XGBoost with grid search; an LSTM with attention; the hybrid SVM+LSTM-res, which fits the LSTM to out-of-fold residuals of the SVM; and three combination mechanisms over the five learning models, HetDS (dynamic selection by seven neighbours), an exponential weighting with scale factor derived from Bayesian model averaging (κ = 5, with a κ grid for sensitivity) and DSAG (dynamic stacking with attention gating). Predictions are converted back to irradiance before scoring.
5. **Evaluation.** Fixed split (50 percent training, 8.3 percent base-model validation, 8.3 percent combination validation, 33.3 percent test) over 11 lags, 3 horizons and 2 stations, 66 cells; rolling origin with 5 folds in the six anchored cells, whose lag is the AMI first minimum. Metrics in W/m²: MBD, MAD, RMSD, coefficient of determination, Willmott index of agreement and skill score against smart persistence. An eligibility audit flags models with coefficient of determination below 0.40 or RMSD above 1.1 times that of persistence, and lists the models flagged in more than half of the cells. The fixed and rolling designs are compared in the anchored cells. No hypothesis test is applied to the results.

## Outputs of `prev_v9.py`

Characterization:

- `qc_summary.csv` records retained at each quality-control step, header coordinates, mean clearness index
- `mu0_sensitivity.csv` sensitivity of the retained set to the diurnal mask threshold
- `phase_check.csv` centroids, solar noon and phase deviations
- `timeseries_stats.csv` stochastic descriptors, tests, periodogram and window criteria per station
- `ami_curve.csv` auto mutual information per lag
- `exog_qc_summary.csv` physical range control of the exogenous variables
- `exog_screening.csv` marginal and incremental relevance with the permutation null, per horizon
- `exog_selection.csv` trail of the greedy selection
- `splits.csv` partition boundaries of the fixed design and of each rolling fold
- `lag_grid.csv` lag grid with the label of each station-lag pair

Results:

- `metrics_fixed.csv` and `metrics_rolling.csv` one record per cell and model, with the audit flags
- `hyperparameters.csv` values selected by grid search and early-stopping epochs
- `bma_weights.csv` combination weights and validation MSE of each base model
- `bma_kappa.csv` combination metrics for each value of the scale factor
- `stage2_audit.csv` and `stage2_failed.csv` eligibility audit and models with systematic failure
- `rolling_vs_fixed.csv` RMSD of the fixed test block against the mean and standard deviation over the rolling folds
- `predictions_fixed.csv` and `predictions_rolling.csv` hourly predictions of every model on the test blocks, in W/m²

Values are written with semicolon as separator and comma as decimal mark. Per-cell checkpoints are written under `cells/` and `phase1.pkl` holds the retained series and the partitions; neither is needed by `figuras_tabelas.py`.

## Outputs of `figuras_tabelas.py`

Figures, under `figuras/`: `fig_fase.pdf`, `fig_periodograma.pdf`, `fig_memoria.pdf` (Chapter 4) and `fig_kt_distribuicao.pdf`, `fig_superficie.pdf`, `fig_superficie_modelos.pdf`, `fig_previsto_real.pdf`, `fig_kappa.pdf`, `fig_origem_movel.pdf` (Chapter 5).

Table bodies, under `tabelas/`, as LaTeX rows to be placed between `\midrule` and `\bottomrule`: `tab_mu0`, `tab_caracterizacao`, `tab_grade`, `tab_exog_cerc`, `tab_exog_pp`, `tab_trilha_cerc`, `tab_trilha_pp`, `tab_particoes` (Chapter 4) and `tab_ancoradas_h1`, `tab_ancoradas_h36`, `tab_criterios`, `tab_vitorias`, `tab_mbd`, `tab_regimes`, `tab_calibracao`, `tab_origem_movel` (Chapter 5). The KPSS statistics quoted in the note of the characterization table are written as comment lines at the end of `tab_caracterizacao.tex`.

The script takes the retained series from `phase1.pkl` when that file is present in the outputs directory; otherwise it rebuilds them from the INMET annual files with the functions of `prev_v9.py`, which must be in the same directory as `figuras_tabelas.py`. The anchored lags, the best model of each anchored cell and the lag grid are read from the outputs, not hard coded.

## Repository structure

```
prev/
├── prev_v9.py            pipeline: acquisition, characterization, screening, modelling, evaluation, export
├── figuras_tabelas.py    figures and tables of the dissertation from the outputs
├── requirements.txt      dependencies
├── outputs/              CSVs of the published run
├── figuras/              figures regenerated from outputs/
├── tabelas/              table bodies regenerated from outputs/
├── CITATION.cff
├── LICENSE
└── README.md
```

## Running

In Google Colab, with a GPU runtime and the annual INMET files in `MyDrive/dados_inmet`:

```
%run prev_v9.py
```

Locally:

```
pip install -r requirements.txt
PREV_DATA_DIR=dados_inmet PREV_OUT_DIR=prev_v9 python prev_v9.py
```

The outputs go to `PREV_OUT_DIR` (default `prev_v9`, or `MyDrive/prev_v9` in Colab). A run that stops before the sweep ends is resumed from the completed cells on the next call. Cells run by priority: fixed design in the anchored cells, fixed design at 1 h, rolling origin, and the remaining cells. The result CSVs are rewritten every five cells and the prediction files and the audit at the end. To rebuild the result CSVs from checkpoints already on disk, without refitting:

```
from prev_v9 import mount, export
mount()
export(final=True)
```

Figures and tables, locally:

```
PREV_DATA_DIR=dados_inmet python figuras_tabelas.py outputs .
```

The first argument is the outputs directory and the second is the destination, which receives the subdirectories `figuras/` and `tabelas/`. Without arguments, the outputs directory is `PREV_OUT_DIR` (default `prev_v9`) and the destination is its `figuras_tabelas/` subdirectory. In Google Colab, with `prev_v9.py` and `figuras_tabelas.py` in the same folder:

```
%run figuras_tabelas.py
```

The script reads `MyDrive/prev_v9`, takes the retained series from `phase1.pkl` and writes `MyDrive/prev_v9/figuras_tabelas/figuras` and `MyDrive/prev_v9/figuras_tabelas/tabelas`. To regenerate the figures and tables of the dissertation, point it at the published outputs (the `outputs/` directory of this repository), not at a new execution of the pipeline (see Reproducibility).

## Reproducibility

The seed 42 is fixed in every random component and does not change with the cell or the fold; the residual LSTM of the hybrid model uses 43, so that it does not share the initialization of the LSTM with attention. Characterization, screening and model calibration use only data preceding the test block. The models fitted with SciPy, scikit-learn and XGBoost (persistence, the five empirical forms, MLP, SVM and XGBoost) reproduce bit for bit between executions on the same platform. The Keras models (LSTM with attention, residual LSTM of the hybrid and DSAG gating network) do not, because cuDNN and oneDNN kernels are not deterministic, and HetDS and BMA inherit their differences through the base-model predictions. In a second execution of the published configuration in Google Colab, the RMSD of these models changed by 0.1 to 1.4 W/m² on average over the 66 cells, with a maximum of 23.7 W/m² for the LSTM with attention in one cell, which moves one of the 66 lowest-RMSD counts. The outputs under `outputs/` are those of the published run and are the ones the dissertation reports; the figures and tables in this repository were regenerated from them, and the table bodies match the dissertation line by line.

## Citation

```
@mastersthesis{ferro2026dissertacao,
  author  = {Ferro de Oliveira, Thiago},
  title   = {Previs{\~a}o hor{\'a}ria de irradi{\^a}ncia global horizontal por combina{\c c}{\~a}o
             heterog{\^e}nea de modelos com dimensionamento informacional da janela autorregressiva},
  school  = {Universidade Federal de Alagoas, Programa de P{\'o}s-Gradua{\c c}{\~a}o em Energias Renov{\'a}veis},
  year    = {2026}
}

@software{ferro2026code,
  author    = {Ferro de Oliveira, Thiago and Aquino, Andr{\'e} Luiz Lins de and Queiroz, Fabiane da Silva},
  title     = {Hourly Solar Irradiance Forecasting Pipeline},
  year      = {2026},
  url       = {https://github.com/thiagoferrobr/prev}
}
```

## License

Code under MIT, see `LICENSE`. Output CSVs, figures and tables under CC BY 4.0.

## Acknowledgements

INMET, Instituto Nacional de Meteorologia, for the public availability of the hourly station data. The empirical models and the proportion of the fixed split follow Krieger et al. (2024). The author acknowledges the support of the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior and of the Programa de Pós-Graduação em Energias Renováveis of the Universidade Federal de Alagoas.
