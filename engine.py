import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="arch")

@dataclass
class CertificateSpec:
    """Specification of a worst-of barrier certificate."""
    notional: float = 100.0           
    issue_price: float = 100.0        
    barrier_pct: float = 0.40         
    coupon_annual_pct: float = 7.80   
    coupon_freq_days: int = 30        
    maturity_days: int = 504          

    # ---- optional features ----
    memory_coupon: bool = False       
    autocall: bool = False            
    autocall_first_day: int = 252     
    autocall_freq_days: int = 252     
    autocall_trigger_pct: float = 1.0 

    # ---- capital protection at maturity ----
    capital_protection_pct: float = 0.0   

def _to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\u00a0", "").replace(" ", "")
    if s == "" or s.lower() in {"nan", "n/a", "-"}:
        return np.nan
    s = s.replace(",", ".")
    mult = 1.0
    if s[-1].upper() in "KMB":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1].upper()]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan

def load_price_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if c in {"date", "data", "datetime"}), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    if df[date_col].isna().any():
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    close_col = next((c for c in df.columns if c in {"close", "chiusura", "ultimo", "price"}), df.columns[0])
    df["close"] = df[close_col].map(_to_float)
    
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df = df.ffill(limit=3)
    return df

def fit_garch(returns_pct: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    am = arch_model(returns_pct, mean="Zero", vol="GARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    P = res.params
    out = {
        "omega": float(P["omega"]) / 1e4,
        "alpha": float(P[f"alpha[1]"]),
        "beta":  float(P[f"beta[1]"]),
        "nu":    float(P["nu"]) if "nu" in P.index else np.inf,
    }
    out["sigma2_lr"] = out["omega"] / max(1 - out["alpha"] - out["beta"], 1e-6)
    return res, out

def simulate_paths(
    S0: dict,
    sigma0: dict,
    garch_params: dict,
    corr_matrix: np.ndarray,
    *,
    mode: str = "risk_neutral",
    risk_free: float = 0.0,
    div_yield: dict | None = None,
    mu_daily: dict | None = None,
    corr_blend: float = 1.0,
    n_days: int = 252,
    n_paths: int = 20000,
    innov_dist: str = "t",
    seed: int = 42,
    dtype = np.float32,
    trading_days: int = 252,
    progress_callback = None
) -> dict:
    rng = np.random.default_rng(seed)
    tickers = list(S0.keys())
    n_assets = len(tickers)
    dt = 1.0 / trading_days

    # Correlation blend
    if corr_blend != 1.0:
        c_mod = corr_matrix.copy()
        mask = ~np.eye(n_assets, dtype=bool)
        c_mod[mask] = c_mod[mask] * corr_blend
        eigvals, eigvecs = np.linalg.eigh(c_mod)
        eigvals = np.maximum(eigvals, 1e-8)
        c_mod = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(c_mod, 1.0)
        corr_matrix = np.clip(c_mod, -0.99, 0.99)
        np.fill_diagonal(corr_matrix, 1.0)
        
    L = np.linalg.cholesky(corr_matrix).astype(dtype)

    # TRUE GARCH SIMULATION: Variance is now shape (n_days+1, n_paths, n_assets)
    sigma2_path = np.zeros((n_days + 1, n_paths, n_assets), dtype=dtype)
    s0_v = np.array([sigma0[t] for t in tickers], dtype=dtype)
    sigma2_path[0] = (s0_v ** 2)[None, :]
    
    omega_v = np.array([garch_params[t]["omega"] for t in tickers], dtype=dtype)
    alpha_v = np.array([garch_params[t]["alpha"] for t in tickers], dtype=dtype)
    beta_v = np.array([garch_params[t]["beta"] for t in tickers], dtype=dtype)
    
    div_yield_v = np.array([div_yield[t] for t in tickers], dtype=dtype) if div_yield else np.zeros(n_assets, dtype=dtype)
    mu_daily_v = np.array([mu_daily[t] for t in tickers], dtype=dtype) if mu_daily else np.zeros(n_assets, dtype=dtype)

    prices_arr = np.empty((n_days + 1, n_paths, n_assets), dtype=dtype)
    log_S0_v = np.array([np.log(S0[t]) for t in tickers], dtype=dtype)
    prices_arr[0] = np.exp(log_S0_v)[None, :]
    log_state = np.broadcast_to(log_S0_v, (n_paths, n_assets)).astype(dtype).copy()

    for d in range(n_days):
        if progress_callback and d % max(1, n_days // 20) == 0:
            progress_callback(int(50 + 50 * d / n_days)) 

        sig2_d = np.clip(sigma2_path[d], 1e-10, 0.05)
        sig_d = np.sqrt(sig2_d)

        # Generate Innovations
        if innov_dist == "t":
            Z_step = np.empty((n_paths, n_assets), dtype=dtype)
            for i, t in enumerate(tickers):
                nu = max(garch_params[t]["nu"], 4.01)
                z = rng.standard_t(df=nu, size=n_paths).astype(dtype)
                Z_step[:, i] = z / np.float32(np.sqrt(nu / (nu - 2)))
        else:
            Z_step = rng.standard_normal((n_paths, n_assets)).astype(dtype)

        Z_corr = Z_step @ L.T
        
        # Calculate Drift
        if mode == "risk_neutral":
            drift = (risk_free - div_yield_v) * dt - 0.5 * sig2_d
        else:
            drift = mu_daily_v[None, :]

        # Price Step
        log_state = log_state + drift + sig_d * Z_corr
        prices_arr[d + 1] = np.exp(log_state)

        # GARCH Update (recursively applied on actual shocked paths)
        eps = sig_d * Z_corr
        sigma2_path[d+1] = omega_v + alpha_v * (eps**2) + beta_v * sig2_d

    if progress_callback: progress_callback(100)

    return {
        "prices": {t: prices_arr[:, :, i] for i, t in enumerate(tickers)},
        "meta": {"mode": mode, "n_days": n_days, "n_paths": n_paths},
        "S0": S0
    }

def price_certificate(sim: dict, spec: CertificateSpec, *, discount_rate: float | None = None, trading_days: int = 252) -> dict:
    tickers = list(sim["prices"].keys())
    P = np.stack([sim["prices"][t] for t in tickers], axis=0)  # (A, T+1, N)
    n_paths = P.shape[2]
    n_steps_total = P.shape[1] - 1

    P = P[:, : spec.maturity_days + 1, :]
    S0_v = P[:, 0, 0].astype(np.float64)

    coupon_days = np.arange(spec.coupon_freq_days, spec.maturity_days + 1, spec.coupon_freq_days)
    if coupon_days.size == 0 or coupon_days[-1] != spec.maturity_days:
        coupon_days = np.append(coupon_days, spec.maturity_days)

    if spec.autocall:
        ac_set = set(np.arange(spec.autocall_first_day, spec.maturity_days, spec.autocall_freq_days).tolist())
    else:
        ac_set = set()

    ratios = (P / S0_v[:, None, None]).astype(np.float64)
    worst_ratio = ratios.min(axis=0)  # (T+1, N)
    coupon_per_period = (spec.coupon_annual_pct / 100.0) * spec.notional * (spec.coupon_freq_days / trading_days)

    coupons_undisc = np.zeros(n_paths)
    coupons_disc = np.zeros(n_paths)
    pending = np.zeros(n_paths)
    autocalled = np.zeros(n_paths, dtype=bool)
    redemption_day = np.full(n_paths, spec.maturity_days, dtype=np.int32)
    do_disc = discount_rate is not None

    barrier_breached_anytime = (worst_ratio < spec.barrier_pct).any(axis=0)
    avg_times_below_barrier = (worst_ratio < spec.barrier_pct).sum(axis=0).mean()

    for d in coupon_days:
        d = int(d)
        alive = ~autocalled
        wr_d = worst_ratio[d]
        pays = alive & (wr_d >= spec.barrier_pct)

        pay_amt = coupon_per_period + pending if spec.memory_coupon else np.full(n_paths, coupon_per_period)

        coupons_undisc[pays] += pay_amt[pays]
        if do_disc:
            coupons_disc[pays] += np.exp(-discount_rate * d / trading_days) * pay_amt[pays]

        if spec.memory_coupon:
            pending[pays] = 0.0
            pending[alive & ~pays] += coupon_per_period

        if d in ac_set:
            trig = alive & (wr_d >= spec.autocall_trigger_pct)
            autocalled[trig] = True
            redemption_day[trig] = d

    final_wr = worst_ratio[spec.maturity_days]
    above = final_wr >= spec.barrier_pct
    barrier_capital = np.where(above, spec.notional, spec.notional * final_wr)
    barrier_capital = np.maximum(barrier_capital, spec.capital_protection_pct * spec.notional)
    capital = np.where(autocalled, spec.notional, barrier_capital)

    total_payoff = capital + coupons_undisc
    pnl = total_payoff - spec.issue_price

    if do_disc:
        T_yrs = redemption_day / trading_days
        pv = capital * np.exp(-discount_rate * T_yrs) + coupons_disc
    else:
        pv = np.full(n_paths, np.nan)

    cummax = np.maximum.accumulate(worst_ratio, axis=0)
    drawdowns = 1.0 - worst_ratio / cummax
    max_dd = drawdowns.max(axis=0)
    
    loss_paths = pnl[pnl < 0]

    def _var_es(arr, pct):
        cutoff = np.percentile(arr, 100 - pct)
        return -cutoff, -arr[arr <= cutoff].mean() if (arr <= cutoff).any() else -cutoff

    var90, es90 = _var_es(pnl, 90)
    var95, es95 = _var_es(pnl, 95)
    var99, es99 = _var_es(pnl, 99)

    summary = {
        # Fair Value
        "fair_value": float(pv.mean()) if do_disc else float("nan"),
        "fv_std_error": float(pv.std() / np.sqrt(n_paths)) if do_disc else float("nan"),
        
        # P&L and Expectations
        "C_avg_total_PL": float(pnl.mean()),
        "C_PL_5pct": float(np.percentile(pnl, 5)),
        "C_PL_50pct": float(np.percentile(pnl, 50)),
        "C_PL_95pct": float(np.percentile(pnl, 95)),
        "C_coupons_5pct": float(np.percentile(coupons_undisc, 5)),
        "C_coupons_50pct": float(np.percentile(coupons_undisc, 50)),
        "C_coupons_95pct": float(np.percentile(coupons_undisc, 95)),
        
        # Barrier Risk
        "A_prob_touch_barrier": float(barrier_breached_anytime.mean()),
        "A_avg_times_below_barrier": float(avg_times_below_barrier),
        "prob_autocall": float(autocalled.mean()),
        
        # Capital Risk
        "B_prob_capital_loss": float((pnl < 0).mean()),
        "B_avg_capital_loss_pct": float((loss_paths.mean() / spec.notional) if len(loss_paths) > 0 else 0),
        "B_pct_profit": float((pnl > 0).mean()),
        "B_pct_loss": float((pnl < 0).mean()),
        
        # Drawdown
        "D_max_drawdown_mean": float(max_dd.mean()),
        "D_max_drawdown_95pct": float(np.percentile(max_dd, 95)),
        
        # Portfolio Risk (VaR / ES)
        "VaR_90_portfolio": float(var90), "ES_90_portfolio": float(es90),
        "VaR_95_portfolio": float(var95), "ES_95_portfolio": float(es95),
        "VaR_99_portfolio": float(var99), "ES_99_portfolio": float(es99),
    }

    # Single Asset VaR/ES at maturity
    for i, t in enumerate(tickers):
        ret_t = ratios[-1, :] - 1.0 # return at maturity
        summary[f"{t}_VaR_95"] = float(-np.percentile(ret_t, 5))
        summary[f"{t}_ES_95"] = float(-ret_t[ret_t <= np.percentile(ret_t, 5)].mean())

    path_df = pd.DataFrame({
        "capital": capital,
        "coupons": coupons_undisc,
        "total_payoff": total_payoff,
        "pnl": pnl,
        "pv": pv,
        "max_drawdown": max_dd
    })

    return {
        "summary": summary,
        "path_df": path_df,
        "worst_ratio": worst_ratio,
    }

def run_simulation(data_dir: str, spec: CertificateSpec, risk_free_rate: float, 
                   dividend_yields: dict, mode: str="risk_neutral", corr_blend: float=1.0, 
                   n_days: int=504, n_paths: int=10000, seed: int=42, progress_callback=None) -> dict:
    if progress_callback: progress_callback(0)
    
    data_path = Path(data_dir)
    files = list(data_path.glob("*.csv"))
    if not files: raise ValueError(f"No CSV files found in {data_dir}")
        
    raw_data = {f.stem: load_price_file(f) for f in files}
    prices_df = pd.concat([d["close"] for d in raw_data.values()], axis=1, keys=raw_data.keys()).dropna()
    tickers = list(prices_df.columns)
    
    if progress_callback: progress_callback(10)
    
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    corr_matrix = log_ret.corr().values
    
    garch_params, sigma0, mu_daily = {}, {}, {}
    for i, t in enumerate(tickers):
        r_pct = log_ret[t].dropna() * 100
        fit, p = fit_garch(r_pct, dist="t")
        garch_params[t] = p
        sigma0[t] = float(fit.conditional_volatility.iloc[-1]) / 100.0
        mu_daily[t] = float(log_ret[t].mean())
        if progress_callback: progress_callback(int(10 + 40 * (i+1) / len(tickers)))
        
    S0 = {t: float(prices_df[t].iloc[-1]) for t in tickers}
    div_y = {t: float(dividend_yields.get(t, 0.0)) for t in tickers}

    sim_rn = simulate_paths(
        S0=S0, sigma0=sigma0, garch_params=garch_params,
        corr_matrix=corr_matrix, mode=mode,
        risk_free=risk_free_rate, div_yield=div_y, mu_daily=mu_daily, corr_blend=corr_blend,
        n_days=n_days, n_paths=n_paths, seed=seed, progress_callback=progress_callback
    )
    
    res = price_certificate(sim_rn, spec, discount_rate=risk_free_rate if mode=="risk_neutral" else None)
    return {
        "summary": res["summary"],
        "path_df": res["path_df"],
        "sim_prices": sim_rn["prices"],
        "tickers": tickers
    }
