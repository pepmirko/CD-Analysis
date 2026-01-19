#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# -------------------------
# Parsing JASCO TXT (XYDATA)
# -------------------------
@dataclass
class CDSpectrum:
    wl_nm: np.ndarray
    cd_mdeg: np.ndarray
    ht_v: np.ndarray | None = None


def load_jasco_txt(path: Path) -> CDSpectrum:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("XYDATA") + 1
    except ValueError as e:
        raise ValueError(f"{path}: 'XYDATA' non trovato") from e

    wl, cd, ht = [], [], []
    for ln in lines[start:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        wl.append(float(parts[0]))
        cd.append(float(parts[1]))
        ht.append(float(parts[2]) if len(parts) >= 3 else np.nan)

    wl = np.asarray(wl, dtype=float)
    cd = np.asarray(cd, dtype=float)
    ht = np.asarray(ht, dtype=float)

    order = np.argsort(wl)
    wl, cd, ht = wl[order], cd[order], ht[order]

    ht_arr = None if np.all(np.isnan(ht)) else ht
    return CDSpectrum(wl_nm=wl, cd_mdeg=cd, ht_v=ht_arr)


# -------------------------
# Processing
# -------------------------
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        raise ValueError("smooth-window deve essere dispari (odd)")
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(window, dtype=float) / window
    return np.convolve(ypad, k, mode="valid")


def baseline_correct_offset(spec: CDSpectrum, lo_hi: tuple[float, float]) -> CDSpectrum:
    lo, hi = lo_hi
    m = (spec.wl_nm >= lo) & (spec.wl_nm <= hi)
    if not np.any(m):
        raise ValueError(f"Nessun punto nell'intervallo baseline {lo}-{hi} nm")
    off = float(np.mean(spec.cd_mdeg[m]))
    return CDSpectrum(spec.wl_nm, spec.cd_mdeg - off, spec.ht_v)


def interp_at(spec: CDSpectrum, target_nm: float) -> float:
    return float(np.interp(target_nm, spec.wl_nm, spec.cd_mdeg))


def mre_from_mdeg(theta_mdeg: float | np.ndarray, conc_uM: float, path_mm: float, n_res: int) -> float | np.ndarray:
    """
    [θ]_MRE (deg·cm^2·dmol^-1·res^-1)
    theta_mdeg -> theta_deg = mdeg/1000
    [θ] = (100 * theta_deg)/(c_M * l_cm) = (0.1 * theta_mdeg)/(c_M*l_cm)
    divide by n_res to get MRE
    """
    c_M = conc_uM * 1e-6
    l_cm = path_mm / 10.0
    if c_M <= 0 or l_cm <= 0 or n_res <= 0:
        raise ValueError("conc-uM, path-mm e nres devono essere > 0")
    return (0.1 * np.asarray(theta_mdeg, dtype=float)) / (c_M * l_cm * n_res)


def helicity_from_theta222_mre(theta222_mre: float, n_res: int, theta_c: float = -2000.0, clamp01: bool = True) -> float:
    """
    Riferimento classico:
      [θ]_H,222 = -40000 * (1 - 2.5/n)
      [θ]_C,222 ~ -2000
    f = (obs - coil) / (helix - coil)
    """
    theta_h = -40000.0 * (1.0 - 2.5 / float(n_res))
    denom = (theta_h - theta_c)
    if denom == 0:
        raise ValueError("Denominatore nullo (theta_h == theta_c).")
    f = (theta222_mre - theta_c) / denom
    if clamp01:
        f = max(0.0, min(1.0, f))
    return float(f)


def beta_fraction_from_theta218_mre(
    theta218_mre: float,
    f_helix: float,
    n_res: int,
    *,
    theta_c_218: float = -2000.0,
    theta_b_218: float = -21000.0,
    helix_218_ratio: float = 0.80,
    clamp01: bool = True,
) -> float:
    """Stima frazione beta-sheet (proxy) usando il valore MRE a 218 nm.

    Modello a 3 stati (helix / beta / coil) e chiusura con:
        f_coil = 1 - f_helix - f_beta

    Si assume che la frazione elicoidale f_helix sia già stata stimata da 222 nm.
    A 218 nm si rimuove il contributo atteso da helix+coil e si attribuisce il
    residuo a beta:

        theta218 = fH*thetaH218 + fB*thetaB218 + (1-fH-fB)*thetaC218

    =>  fB = (theta218 - [fH*thetaH218 + (1-fH)*thetaC218]) / (thetaB218 - thetaC218)

    Note operative:
      - thetaH218 viene approssimato come thetaH222 * helix_218_ratio.
      - thetaH222 segue lo stesso riferimento usato nella funzione di elicità.
      - I valori di default (theta_b_218 e theta_c_218) sono euristici: utili per
        trend comparativi, non equivalgono a una deconvoluzione completa.
    """
    # Helix reference at 222 (stesso modello della helicity_from_theta222_mre)
    theta_h_222 = -40000.0 * (1.0 - 2.5 / float(n_res))
    theta_h_218 = float(theta_h_222) * float(helix_218_ratio)

    denom = float(theta_b_218 - theta_c_218)
    if denom == 0:
        raise ValueError("Denominatore nullo (theta_b_218 == theta_c_218).")

    # Contributo atteso da helix+coil a 218 nm, dato fH
    theta218_pred_hc = float(f_helix) * theta_h_218 + (1.0 - float(f_helix)) * float(theta_c_218)
    fB = (float(theta218_mre) - theta218_pred_hc) / denom

    if clamp01:
        # fisicamente: 0 <= fB <= 1 - fH
        fB = max(0.0, min(float(fB), 1.0 - float(f_helix)))
    return float(fB)


# -------------------------
# Guessers from filename
# -------------------------
def guess_conc_uM_from_name(name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*uM", name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def guess_path_mm_from_name(name: str) -> float | None:
    m = re.search(r"cuv\s*(\d+(?:\.\d+)?)\s*mm", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_nres_map(s: str) -> dict[str, int]:
    """
    "CB02:10,CB03:12" -> {"CB02":10,"CB03":12}
    match by substring in filename (case-insensitive).
    """
    out: dict[str, int] = {}
    if not s.strip():
        return out
    for part in s.split(","):
        k, v = part.split(":")
        out[k.strip().upper()] = int(v.strip())
    return out


def pick_nres_for_file(filename: str, default_nres: int | None, nres_map: dict[str, int]) -> int | None:
    fn = filename.upper()
    for key, val in nres_map.items():
        if key in fn:
            return val
    return default_nres


# -------------------------
# Batch
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Batch CD: baseline offset (opzionale), MRE, e frazione elicoidale da 222 nm. "
                    "Assume che i .txt siano già blank-corrected."
    )
    ap.add_argument("folder", type=str, help="Cartella con i .txt")
    ap.add_argument("--conc-uM", type=float, default=None,
                    help="Concentrazione in µM. Se omesso, prova a leggerla dal filename (es 50uM).")
    ap.add_argument("--path-mm", type=float, default=None,
                    help="Cammino ottico in mm. Se omesso, prova a leggerlo dal filename (es cuv1mm).")

    ap.add_argument("--nres", type=int, default=None,
                    help="Numero residui (default per tutti i file, se non usi nres-map).")
    ap.add_argument("--nres-map", type=str, default="",
                    help="Mappa nres per peptide, es: 'CB02:10,CB03:12,CB26:11' (match su filename).")

    ap.add_argument("--baseline-range", type=str, default="260,300",
                    help="Intervallo nm per baseline offset (media). Usa 'none' per disabilitare.")
    ap.add_argument("--smooth-window", type=int, default=1,
                    help="Finestra moving average (dispari). 1 = off.")
    ap.add_argument("--theta-c", type=float, default=-2000.0,
                    help="Coil reference [θ]_MRE(222) (default -2000).")

    # Beta / coil proxy (218 nm)
    ap.add_argument("--theta-c-218", type=float, default=-2000.0,
                    help="Coil reference [θ]_MRE(218) (default -2000).")
    ap.add_argument("--theta-b-218", type=float, default=-21000.0,
                    help="Beta-sheet reference [θ]_MRE(218) (default -21000).")
    ap.add_argument("--helix-218-ratio", type=float, default=0.80,
                    help="Approssimazione: [θ]_H(218) = [θ]_H(222) * ratio (default 0.80).")
    ap.add_argument("--no-clamp", action="store_true",
                    help="Non clampare f_H in [0,1].")

    ap.add_argument("--out", type=str, default="cd_summary.csv",
                    help="CSV riassuntivo (salvato nella cartella corrente).")
    ap.add_argument("--export-processed", action="store_true",
                    help="Esporta CSV processati in sottocartella 'processed_csv' (wl, mdeg, MRE).")

    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Cartella non valida: {folder}")

    txt_files = sorted([p for p in folder.glob("*.txt") if p.is_file()])
    if not txt_files:
        raise SystemExit(f"Nessun .txt trovato in: {folder}")

    nres_map = parse_nres_map(args.nres_map)

    do_baseline = args.baseline_range.lower() != "none"
    baseline_lohi = None
    if do_baseline:
        lo, hi = (float(x) for x in args.baseline_range.split(","))
        baseline_lohi = (lo, hi)

    processed_dir = Path("processed_csv")
    if args.export_processed:
        processed_dir.mkdir(exist_ok=True)

    rows = []
    for p in txt_files:
        spec = load_jasco_txt(p)

        if do_baseline and baseline_lohi is not None:
            spec = baseline_correct_offset(spec, baseline_lohi)

        if args.smooth_window and args.smooth_window > 1:
            spec = CDSpectrum(spec.wl_nm, moving_average(spec.cd_mdeg, args.smooth_window), spec.ht_v)

        # conc/path from args or filename
        conc = args.conc_uM if args.conc_uM is not None else guess_conc_uM_from_name(p.name)
        path_mm = args.path_mm if args.path_mm is not None else guess_path_mm_from_name(p.name)

        # nres from map or default
        nres = pick_nres_for_file(p.name, args.nres, nres_map)

        cd222 = interp_at(spec, 222.0)
        cd208 = interp_at(spec, 208.0)
        cd218 = interp_at(spec, 218.0)

        # Only compute MRE/helicity if all required are valid
        if (conc is not None and path_mm is not None and nres is not None
                and conc > 0 and path_mm > 0 and nres > 0):
            theta222_mre = float(mre_from_mdeg(cd222, conc, path_mm, nres))
            theta208_mre = float(mre_from_mdeg(cd208, conc, path_mm, nres))
            theta218_mre = float(mre_from_mdeg(cd218, conc, path_mm, nres))
            fH = helicity_from_theta222_mre(theta222_mre, nres, theta_c=args.theta_c, clamp01=not args.no_clamp)

            fB = beta_fraction_from_theta218_mre(
                theta218_mre,
                fH,
                nres,
                theta_c_218=args.theta_c_218,
                theta_b_218=args.theta_b_218,
                helix_218_ratio=args.helix_218_ratio,
                clamp01=not args.no_clamp,
            )
            fC = 1.0 - fH - fB
            if not args.no_clamp:
                # protezione numerica
                fC = max(0.0, min(1.0, fC))
        else:
            theta222_mre = np.nan
            theta208_mre = np.nan
            theta218_mre = np.nan
            fH = np.nan
            fB = np.nan
            fC = np.nan

        rows.append({
            "file": p.name,
            "conc_uM": conc if conc is not None else np.nan,
            "path_mm": path_mm if path_mm is not None else np.nan,
            "nres": nres if nres is not None else np.nan,
            "CD_222_mdeg": cd222,
            "CD_208_mdeg": cd208,
            "CD_218_mdeg": cd218,
            "thetaMRE_222": theta222_mre,
            "thetaMRE_208": theta208_mre,
            "thetaMRE_218": theta218_mre,
            "helix_fraction_222": fH,
            "beta_fraction_218": fB,
            "coil_fraction": fC,
        })

        if args.export_processed and np.isfinite(theta222_mre):
            mre_full = mre_from_mdeg(spec.cd_mdeg, float(conc), float(path_mm), int(nres))
            outp = processed_dir / (p.stem + "_processed.csv")
            header = "wavelength_nm,cd_mdeg,thetaMRE\n"
            data = np.column_stack([spec.wl_nm, spec.cd_mdeg, mre_full])
            outp.write_text(header, encoding="utf-8")
            with outp.open("a", encoding="utf-8") as f:
                np.savetxt(f, data, delimiter=",", fmt="%.6f")

    # Write summary CSV
    out = Path(args.out).resolve()
    cols = [
        "file", "conc_uM", "path_mm", "nres",
        "CD_222_mdeg", "CD_218_mdeg", "CD_208_mdeg",
        "thetaMRE_222", "thetaMRE_218", "thetaMRE_208",
        "helix_fraction_222", "beta_fraction_218", "coil_fraction",
    ]
    out.write_text(",".join(cols) + "\n", encoding="utf-8")
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("OK")
    print(f"Cartella analizzata: {folder}")
    print(f"Output summary: {out}")
    if args.export_processed:
        print(f"CSV processati: {processed_dir.resolve()}")
    if do_baseline:
        print(f"Baseline offset applicato: {baseline_lohi[0]}–{baseline_lohi[1]} nm")
    else:
        print("Baseline offset: DISABILITATO")


if __name__ == "__main__":
    main()