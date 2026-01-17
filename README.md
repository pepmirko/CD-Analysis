# CD Helicity Batch Analyzer

A small Python tool to batch-analyze **Circular Dichroism (CD)** spectra exported as text files (JASCO-style with an `XYDATA` section) and estimate the **α-helical fraction** from the **222 nm** band.

## Features

Given a folder of `.txt` CD files, the script:

- parses JASCO-style exports containing an `XYDATA` block (`wavelength  CD[mdeg]  (optional HT)`),
- optionally applies **baseline offset correction** (mean CD in a wavelength window, default `260–300 nm`),
- optionally applies smoothing (simple moving average),
- extracts **CD at 222 nm and 208 nm** via interpolation,
- converts CD (mdeg) to **Mean Residue Ellipticity** `[θ]MRE` (deg·cm²·dmol⁻¹·res⁻¹),
- estimates **helix fraction** from `[θ]MRE(222)` using a common reference model:
  - `[θ]H,222 = -40000 * (1 - 2.5/n)`
  - `[θ]C,222 ≈ -2000`
  - `fH = ([θ]obs - [θ]C)/([θ]H - [θ]C)`

> Assumption: your `.txt` files are already blank/buffer corrected (i.e., no blank subtraction is performed).

---

## Requirements

- Python 3.9+ (3.10/3.11/3.12 are fine)
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
