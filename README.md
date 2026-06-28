# Certificate Pricing Framework

A comprehensive, institutional-grade Python application for pricing and simulating **Worst-Of Barrier Certificates** and other complex structured products. 

This framework utilizes a high-performance **GARCH(1,1)-t** volatility model paired with **Cholesky-correlated Monte Carlo simulations** to accurately price multi-asset derivatives in both **Risk-Neutral** (Fair Value) and **Real-World** (Risk/VaR) modes.

---

## 🚀 Key Features

### Financial & Quantitative Edge
* **Dual Simulation Modes**: 
  * **Risk-Neutral**: Prices the mathematical Fair Value of the certificate using dividend-adjusted drift and daily GARCH variance scaling.
  * **Real-World**: Simulates historical drift to calculate the true probability of barrier breaches, expected payoffs, and PnL percentiles.
* **Complex Payoff Engine**: Supports continuous/discrete barriers, conditional coupons, **Memory Coupons**, and **Autocall** early redemption features.
* **Correlation Stress Testing**: Manually override historical correlation blending to stress-test your portfolio against extreme market crashes.

### Engineering & UI
* **Desktop Graphical Interface**: Built natively with **PyQt6**, featuring a premium trading-terminal dark mode (`qdarktheme`).
* **High Performance**: Built on highly vectorized NumPy arrays for rapid generation of 10,000+ paths. A background `QThread` ensures the UI remains buttery smooth during intense computations.
* **Excel Export**: One-click dump of the simulated path DataFrames and statistical summaries directly to an `.xlsx` report using `pandas` and `openpyxl`.
* **Save/Load Presets**: Save your certificate specs to `.json` files for instant reloading.

---

## 💻 Installation (Python)

If you wish to run the raw Python code, ensure you have Python 3.10+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/AlexSarde/Certificate_Framework.git
   cd Certificate_Framework
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

---

## 📦 Standalone Executable (No Python Required)

You can package the entire application into a standalone Windows `.exe` using `PyInstaller`. This allows you to share the application with clients or colleagues who do not have Python installed.

Simply run the included build script:
```bash
build.bat
```

Once completed, the packaged application will be available in the generated `dist/Certificate_Pricer` folder. Double click `Certificate_Pricer.exe` to run it!

---

## 📊 Usage Guide

1. **Select Data Source**: Click "Select Data Directory" and point it to the `Data` folder containing your underlying CSV files. The CSVs should have at least a `Date` and `Close` column.
2. **Configure Market**: Enter the prevailing Risk-Free Rate and a comma-separated list of Dividend Yields for your tickers (e.g., `STOCK1:0.05, STOCK2:0.02`).
3. **Configure Certificate**: Input the product's term sheet (Barrier %, Coupon %, Frequency, Maturity, Autocall specs).
4. **Run**: Click "Run Simulation". The background engine will calibrate the GARCH models, simulate the stochastic paths, and render the PnL histogram directly into the application canvas. 

---

## 🗂️ Project Structure

* **`app.py`**: The PyQt6 Desktop UI and main entry point.
* **`engine.py`**: The mathematical backend containing the GARCH calibrator, Path Simulator, and Payoff vectorization.
* **`Certificati/` & `New/`**: The original Jupyter Notebook research environment where the mathematical prototypes (like `Certificati_RN.ipynb`) are housed.
* **`Data/`**: Folder to place your historical daily CSV price feeds.
* **`build.bat`**: Script for PyInstaller executable generation.
