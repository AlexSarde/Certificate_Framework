<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Certificate Pricing Framework</h3>

  <p align="center">
    An institutional-grade Python application for pricing and simulating Worst-Of Barrier Certificates using GARCH(1,1)-t volatility modeling and Cholesky-correlated Monte Carlo simulations.
    <br />
    <a href="https://github.com/AlexSarde/Certificate_Framework"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AlexSarde/Certificate_Framework/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/AlexSarde/Certificate_Framework/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#key-features">Key Features</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#standalone-executable">Standalone Executable</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

Pricing complex structured products like Worst-Of Barrier Certificates requires robust mathematical models. This framework bridges the gap between advanced quantitative finance and user-friendly software by offering a complete Desktop Application built to price derivatives in both **Risk-Neutral** (Fair Value) and **Real-World** (Risk/VaR) environments.

By utilizing a recursive **GARCH(1,1)-t** volatility model, the engine successfully captures the reality of "volatility clustering" and fat tails, ensuring that barrier breach probabilities and tail risks are not dangerously underestimated.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![PyQt6][PyQt6-badge]][PyQt6-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![Pandas][Pandas-badge]][Pandas-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Key Features

- **Dual Simulation Modes**: 
  - *Risk-Neutral*: Prices the theoretical Fair Value of the certificate using dividend-adjusted drift.
  - *Real-World*: Simulates historical drift to calculate the true probability of barrier breaches, expected payoffs, and PnL percentiles.
- **Complex Payoff Engine**: Supports continuous/discrete barriers, conditional coupons, **Memory Coupons**, and **Autocall** early redemption features.
- **True Path-Dependent Volatility**: Incorporates a fully stochastic, recursive GARCH model linked directly to the Monte Carlo shocks to maintain realistic volatility clustering.
- **Comprehensive Risk Analytics**: Calculates Portfolio VaR & Expected Shortfall (90, 95, 99), Max Drawdown, and single-asset tail risks.
- **Correlation Stress Testing**: Manually override historical correlation blending to stress-test your portfolio against market crashes.
- **Excel Export**: Export the full simulation cube (paths, PnL, coupons, stats) directly to `.xlsx`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You need Python 3.10 or higher installed on your system.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/AlexSarde/Certificate_Framework.git
   ```
2. Navigate to the directory
   ```sh
   cd Certificate_Framework
   ```
3. Install the required Python packages
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Start the graphical interface by running:
```sh
python app.py
```

1. **Select Data Source**: Click "Select Data Directory" and point it to the `Data` folder containing your historical daily CSV price feeds.
2. **Configure Market**: Enter the prevailing Risk-Free Rate and a comma-separated list of Dividend Yields for your tickers.
3. **Configure Certificate**: Input the product's term sheet (Barrier %, Coupon %, Frequency, Maturity, Autocall specs).
4. **Run Simulation**: Click "Run Simulation". The background engine will calibrate the GARCH models, simulate 10,000+ stochastic paths, and render the PnL histogram directly into the application canvas. 
5. **Export**: Click "Export to Excel" to save all the summary and path data to a spreadsheet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Standalone Executable

If you wish to share the application with users who do not have Python installed, you can package the entire application into a standalone Windows `.exe` using `PyInstaller`.

Run the included build script:
```sh
build.bat
```
*(Or simply run `python -m PyInstaller --noconfirm --onedir --windowed --name "Certificate_Pricer" --version-file "version.txt" app.py`)*

Once completed, the packaged application will be available in the generated `dist/Certificate_Pricer` folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

* `app.py`: The PyQt6 Desktop UI and main entry point.
* `engine.py`: The mathematical backend containing the GARCH calibrator, Path Simulator, and Payoff vectorization.
* `version.txt`: Windows executable metadata manifest.
* `Certificati/` & `New/`: The original Jupyter Notebook research environments where the mathematical prototypes are housed.
* `Data/`: Folder to place your historical daily CSV price feeds.
* `build.bat`: Script for PyInstaller executable generation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyQt6-badge]: https://img.shields.io/badge/PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white
[PyQt6-url]: https://riverbankcomputing.com/software/pyqt/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas-badge]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
