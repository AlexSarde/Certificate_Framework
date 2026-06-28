import sys
import os
import json
import qdarktheme
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QFormLayout, QGroupBox, QTextEdit, 
                             QMessageBox, QCheckBox, QComboBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from engine import CertificateSpec, run_simulation

class SimulationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir, spec, risk_free_rate, dividend_yields, mode, corr_blend, n_paths, n_days):
        super().__init__()
        self.data_dir = data_dir
        self.spec = spec
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = dividend_yields
        self.mode = mode
        self.corr_blend = corr_blend
        self.n_paths = n_paths
        self.n_days = n_days

    def run(self):
        try:
            res = run_simulation(
                data_dir=self.data_dir,
                spec=self.spec,
                risk_free_rate=self.risk_free_rate,
                dividend_yields=self.dividend_yields,
                mode=self.mode,
                corr_blend=self.corr_blend,
                n_paths=self.n_paths,
                n_days=self.n_days,
                seed=42,
                progress_callback=self.progress.emit
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class CertificateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Certificate Pricing Engine")
        self.resize(1100, 800)

        self.data_dir = ""
        self.last_results = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ---- LEFT PANEL (Inputs) ----
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Config Save/Load
        config_layout = QHBoxLayout()
        btn_save = QPushButton("Save Config")
        btn_load = QPushButton("Load Config")
        btn_save.clicked.connect(self.save_config)
        btn_load.clicked.connect(self.load_config)
        config_layout.addWidget(btn_save)
        config_layout.addWidget(btn_load)
        left_layout.addLayout(config_layout)

        # Data Selection
        data_group = QGroupBox("Data Source")
        data_layout = QVBoxLayout(data_group)
        self.btn_select_data = QPushButton("Select Data Directory")
        self.btn_select_data.clicked.connect(self.select_data_dir)
        self.lbl_data_dir = QLabel("No directory selected")
        self.lbl_data_dir.setWordWrap(True)
        data_layout.addWidget(self.btn_select_data)
        data_layout.addWidget(self.lbl_data_dir)
        left_layout.addWidget(data_group)

        # Market Params
        market_group = QGroupBox("Market Parameters")
        market_layout = QFormLayout(market_group)
        self.txt_rf = QLineEdit("0.0221")
        self.txt_divs = QLineEdit("STOCK1:0.0592, STOCK2:0.0815")
        market_layout.addRow("Risk-Free Rate:", self.txt_rf)
        market_layout.addRow("Div Yields (T:r,...):", self.txt_divs)
        left_layout.addWidget(market_group)

        # Certificate Spec
        cert_group = QGroupBox("Certificate Specs")
        cert_layout = QFormLayout(cert_group)
        self.txt_notional = QLineEdit("100")
        self.txt_barrier = QLineEdit("0.40")
        self.txt_coupon = QLineEdit("7.80")
        self.txt_freq = QLineEdit("30")
        self.txt_maturity = QLineEdit("504")
        
        self.chk_memory = QCheckBox("Memory Coupon")
        self.chk_autocall = QCheckBox("Autocall Enabled")
        self.chk_autocall.stateChanged.connect(self.toggle_autocall)
        
        self.txt_ac_trigger = QLineEdit("1.0")
        self.txt_ac_first = QLineEdit("252")
        self.txt_ac_freq = QLineEdit("252")
        self.toggle_autocall() # initial state
        
        cert_layout.addRow("Notional:", self.txt_notional)
        cert_layout.addRow("Barrier (% of S0):", self.txt_barrier)
        cert_layout.addRow("Coupon (Annual %):", self.txt_coupon)
        cert_layout.addRow("Coupon Freq (days):", self.txt_freq)
        cert_layout.addRow("Maturity (days):", self.txt_maturity)
        cert_layout.addRow("", self.chk_memory)
        cert_layout.addRow("", self.chk_autocall)
        cert_layout.addRow("AC Trigger (% S0):", self.txt_ac_trigger)
        cert_layout.addRow("AC First Day:", self.txt_ac_first)
        cert_layout.addRow("AC Freq (days):", self.txt_ac_freq)
        left_layout.addWidget(cert_group)

        # Simulation Params
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QFormLayout(sim_group)
        self.cbo_mode = QComboBox()
        self.cbo_mode.addItems(["risk_neutral", "real_world"])
        self.txt_paths = QLineEdit("10000")
        self.txt_corr_blend = QLineEdit("1.0")
        sim_layout.addRow("Pricing Mode:", self.cbo_mode)
        sim_layout.addRow("Number of Paths:", self.txt_paths)
        sim_layout.addRow("Corr. Override (x):", self.txt_corr_blend)
        left_layout.addWidget(sim_group)

        # Progress and Run
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.btn_run)
        
        self.btn_export = QPushButton("Export to Excel")
        self.btn_export.clicked.connect(self.export_excel)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        left_layout.addStretch()

        # ---- RIGHT PANEL (Outputs) ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setMaximumHeight(200)
        self.txt_results.setStyleSheet("font-family: Consolas; font-size: 13px;")
        right_layout.addWidget(self.txt_results)

        self.figure = Figure(facecolor="#2b2b2b")
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def toggle_autocall(self):
        enabled = self.chk_autocall.isChecked()
        self.txt_ac_trigger.setEnabled(enabled)
        self.txt_ac_first.setEnabled(enabled)
        self.txt_ac_freq.setEnabled(enabled)

    def select_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir = dir_path
            self.lbl_data_dir.setText(dir_path)

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json)")
        if not path:
            return
        state = {
            "data_dir": self.data_dir,
            "rf": self.txt_rf.text(),
            "divs": self.txt_divs.text(),
            "notional": self.txt_notional.text(),
            "barrier": self.txt_barrier.text(),
            "coupon": self.txt_coupon.text(),
            "freq": self.txt_freq.text(),
            "maturity": self.txt_maturity.text(),
            "memory": self.chk_memory.isChecked(),
            "autocall": self.chk_autocall.isChecked(),
            "ac_trigger": self.txt_ac_trigger.text(),
            "ac_first": self.txt_ac_first.text(),
            "ac_freq": self.txt_ac_freq.text(),
            "mode": self.cbo_mode.currentText(),
            "paths": self.txt_paths.text(),
            "corr_blend": self.txt_corr_blend.text()
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
        QMessageBox.information(self, "Saved", "Configuration saved successfully.")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.data_dir = state.get("data_dir", "")
            self.lbl_data_dir.setText(self.data_dir if self.data_dir else "No directory selected")
            self.txt_rf.setText(state.get("rf", "0.0221"))
            self.txt_divs.setText(state.get("divs", ""))
            self.txt_notional.setText(state.get("notional", "100"))
            self.txt_barrier.setText(state.get("barrier", "0.40"))
            self.txt_coupon.setText(state.get("coupon", "7.80"))
            self.txt_freq.setText(state.get("freq", "30"))
            self.txt_maturity.setText(state.get("maturity", "504"))
            self.chk_memory.setChecked(state.get("memory", False))
            self.chk_autocall.setChecked(state.get("autocall", False))
            self.txt_ac_trigger.setText(state.get("ac_trigger", "1.0"))
            self.txt_ac_first.setText(state.get("ac_first", "252"))
            self.txt_ac_freq.setText(state.get("ac_freq", "252"))
            self.cbo_mode.setCurrentText(state.get("mode", "risk_neutral"))
            self.txt_paths.setText(state.get("paths", "10000"))
            self.txt_corr_blend.setText(state.get("corr_blend", "1.0"))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def run_simulation(self):
        if not self.data_dir:
            QMessageBox.warning(self, "Missing Data", "Please select a data directory containing CSV files.")
            return

        try:
            rf = float(self.txt_rf.text())
            div_str = self.txt_divs.text().strip()
            dividend_yields = {}
            if div_str:
                for pair in div_str.split(","):
                    if ":" in pair:
                        t, y = pair.split(":")
                        dividend_yields[t.strip()] = float(y.strip())
            
            spec = CertificateSpec(
                notional=float(self.txt_notional.text()),
                barrier_pct=float(self.txt_barrier.text()),
                coupon_annual_pct=float(self.txt_coupon.text()),
                coupon_freq_days=int(self.txt_freq.text()),
                maturity_days=int(self.txt_maturity.text()),
                memory_coupon=self.chk_memory.isChecked(),
                autocall=self.chk_autocall.isChecked(),
                autocall_trigger_pct=float(self.txt_ac_trigger.text()),
                autocall_first_day=int(self.txt_ac_first.text()),
                autocall_freq_days=int(self.txt_ac_freq.text()),
            )
            mode = self.cbo_mode.currentText()
            corr_blend = float(self.txt_corr_blend.text())
            n_paths = int(self.txt_paths.text())
            n_days = spec.maturity_days
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input format: {e}")
            return

        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_run.setText("Computing...")
        self.txt_results.setText("Running Monte Carlo simulation, please wait...")
        self.progress_bar.setValue(0)
        self.last_results = None

        self.worker = SimulationWorker(self.data_dir, spec, rf, dividend_yields, mode, corr_blend, n_paths, n_days)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Simulation Error", error_msg)
        self.txt_results.setText(f"Error: {error_msg}")

    def on_simulation_finished(self, res):
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_run.setText("Run Simulation")
        self.progress_bar.setValue(100)
        self.last_results = res

        summary = res["summary"]
        text = "=== Simulation Results ===\n"
        if "fair_value" in summary and not pd.isna(summary["fair_value"]):
            text += f"Fair Value (Discounted): {summary['fair_value']:.4f} (Std Error: {summary.get('fv_std_error', 0):.4f})\n"
        text += f"Expected Payoff: {summary.get('expected_payoff', 0):.2f}\n"
        text += f"Prob of Profit: {summary.get('prob_profit', 0):.2%}\n"
        text += f"Prob Loss: {summary.get('prob_loss', 0):.2%}\n"
        text += f"Prob Barrier Breach: {summary.get('prob_breach_barrier', 0):.2%}\n"
        text += f"Prob Autocall: {summary.get('prob_autocall', 0):.2%}\n"
        
        self.txt_results.setText(text)
        self.plot_results(res)

    def plot_results(self, res):
        self.figure.clear()
        
        # Colors for dark theme
        bg_color = "#1e1e1e"
        text_color = "#e0e0e0"
        self.figure.patch.set_facecolor(bg_color)
        
        ax1 = self.figure.add_subplot(121)
        ax1.set_facecolor(bg_color)
        ax1.hist(res["path_df"]["pnl"], bins=50, color="#2196F3", alpha=0.8)
        ax1.set_title("PnL Distribution", color=text_color)
        ax1.set_xlabel("PnL", color=text_color)
        ax1.tick_params(colors=text_color)
        ax1.spines['bottom'].set_color(text_color)
        ax1.spines['left'].set_color(text_color)
        
        tickers = res["tickers"]
        if tickers:
            t = tickers[0]
            P = res["sim_prices"][t]
            ax2 = self.figure.add_subplot(122)
            ax2.set_facecolor(bg_color)
            import numpy as np
            sample = np.random.choice(P.shape[1], size=min(30, P.shape[1]), replace=False)
            days = np.arange(P.shape[0])
            ax2.plot(days, P[:, sample], alpha=0.4, lw=1)
            ax2.set_title(f"Simulated Paths ({t})", color=text_color)
            ax2.set_xlabel("Days", color=text_color)
            ax2.tick_params(colors=text_color)
            ax2.spines['bottom'].set_color(text_color)
            ax2.spines['left'].set_color(text_color)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def export_excel(self):
        if not self.last_results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "Certificate_Report.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
            
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                # Summary Sheet
                summary_df = pd.Series(self.last_results["summary"], name="Value").to_frame()
                summary_df.to_excel(xw, sheet_name="Summary")
                
                # Path DF Sheet
                path_df = self.last_results["path_df"]
                path_df.to_excel(xw, sheet_name="Path_Details", index=False)
                
            QMessageBox.information(self, "Exported", f"Successfully exported to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    window = CertificateApp()
    window.show()
    sys.exit(app.exec())
