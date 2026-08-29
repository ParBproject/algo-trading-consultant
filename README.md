# Algorithmic Trading Research Framework

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Research](https://img.shields.io/badge/Mode-Backtest_%26_Paper_Trading-2ea44f)](src/backtester.py)
[![Configuration](https://img.shields.io/badge/Configuration-YAML-cb171e)](config/example.yaml)

A modular Python framework for researching, backtesting, optimizing, and paper-trading systematic strategies. The project emphasizes reusable components, risk controls, and evidence-based strategy comparison.

## Capabilities

| Module | Capability |
|---|---|
| Market data | Historical price retrieval and preprocessing |
| Indicators | RSI, MACD, Bollinger Bands, ATR, ADX, volume, and OBV |
| Strategies | Mean reversion, momentum, and pairs trading |
| Backtesting | Trades, equity curve, drawdown, and benchmark comparison |
| Optimization | Grid search, Bayesian search, and walk-forward review |
| Risk | Position sizing, exposure limits, and protective controls |
| Execution | Paper-trading workflow and trade logging |

## Evidence

### Technical-Indicator Dashboard

![Technical indicator dashboard](screenshots/01_indicators_dashboard.png)

### Mean-Reversion Strategy

![Mean reversion backtest](screenshots/02_mean_reversion_strategy.png)

### Parameter Optimization

![Parameter optimization](screenshots/05_optimization.png)

### Performance Summary

![Strategy performance comparison](screenshots/06_performance_summary.png)

### Paper-Trading Session

![Paper trading session](screenshots/07_paper_trading.png)

## Architecture

~~~text
Market Data
    ↓
Indicators → Strategy Signals
    ↓
Risk Manager → Position Sizing
    ↓
Backtester / Paper Executor
    ↓
Metrics, Trade Log, Visual Reports
~~~

## Run Locally

~~~bash
git clone https://github.com/ParBproject/algo-trading-consultant.git
cd algo-trading-consultant

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Copy "config/example.yaml", adjust the research parameters, and use the notebooks for guided demonstrations or "src/run_bot.py" for the paper-trading workflow.

## Repository Structure

~~~text
algo-trading-consultant/
├── config/example.yaml
├── data/fetcher.py
├── src/
│   ├── indicators.py
│   ├── strategies.py
│   ├── backtester.py
│   ├── optimizer.py
│   ├── risk_manager.py
│   ├── executor.py
│   └── run_bot.py
├── notebooks/
├── screenshots/
└── requirements.txt
~~~

## Skills Demonstrated

Python architecture, quantitative research, technical indicators, backtesting, risk management, parameter optimization, configuration design, notebook-based analysis, and results communication.

## Risk Notice

This software is for education and research. Backtested performance is not a forecast, and paper trading does not replicate all live-market conditions. Real deployment would require broker-specific testing, secure secret management, monitoring, compliance review, and strict capital controls.
