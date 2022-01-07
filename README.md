# 1. Kaggle competition: [G-Research Crypto Forecasting](https://www.kaggle.com/c/g-research-crypto-forecasting/overview)


![Data Frame preview](./pic/datapic.png)
<img src="./pic/assetlist.png" alt="Asset List" width="200" height="200"/>

The simultaneous activity of thousands of traders ensures that most **signals** will be transitory, persistent **alpha** will be exceptionally difficult to find, and the danger of **overfitting** will be considerable. In addition, since 2018, interest in the cryptomarket has exploded, so the **volatility** and **correlation structure** in our data are likely to be highly non-stationary. The successful contestant will pay careful attention to these considerations, and in the process gain valuable insight into the art and science of financial forecasting.

# 2. Github file info (updating)  

- [./script/](./script): scripts folder contains the utility script for LSTM, XGBoost, Paper presentation, general statistical inference.
- [./trainedNN](./trainedNN): store trained RNN/LSTM models(hidden).  
- [./trainedXGB](./trainedXGB): store trained XGBoost models(hidden).  


## 2.1 Statistical Inference  

- [statistical-analysis-additional.ipynb](./statistical-analysis-additional.ipynb): explore the crypto market by:   
  - frequency manipulation [script/morestates.py](./script/morestats.py)`-> ts_with_frequency()`
  - autocorrelation
  - time-series decomposition [script/morestates.py](./script/morestats.py)`> ts_decomp()`
  - stationarity tests `Augmented Dickey-Fuller test` 

## 2.2 Neural Network Forecasting  

- [RNN_forecasting.ipynb](./RNN_forecasting.ipynb): Do RNN forecasting on the single crypto `BTC` OHLCV.  
  - Tensorflow
  - Keras: `tensorflow.keras`
  - RNN-LSTM: `tensorflow.keras.layers.LSTM`
- [crypto_lstm.ipynb](./crypto_lstm.ipynb): Manage the Kaggle competition with LSTM forecasting on the 14 cryptocurrencies returns simulaneously.
  - Tensorflow
  - Keras

## 2.3 XGBoost/LigntGBM Forecasting (Core)  

- [crypto-xgb.ipynb](./crypto-xgb.ipynb)
  - `xgboost`: [model script](./script/XGB/xgbmodel.py)
  - `lightgbm`
  - `Hyperparameters CV-Tune`: [crypto-xgb-cv.ipynb](./crypto-xgb-cv.ipynb)

## 2.4 Trading Strategy (backtest simulation)  

- [MA_cross_strategy.ipynb](./MA_cross_strategy.ipynb): Moving Average Crossing example of **trading strategy**, **backtesting** and **evaluation**.
  - generate strategy signals: `script/strategy.py -> mac()`
  - backtest: `script/backtest.py -> bt()`
  - evaluation:  
    - Sharpe ratio
    - Maximum Drawdown
    - Compound Annual Growth Rate (CAGR)      
    - distribution of returns
    - trade-level metrics
- [cointegration_strategy.ipynb](./cointegration_strategy.ipynb): Cointegration Strategy
  - `statsmodels.tsa.stattools.coint`
  - $y_{t}-\beta x_{t}=u_{t}$  