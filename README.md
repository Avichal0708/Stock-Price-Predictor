# Stock Price Predictor (app version)

Your notebook, turned into a Streamlit app: KNN regressor (next-day close) + KNN classifier (up/down),
trained automatically on live Yahoo Finance data.

## Desktop version (Windows window, no browser)
- **Run:** double-click `run_desktop.bat` (first run installs packages, then the app window opens).
- **Desktop shortcut:** after the first run, double-click `create_shortcut.bat`. A "Stock Predictor" icon appears
  on your Desktop and opens the app with no console window. Keep this folder where it is (the shortcut points to it).
- **Make an .exe:** double-click `build_exe.bat` on Windows. The result is `dist\StockPredictor\StockPredictor.exe`
  (share the whole `StockPredictor` folder). Needs internet at runtime for stock data.
- Files: `desktop_app_modern.py` (new dark UI) + `predictor.py` (model, shared with the web version).
  The older plain window is still there as `desktop_app.py` (`python desktop_app.py`).

## Web version (Streamlit)
**Windows:** double-click `run.bat` (first run installs everything), then open the link it prints
(usually http://localhost:8501).

**Manual:**
```
pip install -r requirements.txt
streamlit run app.py
```

Tickers: `AAPL`, `TSLA`, Indian stocks need a suffix: `RELIANCE.NS`, `TCS.NS`, `INFY.BO`.

## Files
- `predictor.py` - data download, features, model training, metrics, prediction
- `app.py` - the web UI (Prediction / Model performance / What-if / Data tabs)

## What I changed from the notebook (and why)
1. **Chronological train/test split** instead of random shuffle. Shuffling lets the model train on days
   next to its test days, which inflates accuracy.
2. **`scaler.transform` on test data** (notebook used `fit_transform` on test, scaling it with its own stats).
3. **Predicts next-day return, then converts to price.** KNN can't output prices above what it saw in
   training, which is why the notebook needed the `- 34` correction. That hack is gone.
4. **Scale-free features** (Open/Close, High/Close, MA ratios, volume ratio, daily return) so one model
   logic works for any stock. The MA columns you built but never used are now actually used.
5. **Direction bug fixed:** the notebook's direction function returned `prediction[0]` (undefined) and its
   UP/DOWN labels were reversed (class 1 = up).
6. **Zero/NaN handling:** invalid rows are dropped instead of filled with the column mean.
7. **K** = sqrt(training rows), made odd; adjustable in the sidebar.
8. **Honest metrics:** results are shown next to two baselines ("tomorrow = today" for price,
   "always guess majority" for direction).

## Reality check
Daily stock moves are close to random. Expect direction accuracy near 50 percent; if it isn't clearly above
the baseline in the Model performance tab, the prediction isn't a usable signal. Not financial advice.
