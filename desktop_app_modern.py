"""Stock Price Predictor - modern dark desktop UI (customtkinter + Matplotlib)."""
import queue
import threading
from tkinter import filedialog

import customtkinter as ctk
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from predictor import build_features, load_data, predict_next, train_models

# ------------------------------------------------------------------ theme
BG = "#0e1117"
SIDEBAR = "#151922"
CARD = "#1b2030"
BORDER = "#262c40"
TEXT = "#e6e9f2"
MUTED = "#8a93ab"
ACCENT = "#4f8cff"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
PURPLE = "#a78bfa"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def font(size=13, weight="normal"):
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


def style_axes(fig, ax):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=BORDER, alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def style_legend(ax):
    leg = ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=TEXT)
    return leg


class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=BG)
        self.title("Stock Price Predictor")
        self.geometry("1240x780")
        self.minsize(1050, 680)

        self.q = queue.Queue()
        self.raw = self.df = self.bundle = None
        self.ticker = ""

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ---------------------------------------------------------- sidebar
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=270, corner_radius=0, fg_color=SIDEBAR)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)

        ctk.CTkLabel(sb, text="Stock Predictor", font=font(22, "bold"), text_color=TEXT).pack(
            anchor="w", padx=22, pady=(26, 0)
        )
        ctk.CTkLabel(sb, text="KNN price & direction model", font=font(12), text_color=MUTED).pack(
            anchor="w", padx=22, pady=(0, 22)
        )

        ctk.CTkLabel(sb, text="TICKER", font=font(11, "bold"), text_color=MUTED).pack(
            anchor="w", padx=22
        )
        self.ticker_var = ctk.StringVar(value="AAPL")
        self.entry = ctk.CTkEntry(
            sb, textvariable=self.ticker_var, height=40, font=font(15, "bold"),
            fg_color=CARD, border_color=BORDER, corner_radius=8,
            placeholder_text="e.g. RELIANCE.NS",
        )
        self.entry.pack(fill="x", padx=22, pady=(4, 6))
        self.entry.bind("<Return>", lambda _e: self.on_predict())
        ctk.CTkLabel(
            sb, text="Indian stocks need .NS or .BO", font=font(11), text_color=MUTED
        ).pack(anchor="w", padx=22, pady=(0, 14))

        # quick picks
        qp = ctk.CTkFrame(sb, fg_color="transparent")
        qp.pack(fill="x", padx=18, pady=(0, 18))
        for sym in ("RELIANCE.NS", "TCS.NS", "AAPL"):
            ctk.CTkButton(
                qp, text=sym.replace(".NS", ""), width=70, height=26, font=font(11),
                fg_color=CARD, hover_color=BORDER, text_color=MUTED, corner_radius=13,
                command=lambda s=sym: self._pick(s),
            ).pack(side="left", padx=4)

        self.years_lbl = self._slider_row(sb, "History (years)", 3, 15, 12, 10, self._years_changed)
        self.test_lbl = self._slider_row(sb, "Test period (%)", 10, 40, 6, 20, self._test_changed)
        self.years = 10
        self.test_pct = 20

        self.auto_k = ctk.CTkSwitch(
            sb, text="Auto K (√n rule)", font=font(13), text_color=TEXT,
            command=self._toggle_k, progress_color=ACCENT,
        )
        self.auto_k.select()
        self.auto_k.pack(anchor="w", padx=22, pady=(14, 4))
        self.k_lbl = ctk.CTkLabel(sb, text="K = 51", font=font(12), text_color=MUTED)
        self.k_slider = ctk.CTkSlider(
            sb, from_=3, to=201, number_of_steps=99, command=self._k_changed,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#6ea1ff",
        )
        self.k_slider.set(51)
        self.k_value = 51

        self.btn = ctk.CTkButton(
            sb, text="Predict", height=44, font=font(15, "bold"), corner_radius=10,
            fg_color=ACCENT, hover_color="#3b74e0", command=self.on_predict,
        )
        self.btn.pack(side="bottom", fill="x", padx=22, pady=(0, 22))
        self.progress = ctk.CTkProgressBar(sb, mode="indeterminate", height=4, progress_color=ACCENT)
        self.status = ctk.CTkLabel(
            sb, text="Enter a ticker and press Predict.", font=font(12), text_color=MUTED,
            wraplength=220, justify="left",
        )
        self.status.pack(side="bottom", anchor="w", padx=22, pady=(0, 10))
        self.progress.pack(side="bottom", fill="x", padx=22, pady=(0, 8))
        self.progress.set(0)

    def _slider_row(self, parent, title, lo, hi, steps, init, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=22, pady=(4, 0))
        ctk.CTkLabel(row, text=title, font=font(13), text_color=TEXT).pack(side="left")
        val = ctk.CTkLabel(row, text=str(init), font=font(13, "bold"), text_color=ACCENT)
        val.pack(side="right")
        s = ctk.CTkSlider(
            parent, from_=lo, to=hi, number_of_steps=steps, command=cmd,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#6ea1ff",
        )
        s.set(init)
        s.pack(fill="x", padx=22, pady=(4, 6))
        return val

    def _pick(self, sym):
        self.ticker_var.set(sym)
        self.on_predict()

    def _years_changed(self, v):
        self.years = int(round(v))
        self.years_lbl.configure(text=str(self.years))

    def _test_changed(self, v):
        self.test_pct = int(round(v / 5) * 5)
        self.test_lbl.configure(text=str(self.test_pct))

    def _k_changed(self, v):
        self.k_value = int(round(v))
        if self.k_value % 2 == 0:
            self.k_value += 1
        self.k_lbl.configure(text=f"K = {self.k_value}")

    def _toggle_k(self):
        if self.auto_k.get():
            self.k_lbl.pack_forget()
            self.k_slider.pack_forget()
        else:
            self.k_lbl.pack(anchor="w", padx=22)
            self.k_slider.pack(fill="x", padx=22, pady=(2, 6))

    # ------------------------------------------------------------- main
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=(18, 18), pady=(14, 14))
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.h_title = ctk.CTkLabel(head, text="No prediction yet", font=font(24, "bold"), text_color=TEXT)
        self.h_title.pack(anchor="w")
        self.h_sub = ctk.CTkLabel(
            head, text="Pick a stock on the left to get started.", font=font(13), text_color=MUTED
        )
        self.h_sub.pack(anchor="w")

        self.tabs = ctk.CTkTabview(
            main, fg_color=BG, segmented_button_fg_color=CARD,
            segmented_button_selected_color=ACCENT, segmented_button_selected_hover_color="#3b74e0",
            segmented_button_unselected_color=CARD, segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT,
        )
        self.tabs.grid(row=1, column=0, sticky="nsew")
        for name in ("Prediction", "Performance", "What-if", "Data"):
            self.tabs.add(name)
        self._build_pred(self.tabs.tab("Prediction"))
        self._build_perf(self.tabs.tab("Performance"))
        self._build_whatif(self.tabs.tab("What-if"))
        self._build_data(self.tabs.tab("Data"))

    def _card(self, parent, title, col):
        c = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        c.grid(row=0, column=col, sticky="nsew", padx=6, pady=(4, 8))
        ctk.CTkLabel(c, text=title.upper(), font=font(11, "bold"), text_color=MUTED).pack(
            anchor="w", padx=16, pady=(12, 0)
        )
        val = ctk.CTkLabel(c, text="-", font=font(26, "bold"), text_color=TEXT)
        val.pack(anchor="w", padx=16)
        sub = ctk.CTkLabel(c, text=" ", font=font(12), text_color=MUTED)
        sub.pack(anchor="w", padx=16, pady=(0, 12))
        return val, sub

    def _chart_card(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        fig = Figure(figsize=(8, 4), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        return frame, fig, canvas

    def _build_pred(self, tab):
        tab.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="c")
        tab.grid_rowconfigure(1, weight=1)
        self.c_last, self.c_last_sub = self._card(tab, "Last close", 0)
        self.c_pred, self.c_pred_sub = self._card(tab, "Predicted next close", 1)
        self.c_dir, self.c_dir_sub = self._card(tab, "Direction", 2)
        self.c_prob, self.c_prob_sub = self._card(tab, "Probability of up move", 3)

        frame, self.fig1, self.canvas1 = self._chart_card(tab)
        frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=6, pady=(0, 6))
        ctk.CTkLabel(
            tab,
            text="Daily stock moves are close to random. If direction accuracy (Performance tab) is not "
                 "clearly above the baseline, treat this as a demo, not a signal. Not financial advice.",
            font=font(11), text_color=AMBER, wraplength=900, justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 2))

    def _build_perf(self, tab):
        tab.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="c")
        tab.grid_rowconfigure(2, weight=1)
        self.perf_info = ctk.CTkLabel(tab, text="", font=font(12), text_color=MUTED)
        self.perf_info.grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 2))
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.grid(row=1, column=0, columnspan=4, sticky="ew")
        row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="p")
        self.m_mae, self.m_mae_sub = self._card(row, "MAE (price error)", 0)
        self.m_rmse, self.m_rmse_sub = self._card(row, "RMSE", 1)
        self.m_dir, self.m_dir_sub = self._card(row, "Direction accuracy", 2)
        self.m_r2, self.m_r2_sub = self._card(row, "R² / MAPE", 3)
        frame, self.fig2, self.canvas2 = self._chart_card(tab)
        frame.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=6, pady=(0, 6))
        ctk.CTkLabel(
            tab,
            text="Honest test: MAE should beat the naive baseline (tomorrow = today) and direction "
                 "accuracy should beat the majority-class baseline.",
            font=font(11), text_color=MUTED, wraplength=900, justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=8)

    def _build_whatif(self, tab):
        box = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        box.pack(anchor="nw", padx=6, pady=6)
        ctk.CTkLabel(
            box, text="Replace the latest day's values and re-run the prediction.",
            font=font(13), text_color=MUTED,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))
        self.w_vars = {}
        for i, name in enumerate(["Open", "High", "Low", "Close", "Volume"]):
            ctk.CTkLabel(box, text=name, font=font(13), text_color=TEXT, width=70, anchor="w").grid(
                row=i + 1, column=0, padx=(18, 6), pady=5, sticky="w"
            )
            v = ctk.StringVar()
            ctk.CTkEntry(
                box, textvariable=v, width=200, height=34, fg_color=BG, border_color=BORDER
            ).grid(row=i + 1, column=1, padx=(0, 18), pady=5, sticky="w")
            self.w_vars[name] = v
        ctk.CTkButton(
            box, text="Predict from these values", height=38, font=font(13, "bold"),
            fg_color=ACCENT, hover_color="#3b74e0", command=self.on_whatif,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 8))
        self.w_out = ctk.CTkLabel(box, text="", font=font(15, "bold"), text_color=TEXT, justify="left")
        self.w_out.grid(row=7, column=0, columnspan=2, sticky="w", padx=18, pady=(4, 16))

    def _build_data(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(bar, text="Last 60 trading days", font=font(13), text_color=MUTED).pack(side="left", padx=6)
        ctk.CTkButton(
            bar, text="Export CSV", width=110, height=30, fg_color=CARD, hover_color=BORDER,
            border_width=1, border_color=BORDER, command=self.on_export,
        ).pack(side="right", padx=6)
        self.table = ctk.CTkTextbox(
            tab, fg_color=CARD, border_width=1, border_color=BORDER, corner_radius=12,
            font=ctk.CTkFont(family="Consolas", size=12), text_color=TEXT, wrap="none",
        )
        self.table.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.table.configure(state="disabled")

    # ---------------------------------------------------------- actions
    def on_predict(self):
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            self._set_status("Please enter a ticker symbol.", RED)
            return
        k = None if self.auto_k.get() else self.k_value
        self.btn.configure(state="disabled", text="Working...")
        self._set_status(f"Downloading {ticker} and training...", MUTED)
        self.progress.start()
        threading.Thread(
            target=self._worker, args=(ticker, self.years, k, self.test_pct / 100), daemon=True
        ).start()
        self.after(100, self._poll)

    def _worker(self, ticker, years, k, test):
        try:
            raw = load_data(ticker, years)
            df = build_features(raw)
            bundle = train_models(df, k=k, test_size=test)
            res = predict_next(bundle, df)
            self.q.put(("ok", ticker, raw, df, bundle, res))
        except Exception as e:
            self.q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self.progress.stop()
        self.progress.set(0)
        self.btn.configure(state="normal", text="Predict")
        if msg[0] == "err":
            self._set_status(msg[1], RED)
            return
        _, self.ticker, self.raw, self.df, self.bundle, res = msg
        self._render(res)
        self._set_status(f"Done. K = {self.bundle.k}", GREEN)

    def _set_status(self, text, color):
        self.status.configure(text=text, text_color=color)

    def _render(self, res):
        up = res["direction"] == "UP"
        col = GREEN if up else RED
        m = self.bundle.metrics

        self.h_title.configure(text=self.ticker)
        self.h_sub.configure(text=f"Forecast for the trading day after {res['as_of']:%d %b %Y}")

        self.c_last.configure(text=f"{res['last_close']:,.2f}")
        self.c_last_sub.configure(text=f"{res['as_of']:%d %b %Y}")
        self.c_pred.configure(text=f"{res['pred_close']:,.2f}", text_color=col)
        self.c_pred_sub.configure(text=f"{res['pred_return_pct']:+.2f}% expected", text_color=col)
        self.c_dir.configure(text=("▲ UP" if up else "▼ DOWN"), text_color=col)
        self.c_dir_sub.configure(text="classifier output")
        self.c_prob.configure(text=f"{res['prob_up'] * 100:.0f}%")
        self.c_prob_sub.configure(text="chance price rises")

        # price chart
        recent = self.df.tail(180)
        self.fig1.clear()
        ax = self.fig1.add_subplot(111)
        style_axes(self.fig1, ax)
        ax.plot(recent.index, recent["Close"], color=ACCENT, lw=1.8, label="Close")
        ax.fill_between(recent.index, recent["Close"], recent["Close"].min() * 0.98, color=ACCENT, alpha=0.08)
        ax.plot(recent.index, recent["MA_5"], color=AMBER, lw=1, ls="--", label="MA 5")
        ax.plot(recent.index, recent["MA_20"], color=PURPLE, lw=1, ls="--", label="MA 20")
        nxt = res["as_of"] + pd.tseries.offsets.BDay(1)
        ax.plot([res["as_of"], nxt], [res["last_close"], res["pred_close"]], color=col, lw=2.2)
        ax.scatter([nxt], [res["pred_close"]], color=col, s=55, zorder=5, label="Prediction")
        ax.annotate(
            f"{res['pred_close']:,.2f}", (nxt, res["pred_close"]), xytext=(-6, 12),
            textcoords="offset points", color=col, fontsize=10, fontweight="bold", ha="right",
        )
        style_legend(ax)
        self.fig1.tight_layout()
        self.canvas1.draw()

        # performance
        self.perf_info.configure(
            text=f"Trained on {m['Train_Rows']:,} days from {m['Train_Start']:%b %Y}; tested on "
                 f"{m['Test_Rows']:,} days ({m['Test_Start']:%d %b %Y} to {m['Test_End']:%d %b %Y}). K = {self.bundle.k}."
        )
        beats_mae = m["MAE"] < m["Baseline_MAE_naive"]
        self.m_mae.configure(text=f"{m['MAE']:,.2f}")
        self.m_mae_sub.configure(
            text=f"naive baseline {m['Baseline_MAE_naive']:,.2f}",
            text_color=GREEN if beats_mae else AMBER,
        )
        self.m_rmse.configure(text=f"{m['RMSE']:,.2f}")
        self.m_rmse_sub.configure(text="same units as price")
        edge = m["Direction_Accuracy_pct"] - m["Baseline_Direction_pct"]
        self.m_dir.configure(text=f"{m['Direction_Accuracy_pct']:.1f}%")
        self.m_dir_sub.configure(
            text=f"{edge:+.1f} pts vs baseline ({m['Baseline_Direction_pct']:.1f}%)",
            text_color=GREEN if edge > 1 else AMBER,
        )
        self.m_r2.configure(text=f"{m['R2']:.3f}")
        self.m_r2_sub.configure(text=f"MAPE {m['MAPE_pct']:.2f}%")

        tf = self.bundle.test_frame
        self.fig2.clear()
        ax2 = self.fig2.add_subplot(111)
        style_axes(self.fig2, ax2)
        ax2.plot(tf.index, tf["Actual"], color=ACCENT, lw=1.5, label="Actual next-day close")
        ax2.plot(tf.index, tf["Predicted"], color=AMBER, lw=1, ls="--", label="Predicted")
        style_legend(ax2)
        self.fig2.tight_layout()
        self.canvas2.draw()

        # what-if
        last = self.raw.iloc[-1]
        for name, v in self.w_vars.items():
            v.set(f"{last[name]:.0f}" if name == "Volume" else f"{last[name]:.2f}")
        self.w_out.configure(text="")

        # data table
        cols = ["Date", "Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "Ret %"]
        lines = ["".join(f"{c:>13}" if c != "Date" else f"{c:<13}" for c in cols)]
        for ts, r in self.df.tail(60).iloc[::-1].iterrows():
            ma5 = "" if pd.isna(r["MA_5"]) else f"{r['MA_5']:.2f}"
            ma20 = "" if pd.isna(r["MA_20"]) else f"{r['MA_20']:.2f}"
            ret = "" if pd.isna(r["Daily_Return"]) else f"{r['Daily_Return'] * 100:+.2f}"
            vals = [f"{ts:%Y-%m-%d}", f"{r['Open']:.2f}", f"{r['High']:.2f}", f"{r['Low']:.2f}",
                    f"{r['Close']:.2f}", f"{r['Volume']:,.0f}", ma5, ma20, ret]
            lines.append("".join(f"{v:>13}" if i else f"{v:<13}" for i, v in enumerate(vals)))
        self.table.configure(state="normal")
        self.table.delete("1.0", "end")
        self.table.insert("1.0", "\n".join(lines))
        self.table.configure(state="disabled")

    def on_whatif(self):
        if self.raw is None:
            self.w_out.configure(text="Run a prediction first.", text_color=AMBER)
            return
        try:
            vals = {k: float(v.get().replace(",", "")) for k, v in self.w_vars.items()}
        except ValueError:
            self.w_out.configure(text="All fields must be numbers.", text_color=RED)
            return
        o, h, l, c = vals["Open"], vals["High"], vals["Low"], vals["Close"]
        if min(o, h, l, c) <= 0 or vals["Volume"] < 0:
            self.w_out.configure(text="Prices must be positive.", text_color=RED)
            return
        if not (l <= min(o, c) and h >= max(o, c)):
            self.w_out.configure(text="High must be the highest price and Low the lowest.", text_color=RED)
            return
        mod = self.raw.copy()
        for k, v in vals.items():
            mod.iloc[-1, mod.columns.get_loc(k)] = v
        try:
            r = predict_next(self.bundle, build_features(mod))
        except Exception as e:
            self.w_out.configure(text=str(e), text_color=RED)
            return
        up = r["direction"] == "UP"
        self.w_out.configure(
            text=f"Next close: {r['pred_close']:,.2f} ({r['pred_return_pct']:+.2f}%)\n"
                 f"{'▲ UP' if up else '▼ DOWN'}  |  Probability of up move: {r['prob_up'] * 100:.0f}%",
            text_color=GREEN if up else RED,
        )

    def on_export(self):
        if self.df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=f"{self.ticker}_features.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            self.df.to_csv(path)
            self._set_status(f"Saved {path}", GREEN)


if __name__ == "__main__":
    App().mainloop()
