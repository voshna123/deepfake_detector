"""
Deepfake Detector – desktop GUI (Tkinter).
Auto-loads exports/detector.onnx on startup.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Canvas, StringVar, DoubleVar,
    filedialog, messagebox, ttk,
    END, LEFT, RIGHT, BOTH, X, Y, TOP, BOTTOM, W, E, N, S,
)
from typing import List

SUPPORTED_IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED_VIDEOS  = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SUPPORTED_MEDIA   = SUPPORTED_IMAGES + SUPPORTED_VIDEOS

MODEL_PATH = "exports/detector.onnx"

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#0f0f13"
SURFACE   = "#1a1a24"
SURFACE2  = "#22222f"
ACCENT    = "#7c6af7"          # purple
ACCENT2   = "#5b52d4"
REAL_CLR  = "#22d17a"          # green
FAKE_CLR  = "#f25c6e"          # red
TEXT      = "#e8e8f0"
SUBTEXT   = "#8888a8"
BORDER    = "#2e2e42"


class DeepfakeApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Detector")
        self.geometry("860x700")
        self.minsize(760, 620)
        self.configure(bg=BG)

        self._predictor = None
        self._result_queue: queue.Queue = queue.Queue()
        self._batch_results: List[dict] = []

        self._apply_styles()
        self._build_ui()
        self._poll_results()
        self._auto_load_model()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",
                     background=BG, foreground=TEXT,
                     font=("Segoe UI", 10))
        s.configure("TFrame",      background=BG)
        s.configure("Card.TFrame", background=SURFACE, relief="flat")
        s.configure("TLabel",      background=BG, foreground=TEXT)
        s.configure("Sub.TLabel",  background=BG, foreground=SUBTEXT,
                    font=("Segoe UI", 9))
        s.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
        s.configure("TEntry",
                     fieldbackground=SURFACE2, foreground=TEXT,
                     insertcolor=TEXT, bordercolor=BORDER,
                     lightcolor=BORDER, darkcolor=BORDER)
        s.configure("Accent.TButton",
                     background=ACCENT, foreground="#ffffff",
                     font=("Segoe UI", 10, "bold"),
                     borderwidth=0, focusthickness=0)
        s.map("Accent.TButton",
              background=[("active", ACCENT2), ("pressed", ACCENT2)])
        s.configure("Ghost.TButton",
                     background=SURFACE2, foreground=TEXT,
                     font=("Segoe UI", 10),
                     borderwidth=0, focusthickness=0)
        s.map("Ghost.TButton",
              background=[("active", BORDER)])
        s.configure("TProgressbar",
                     troughcolor=SURFACE2, background=ACCENT,
                     thickness=4, borderwidth=0)
        s.configure("Treeview",
                     background=SURFACE, foreground=TEXT,
                     fieldbackground=SURFACE, rowheight=28,
                     font=("Segoe UI", 9))
        s.configure("Treeview.Heading",
                     background=SURFACE2, foreground=SUBTEXT,
                     font=("Segoe UI", 9, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", ACCENT2)])
        s.configure("TScale",
                     background=BG, troughcolor=SURFACE2,
                     sliderlength=14, sliderrelief="flat")

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = Frame(self, bg=SURFACE, pady=0)
        hdr.pack(fill=X, side=TOP)

        inner_hdr = Frame(hdr, bg=SURFACE)
        inner_hdr.pack(fill=X, padx=28, pady=18)

        title_frame = Frame(inner_hdr, bg=SURFACE)
        title_frame.pack(side=LEFT)

        # Accent bar
        Frame(title_frame, bg=ACCENT, width=4, height=32).pack(side=LEFT, padx=(0, 12))

        Label(title_frame, text="Deepfake Detector",
              font=("Segoe UI", 18, "bold"),
              bg=SURFACE, fg=TEXT).pack(side=LEFT)

        Label(inner_hdr,
              text="MobileNetV3 + CBAM  ·  Trained on CelebDF & FaceForensics++",
              font=("Segoe UI", 9), bg=SURFACE, fg=SUBTEXT).pack(side=RIGHT, anchor=E)

        # Model status dot
        self._model_dot = Label(inner_hdr, text="●  Loading model…",
                                 font=("Segoe UI", 9), bg=SURFACE, fg=SUBTEXT)
        self._model_dot.pack(side=RIGHT, padx=(0, 20))

        # ── Thin separator ───────────────────────────────────────────────────
        Frame(self, bg=BORDER, height=1).pack(fill=X)

        # ── Content area ─────────────────────────────────────────────────────
        content = Frame(self, bg=BG)
        content.pack(fill=BOTH, expand=True, padx=28, pady=20)

        # ── Input card ───────────────────────────────────────────────────────
        inp_card = Frame(content, bg=SURFACE, bd=0)
        inp_card.pack(fill=X, pady=(0, 14))
        self._add_card_label(inp_card, "Input")

        inp_body = Frame(inp_card, bg=SURFACE)
        inp_body.pack(fill=X, padx=18, pady=(0, 16))

        path_row = Frame(inp_body, bg=SURFACE)
        path_row.pack(fill=X, pady=(0, 10))

        self.input_path_var = StringVar()
        path_entry = ttk.Entry(path_row, textvariable=self.input_path_var,
                               font=("Segoe UI", 10))
        path_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 8), ipady=5)

        ttk.Button(path_row, text="Image / Video",
                   style="Ghost.TButton",
                   command=self._browse_file).pack(side=LEFT, padx=(0, 6), ipady=4)
        ttk.Button(path_row, text="Folder (batch)",
                   style="Ghost.TButton",
                   command=self._browse_folder).pack(side=LEFT, ipady=4)

        btn_row = Frame(inp_body, bg=SURFACE)
        btn_row.pack(fill=X)

        ttk.Button(btn_row, text="  ▶   Analyse  ",
                   style="Accent.TButton",
                   command=self._run_analysis).pack(side=LEFT, ipady=6)

        # Threshold
        thresh_frame = Frame(btn_row, bg=SURFACE)
        thresh_frame.pack(side=RIGHT)
        Label(thresh_frame, text="Threshold", bg=SURFACE, fg=SUBTEXT,
              font=("Segoe UI", 9)).pack(side=LEFT, padx=(0, 8))
        self.threshold_var = DoubleVar(value=0.5)
        ttk.Scale(thresh_frame, from_=0.0, to=1.0,
                  variable=self.threshold_var, orient="horizontal", length=140,
                  command=self._on_threshold_change).pack(side=LEFT)
        self.thresh_label = Label(thresh_frame, text="0.50",
                                   bg=SURFACE, fg=TEXT,
                                   font=("Segoe UI", 9, "bold"), width=4)
        self.thresh_label.pack(side=LEFT, padx=(6, 0))

        # ── Progress bar (thin) ───────────────────────────────────────────────
        self.progress_bar = ttk.Progressbar(content, mode="determinate", length=400)
        self.progress_bar.pack(fill=X, pady=(0, 14))

        # ── Result card ───────────────────────────────────────────────────────
        res_card = Frame(content, bg=SURFACE)
        res_card.pack(fill=BOTH, expand=True)
        self._add_card_label(res_card, "Result")

        res_body = Frame(res_card, bg=SURFACE)
        res_body.pack(fill=BOTH, expand=True, padx=18, pady=(0, 16))

        # Top: verdict + confidence side by side
        top_row = Frame(res_body, bg=SURFACE)
        top_row.pack(fill=X, pady=(0, 12))

        # Verdict box
        verdict_box = Frame(top_row, bg=SURFACE2, width=180)
        verdict_box.pack(side=LEFT, padx=(0, 16))
        verdict_box.pack_propagate(False)

        self.verdict_label = Label(verdict_box, text="–",
                                    font=("Segoe UI", 40, "bold"),
                                    bg=SURFACE2, fg=SUBTEXT)
        self.verdict_label.pack(expand=True, pady=18)

        # Right side: confidence + detail
        right_col = Frame(top_row, bg=SURFACE)
        right_col.pack(side=LEFT, fill=BOTH, expand=True)

        conf_header = Frame(right_col, bg=SURFACE)
        conf_header.pack(fill=X, pady=(6, 4))
        Label(conf_header, text="REAL", bg=SURFACE, fg=REAL_CLR,
              font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        Label(conf_header, text="FAKE", bg=SURFACE, fg=FAKE_CLR,
              font=("Segoe UI", 9, "bold")).pack(side=RIGHT)

        self.conf_canvas = Canvas(right_col, height=10, bg=SURFACE2,
                                   bd=0, highlightthickness=0)
        self.conf_canvas.pack(fill=X, pady=(0, 10))

        # Probabilities row
        prob_row = Frame(right_col, bg=SURFACE)
        prob_row.pack(fill=X, pady=(0, 8))

        self.real_pct = Label(prob_row, text="Real: –",
                               font=("Segoe UI", 12, "bold"),
                               bg=SURFACE, fg=REAL_CLR)
        self.real_pct.pack(side=LEFT)

        self.fake_pct = Label(prob_row, text="Fake: –",
                               font=("Segoe UI", 12, "bold"),
                               bg=SURFACE, fg=FAKE_CLR)
        self.fake_pct.pack(side=RIGHT)

        self.detail_text = Label(right_col, text="Load a model, then select a file to analyse.",
                                  font=("Segoe UI", 9), bg=SURFACE, fg=SUBTEXT,
                                  wraplength=500, anchor=W, justify=LEFT)
        self.detail_text.pack(fill=X)

        # ── Separator ────────────────────────────────────────────────────────
        Frame(res_body, bg=BORDER, height=1).pack(fill=X, pady=(0, 10))

        # ── Batch table ───────────────────────────────────────────────────────
        table_frame = Frame(res_body, bg=SURFACE)
        table_frame.pack(fill=BOTH, expand=True)

        cols = ("File", "Label", "Fake %", "Real %")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=7)
        widths = {"File": 320, "Label": 90, "Fake %": 110, "Real %": 110}
        anchors = {"File": W, "Label": "center", "Fake %": "center", "Real %": "center"}
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor=anchors[col])

        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        sb.pack(side=RIGHT, fill=Y)

        # ── Status bar ────────────────────────────────────────────────────────
        Frame(self, bg=BORDER, height=1).pack(fill=X)
        status_bar = Frame(self, bg=SURFACE2, pady=6)
        status_bar.pack(fill=X, side=BOTTOM)

        self.status_var = StringVar(value="Initialising…")
        Label(status_bar, textvariable=self.status_var,
              font=("Segoe UI", 9), bg=SURFACE2, fg=SUBTEXT).pack(side=LEFT, padx=16)

        ttk.Button(status_bar, text="Export CSV",
                   style="Ghost.TButton",
                   command=self._export_csv).pack(side=RIGHT, padx=12, ipady=2)

    def _add_card_label(self, parent, text):
        hdr = Frame(parent, bg=SURFACE)
        hdr.pack(fill=X, padx=18, pady=(14, 10))
        Label(hdr, text=text, font=("Segoe UI", 11, "bold"),
              bg=SURFACE, fg=TEXT).pack(side=LEFT)

    # ── Model auto-load ───────────────────────────────────────────────────────

    def _auto_load_model(self):
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        try:
            from app.inference import DeepfakePredictor
            self._predictor = DeepfakePredictor(
                model_path=MODEL_PATH,
                face_detector="mtcnn",
                image_size=224,
                seq_len=8,
                threshold=self.threshold_var.get(),
            )
            self._result_queue.put(("model_ready", None))
        except Exception as exc:
            self._result_queue.put(("model_error", str(exc)))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_threshold_change(self, _=None):
        v = self.threshold_var.get()
        self.thresh_label.config(text=f"{v:.2f}")
        if self._predictor:
            self._predictor.threshold = v

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[("Media files", " ".join(f"*{e}" for e in SUPPORTED_MEDIA)),
                       ("All", "*.*")],
        )
        if path:
            self.input_path_var.set(path)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder for batch processing")
        if folder:
            self.input_path_var.set(folder)

    def _run_analysis(self):
        if self._predictor is None:
            messagebox.showwarning("Model loading", "Model is still loading. Please wait.")
            return
        path = self.input_path_var.get().strip()
        if not path:
            messagebox.showwarning("No input", "Please select a file or folder.")
            return

        self._predictor.threshold = self.threshold_var.get()
        self._batch_results.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)

        self._set_status("Analysing…")
        self.progress_bar["value"] = 0
        threading.Thread(target=self._analysis_thread, args=(path,), daemon=True).start()

    def _analysis_thread(self, path: str):
        try:
            if os.path.isdir(path):
                self._batch_analysis(path)
            else:
                self._single_analysis(path)
        except Exception as exc:
            self._result_queue.put(("error", str(exc)))

    def _single_analysis(self, path: str):
        ext = Path(path).suffix.lower()
        if ext in SUPPORTED_VIDEOS:
            def cb(cur, total):
                self._result_queue.put(("progress", int(100 * cur / total)))
            result = self._predictor.predict_video(path, progress_callback=cb)
        else:
            result = self._predictor.predict_image(path)
        result["path"] = path
        self._result_queue.put(("single_result", result))

    def _batch_analysis(self, folder: str):
        files = [str(p) for p in Path(folder).rglob("*")
                 if p.suffix.lower() in SUPPORTED_MEDIA]
        total = len(files)
        if total == 0:
            self._result_queue.put(("status", "No media files found in folder."))
            return
        for i, fp in enumerate(files):
            ext = Path(fp).suffix.lower()
            result = (self._predictor.predict_video(fp) if ext in SUPPORTED_VIDEOS
                      else self._predictor.predict_image(fp))
            result["path"] = fp
            self._result_queue.put(("batch_row", result))
            self._result_queue.put(("progress", int(100 * (i + 1) / total)))
        self._result_queue.put(("batch_done", total))

    # ── Result polling ────────────────────────────────────────────────────────

    def _poll_results(self):
        try:
            while True:
                msg_type, payload = self._result_queue.get_nowait()
                if msg_type == "model_ready":
                    self._model_dot.config(text="●  FP32 model ready", fg=REAL_CLR)
                    self._set_status("Model loaded. Select a file and click Analyse.")
                elif msg_type == "model_error":
                    self._model_dot.config(text="●  Model error", fg=FAKE_CLR)
                    self._set_status(f"Model error: {payload}")
                elif msg_type == "status":
                    self._set_status(str(payload))
                elif msg_type == "error":
                    messagebox.showerror("Error", str(payload))
                    self._set_status("Error — see dialog.")
                elif msg_type == "progress":
                    self.progress_bar["value"] = payload
                elif msg_type == "single_result":
                    self._show_single_result(payload)
                elif msg_type == "batch_row":
                    self._add_batch_row(payload)
                elif msg_type == "batch_done":
                    self._set_status(f"Batch complete — {payload} files processed.")
                    self.progress_bar["value"] = 100
        except Exception:
            pass
        finally:
            self.after(100, self._poll_results)

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show_single_result(self, result: dict):
        label    = result.get("label", "?")
        fake_p   = result.get("fake_probability", 0.0)
        real_p   = result.get("real_probability", 1.0)
        color    = FAKE_CLR if label == "FAKE" else REAL_CLR

        self.verdict_label.config(text=label, fg=color)
        self.real_pct.config(text=f"Real  {real_p:.1%}")
        self.fake_pct.config(text=f"Fake  {fake_p:.1%}")

        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width() or 400
        self.conf_canvas.delete("all")
        bar_w = int(w * real_p)
        self.conf_canvas.create_rectangle(0, 0, bar_w, 10, fill=REAL_CLR, outline="")
        self.conf_canvas.create_rectangle(bar_w, 0, w, 10, fill=FAKE_CLR, outline="")

        name = Path(result.get("path", "")).name
        self.detail_text.config(text=name)
        self._add_batch_row(result)
        self._set_status("Analysis complete.")
        self.progress_bar["value"] = 100

    def _add_batch_row(self, result: dict):
        self._batch_results.append(result)
        name   = Path(result.get("path", "?")).name
        label  = result.get("label", "?")
        fake_p = f"{result.get('fake_probability', 0.0):.1%}"
        real_p = f"{result.get('real_probability', 1.0):.1%}"
        tag    = "fake" if label == "FAKE" else "real"
        self.tree.insert("", END, values=(name, label, fake_p, real_p), tags=(tag,))
        self.tree.tag_configure("fake", foreground=FAKE_CLR)
        self.tree.tag_configure("real", foreground=REAL_CLR)

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _export_csv(self):
        if not self._batch_results:
            messagebox.showinfo("Nothing to export", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Save results",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "label", "fake_probability", "real_probability"])
            writer.writeheader()
            for row in self._batch_results:
                writer.writerow({
                    "path": row.get("path", ""),
                    "label": row.get("label", ""),
                    "fake_probability": f"{row.get('fake_probability', 0.0):.4f}",
                    "real_probability": f"{row.get('real_probability', 1.0):.4f}",
                })
        messagebox.showinfo("Exported", f"Saved to:\n{path}")


def main():
    app = DeepfakeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
