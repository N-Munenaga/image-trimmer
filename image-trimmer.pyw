import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageTk
from pillow_heif import register_heif_opener
from rembg import remove, new_session

register_heif_opener()

class ImageItem(tk.Frame):
    def __init__(self, master, file_path, session):
        super().__init__(master, bg="#2b2b2b", bd=1, relief="flat")
        self.file_path = file_path
        self.session = session
        self.current_img = None 
        
        self.label = tk.Label(self, bg="#1e1e1e", cursor="hand2")
        self.label.pack(padx=2, pady=2)
        
        # 【操作】左クリック: 時計回り / 右クリック: 反時計回り
        self.label.bind("<Button-1>", lambda e: self.manual_rotate(-90))
        self.label.bind("<Button-3>", lambda e: self.manual_rotate(90))

    def auto_process(self):
        try:
            raw = ImageOps.exif_transpose(Image.open(self.file_path)).convert("RGBA")
            mask_img = remove(raw, session=self.session, only_mask=True)
            mask = np.array(mask_img)
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 10:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                raw = raw.rotate(angle, expand=True)

            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                pad = 40
                raw = raw.crop((max(0, x-pad), max(0, y-pad), min(raw.width, x+w+pad), min(raw.height, y+h+pad)))

            ms = max(raw.size)
            sq = Image.new("RGBA", (ms, ms), (0,0,0,0))
            sq.paste(raw, ((ms - raw.width)//2, (ms - raw.height)//2), raw)
            self.current_img = sq.resize((800, 800), Image.LANCZOS)
            self.refresh_preview()
            return True
        except:
            return False

    def manual_rotate(self, angle):
        if self.current_img:
            self.current_img = self.current_img.rotate(angle, expand=True)
            self.refresh_preview()

    def refresh_preview(self):
        p = self.current_img.copy()
        p.thumbnail((180, 180))
        self.tk_img = ImageTk.PhotoImage(p)
        self.label.config(image=self.tk_img)

class TrimmingGallery(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Image Trimmer")
        self.geometry("1100x850")
        self.configure(bg="#1e1e1e")
        self.session = new_session("u2net")
        self.items = []

        header = tk.Frame(self, bg="#1e1e1e")
        header.pack(fill="x", pady=15)
        
        tk.Button(header, text="写真を選択", command=self.load_files, 
                  bg="#2196F3", fg="white", font=("MS Gothic", 12, "bold"), height=2).pack(side="left", padx=20, expand=True, fill="x")
        
        tk.Button(header, text="結果を保存", command=self.save_all, 
                  bg="#4CAF50", fg="white", font=("MS Gothic", 12, "bold"), height=2).pack(side="left", padx=20, expand=True, fill="x")

        self.guide = tk.Label(self, text="【操作】 左クリック: 時計回り / 右クリック: 反時計回り", 
                         bg="#1e1e1e", fg="#ffffff", font=("MS Gothic", 11, "bold"))
        self.guide.pack(pady=5)

        container = tk.Frame(self, bg="#1e1e1e")
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container, bg="#1e1e1e", highlightthickness=0)
        self.scroll_frame = tk.Frame(self.canvas, bg="#1e1e1e")
        sb = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def load_files(self):
        """ファイル選択"""
        paths = filedialog.askopenfilenames(title="写真を選択してください", 
                                           filetypes=[("Images", "*.heic *.jpg *.jpeg *.png")])
        if not paths: return
        
        # 画面をリセット
        for item in self.items: item.destroy()
        self.items = []
        
        cols = 5 
        for i, f in enumerate(paths):
            item = ImageItem(self.scroll_frame, f, self.session)
            item.grid(row=i//cols, column=i%cols, padx=8, pady=8)
            if item.auto_process():
                self.items.append(item)
            self.update()

    def save_all(self):
        if not self.items: return
        # 最初の写真がある場所に保存フォルダを作る
        base_dir = os.path.dirname(self.items[0].file_path)
        out = os.path.join(base_dir, "整形済み")
        os.makedirs(out, exist_ok=True)

        for item in self.items:
            if item.current_img:
                p = os.path.join(out, os.path.splitext(os.path.basename(item.file_path))[0] + "_final.png")
                item.current_img.save(p)
        messagebox.showinfo("保存完了", f"「整形済み」フォルダに保存しました！：\n{out}")

if __name__ == "__main__":
    app = TrimmingGallery()
    app.mainloop()