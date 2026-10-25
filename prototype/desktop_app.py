"""
Desktop GUI Prototype for Fruit Ripeness Detection System.
Built using Tkinter, Pillow, and SQLite.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import sys

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(__file__))
from database import init_prototype_db, authenticate, save_image_record

init_prototype_db()


class FruitRipenessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Automated Fruit Ripeness Detection - Prototype")
        self.root.geometry("450x400")
        self.root.resizable(False, False)
        self.current_user = None

        self.show_login_frame()

    def show_login_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("400x320")

        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(expand=True)

        tk.Label(frame, text="Fruit Ripeness System", font=("Helvetica", 16, "bold"), fg="#2c3e50").pack(pady=(0, 15))
        tk.Label(frame, text="Sign In to Your Account", font=("Helvetica", 10), fg="#7f8c8d").pack(pady=(0, 15))

        tk.Label(frame, text="Username:", anchor="w").pack(fill="x")
        self.entry_username = tk.Entry(frame, width=30)
        self.entry_username.insert(0, "user1")
        self.entry_username.pack(pady=(2, 10))

        tk.Label(frame, text="Password:", anchor="w").pack(fill="x")
        self.entry_password = tk.Entry(frame, show="*", width=30)
        self.entry_password.insert(0, "validPass")
        self.entry_password.pack(pady=(2, 15))

        btn_login = tk.Button(frame, text="Login", command=self.handle_login, bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"), width=20, pady=4)
        btn_login.pack()

        tk.Label(frame, text="Demo: user1 / validPass", font=("Helvetica", 8, "italic"), fg="#95a5a6").pack(pady=(12, 0))

    def handle_login(self):
        username = self.entry_username.get().strip()
        password = self.entry_password.get().strip()

        user = authenticate(username, password)
        if user:
            self.current_user = user
            messagebox.showinfo("Login Successful", f"Welcome, {username}!")
            self.show_main_dashboard()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def show_main_dashboard(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("520x520")

        frame = tk.Frame(self.root, padx=20, pady=15)
        frame.pack(expand=True, fill="both")

        header_frame = tk.Frame(frame)
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(header_frame, text="Dashboard", font=("Helvetica", 14, "bold"), fg="#2c3e50").pack(side="left")
        btn_logout = tk.Button(header_frame, text="Logout", command=self.show_login_frame, bg="#e74c3c", fg="white", font=("Helvetica", 9))
        btn_logout.pack(side="right")

        tk.Label(frame, text=f"Logged in as: {self.current_user[1]}", font=("Helvetica", 10, "italic"), fg="#7f8c8d").pack(anchor="w", pady=(0, 10))

        # Image preview box
        self.preview_frame = tk.LabelFrame(frame, text="Fruit Image Preview", padx=10, pady=10)
        self.preview_frame.pack(fill="both", expand=True, pady=5)

        self.label_image = tk.Label(self.preview_frame, text="No image selected", fg="#95a5a6", height=10)
        self.label_image.pack(expand=True)

        self.label_tag = tk.Label(frame, text="Status / Tag: None", font=("Helvetica", 11, "bold"), fg="#2980b9")
        self.label_tag.pack(pady=5)

        btn_upload = tk.Button(frame, text="Upload Fruit Image", command=self.upload_image, bg="#3498db", fg="white", font=("Helvetica", 10, "bold"), pady=5)
        btn_upload.pack(fill="x", pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            try:
                img = Image.open(file_path)
                img.thumbnail((220, 220))
                self.tk_img = ImageTk.PhotoImage(img)
                self.label_image.config(image=self.tk_img, text="")

                # Determine tag
                tag = "Ripe Fruit" if "ripe" in file_path.lower() else "Fruit Image Analyzed"
                self.label_tag.config(text=f"Detected Tag: {tag}", fg="#27ae60")

                # Save record to database
                save_image_record(self.current_user[0], file_path, tag)
                messagebox.showinfo("Success", "Image uploaded and logged to database successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")


def run_app():
    root = tk.Tk()
    app = FruitRipenessApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
