import tkinter as tk
from tkinter import messagebox
import sqlite3

def login():
    username = entry_username.get()
    password = entry_password.get()

    conn = sqlite3.connect('fruit_app.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()

    if user:
        messagebox.showinfo("Login Successful", "Welcome, " + username)
        open_fruit_app(user[0])  # Open the main app with user ID
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

    conn.close()

def open_fruit_app(user_id):
    login_window.destroy()
    # Open the main application window
    fruit_app_window(user_id)

# Create the login window
login_window = tk.Tk()
login_window.title("Login")

label_username = tk.Label(login_window, text="Username")
label_username.pack()
entry_username = tk.Entry(login_window)
entry_username.pack()

label_password = tk.Label(login_window, text="Password")
label_password.pack()
entry_password = tk.Entry(login_window, show="*")
entry_password.pack()

button_login = tk.Button(login_window, text="Login", command=login)
button_login.pack()

login_window.mainloop()
