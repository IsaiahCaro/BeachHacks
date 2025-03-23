import tkinter as tk
from tkinter import Label, Button, Entry, StringVar
import subprocess
import os
import sys

def end_app():
    app.destroy()
    sys.exit()

def start_collection():
    label = label_var.get().strip()
    if not label:
        status_label.config(text="❌ Please enter a label.")
        return

    subprocess.Popen(['python', 'user_collect_data.py', label])
    status_label.config(text=f"📸 Collecting data for '{label}'...")

def run_trained_model():
    subprocess.Popen(['python', 'test_method.py'])
    status_label.config(text="🧠 Running pretrained model...")

def train_user_model():
    subprocess.Popen(['python', 'user_train_classifier.py'])
    status_label.config(text="🔁 Training user model...")

def user_mode_test():
    subprocess.Popen(['python', 'user_test_method.py'])
    status_label.config(text="🧠 Testing user model...")

# Create GUI
app = tk.Tk()
app.title("ASL Recognition System")
app.geometry("520x520")
app.configure(bg="#f5f5f5")  # Light gray background

label_var = StringVar()

# === Styling Variables ===
BUTTON_BG = "#1a73e8"
BUTTON_FG = "white"
BUTTON_FONT = ("Helvetica", 13, "bold")
TEXT_COLOR = "#202124"

# === Title ===
Label(app, text="🤟 ASL Recognition System", font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg=TEXT_COLOR).pack(pady=(20, 10))
Label(app, text="Enter a label for your ASL gesture:", bg="#f5f5f5", fg=TEXT_COLOR, font=("Helvetica", 12)).pack()

# === Input Field ===
Entry(app, textvariable=label_var, font=("Helvetica", 14), width=25, bd=2, relief="solid").pack(pady=10)

# === Buttons ===
def make_button(text, command):
    return Button(
        app,
        text=text,
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground="#1558b0",
        activeforeground="white",
        relief="raised",
        bd=2,
        width=22,
        pady=8,
        command=command
    )

make_button("📸 Start Collecting", start_collection).pack(pady=6)
make_button("🔁 Train User Model", train_user_model).pack(pady=6)
make_button("🧠 Test User Model", user_mode_test).pack(pady=6)
make_button("🧠 Use Trained Model", run_trained_model).pack(pady=6)
make_button("❌ Exit", end_app).pack(pady=(20, 10))

# === Status Label ===
status_label = Label(app, text="", font=("Helvetica", 10), bg="#f5f5f5", fg=TEXT_COLOR)
status_label.pack(pady=10)

app.mainloop()