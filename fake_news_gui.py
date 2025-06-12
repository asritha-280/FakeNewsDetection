import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Prediction function
def detect_fake_news():
    news = input_text.get("1.0", tk.END).strip()
    if not news:
        messagebox.showwarning("⚠️ Input Error", "Please enter some news content.")
        return
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)[0]

    if prediction == "REAL":
        result_label.config(text="✅ This news is REAL", bg="#d4edda", fg="#155724")
    else:
        result_label.config(text="❌ This news is FAKE", bg="#f8d7da", fg="#721c24")

# GUI Setup
root = tk.Tk()
root.title("🧠 AI Fake News Detector")
root.geometry("800x550")
root.configure(bg="#f0f2f5")
root.resizable(False, False)

# Header Frame
header_frame = tk.Frame(root, bg="#343a40")
header_frame.pack(fill=tk.X)

tk.Label(header_frame, text="📰 FAKE NEWS DETECTOR", font=("Helvetica", 20, "bold"),
         bg="#343a40", fg="white", pady=10).pack()

# Main Frame
main_frame = tk.Frame(root, bg="#f0f2f5", padx=20, pady=20)
main_frame.pack(expand=True, fill=tk.BOTH)

# Instructions
tk.Label(main_frame, text="🔎 Paste news article content below:",
         font=("Arial", 14), bg="#f0f2f5", anchor="w").pack(anchor="w", pady=(0, 10))

# Scrollable Text Area
input_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Arial", 12),
                                       height=12, width=80, borderwidth=2, relief="solid")
input_text.pack(pady=10)

# Detect Button
tk.Button(main_frame, text="🧠 Detect Fake News", font=("Arial", 13, "bold"),
          bg="#007bff", fg="white", padx=20, pady=8,
          command=detect_fake_news).pack(pady=15)

# Result Label
result_label = tk.Label(main_frame, text="", font=("Arial", 16, "bold"),
                        bg="#f0f2f5", fg="black", pady=10)
result_label.pack()

# Footer
tk.Label(root, text="Made with ❤️ using Machine Learning", font=("Arial", 10),
         bg="#f0f2f5", fg="gray").pack(side=tk.BOTTOM, pady=10)

# Launch GUI
root.mainloop()
