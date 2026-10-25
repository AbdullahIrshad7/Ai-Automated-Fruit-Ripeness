from tkinter import filedialog
from PIL import Image, ImageTk

def fruit_app_window(user_id):
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            # Display the image
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(image)
            label_image.config(image=img_tk)
            label_image.image = img_tk

            # Add dummy tags
            tags = "Ripe"  # Replace with your logic for determining tags
            label_tags.config(text=f"Tags: {tags}")

            # Save image metadata to the database
            conn = sqlite3.connect('fruit_app.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO images (user_id, image_path, tags) VALUES (?, ?, ?)', (user_id, file_path, tags))
            conn.commit()
            conn.close()

    # Create the main application window
    app_window = tk.Tk()
    app_window.title("Fruit App")

    label_image = tk.Label(app_window)
    label_image.pack()

    label_tags = tk.Label(app_window, text="Tags: ")
    label_tags.pack()

    button_upload = tk.Button(app_window, text="Upload Image", command=upload_image)
    button_upload.pack()

    app_window.mainloop()