import tkinter as tk

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [''] * size
        self.current_index = 0

    def add(self, item):
        self.buffer[self.current_index] = item
        self.current_index = (self.current_index + 1) % self.size

    def get_display_string(self, item):
        display_list = self.buffer[:]
        return ' '.join(display_list)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.buffer = CircularBuffer(4)
        self.title("Circular Buffer GUI")
        self.configure(background="#f0f0f0")
        
        self.entry_label = tk.Label(self, text="Enter input:", bg="#f0f0f0", font=("Helvetica", 14))
        self.entry_label.pack(pady=5)
        
        self.entry = tk.Entry(self, font=("Helvetica", 14))
        self.entry.pack(pady=5)
        
        self.display_frame = tk.Frame(self, bg="#f0f0f0")
        self.display_frame.pack(pady=5)
        
        self.update_display()  # Display initial state
        
        self.entry.bind("<Return>", self.process_input)

    def process_input(self, event):
        user_input = self.entry.get().strip()
        self.entry.delete(0, tk.END)

        if not user_input:
            return

        word = user_input
        self.buffer.add(word)
        self.update_display()

    def update_display(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()  # Clear existing display

        for i, word in enumerate(self.buffer.buffer):
            label_text = word
            if i == self.buffer.current_index - 1:
                label_bg = "orange"
            else:
                label_bg = "#f0f0f0"
            label = tk.Label(self.display_frame, text=label_text, bg=label_bg, font=("Helvetica", 14), padx=10, pady=5)
            label.pack(side=tk.LEFT, padx=5, pady=5)

def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()
