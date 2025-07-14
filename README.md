# 🎬 BERT + BiLSTM Sentiment Classifier: Unlocking Text Emotions

Dive into the world of sentiment analysis with this **lightweight yet powerful classifier**, designed for understanding the emotional tone of text. This project combines the cutting-edge contextual understanding of **BERT** with the sequential modeling prowess of **Bidirectional LSTMs (BiLSTM)**, all wrapped in an interactive interface powered by **Gradio**.

---

## ✨ Features: Why This Model Shines

This sentiment classifier is built for efficiency and clarity, offering key advantages:

* **Hybrid Powerhouse:** Leverages **BERT** (specifically `bert-base-uncased`) as a robust contextual embedding extractor, feeding into a **BiLSTM** layer for capturing long-range dependencies and sequential patterns in text. This hybrid approach allows for nuanced sentiment detection.
* **Custom PyTorch Implementation:** Built with a **custom PyTorch model** rather than relying on the Hugging Face Trainer, providing full control and transparency over the training and model architecture.
* **Real-time Interaction:** Features a **Gradio-powered web interface** for instant, real-time sentiment predictions directly from your browser. Input your text and see the results immediately!
* **Clean & Deployable:** The codebase is **clean, well-structured, and ready for easy deployment** on platforms like Hugging Face Spaces or for quick demonstrations on GitHub.

---

## 🚀 Run Locally: Get Your Sentiment Classifier Up and Running

Follow these steps to set up and run the sentiment classifier on your local machine.

1.  **Clone the Repository:**
    Start by getting a copy of the project files to your local system.
    ```bash
    git clone [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name) # Replace with your actual repo URL
    cd your-repo-name # Replace with your actual repo name
    ```

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies. This isolates your project's libraries from others on your system, preventing conflicts.
    ```bash
    python3 -m venv env
    source env/bin/activate  # On macOS/Linux
    # On Windows: .\env\Scripts\activate
    ```

3.  **Install Dependencies:**
    Once your virtual environment is active, install all necessary Python libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Gradio Application:**
    Finally, launch the interactive Gradio interface.
    ```bash
    python app.py
    ```
    The application will start, and you'll typically find the web interface accessible in your browser at `http://127.0.0.1:7860/`.

---
