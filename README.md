GRU-Based Real-Time Web Threat Detection

This project implements a deep learning model using Gated Recurrent Units (GRUs) to detect malicious web requests in real-time. By treating request payloads as sequences of characters, the model learns to identify patterns indicative of common web attacks such as SQL Injection (SQLi), Cross-Site Scripting (XSS), and others.

The model is trained on a large, diverse dataset aggregated from four different public sources, providing comprehensive coverage of various threat vectors.

# Features

  * **Deep Learning Core**: Utilizes a GRU network, which is highly effective for sequential data analysis.
  * **Comprehensive Dataset**: Trained on a combined dataset of over 700,000 samples, including general web attacks, SQLi, XSS, and malicious URLs.
  * **High Performance**: Achieves **\~97% accuracy** on the test set, with high precision and recall.
  * **Real-Time Prediction**: Includes functionality to analyze new, unseen payloads instantly.
  * **Jupyter Notebook Environment**: The entire pipeline, from data preparation to model training and evaluation, is contained within a single, easy-to-run Jupyter Notebook (`model.ipynb`).

## 🧠 Model Architecture

The model is a Sequential Keras model designed for binary classification (benign vs. malicious).

1.  **Input Layer**: Accepts integer-encoded character sequences of a fixed length (`MAX_LEN = 250`).
2.  **Embedding Layer**: Maps each character's integer index to a dense vector representation (`EMBEDDING_DIM = 128`). This allows the model to learn relationships between characters.
3.  **SpatialDropout1D**: A regularization layer to prevent overfitting by dropping entire feature maps.
4.  **GRU Layer**: The core of the model, with 128 units, which processes the sequence of character embeddings to learn temporal patterns.
5.  **Dense Output Layer**: A single neuron with a **sigmoid** activation function that outputs a probability score between 0 (benign) and 1 (malicious).

## 📊 Dataset

The model's robustness comes from a master dataset created by combining, cleaning, and shuffling the following four sources:

  * **CSIC 2010**: Contains general HTTP requests, including a wide variety of web attacks.
  * **Kaggle SQL Injection Dataset**: A large collection of labeled SQL queries.
  * **Kaggle XSS Dataset**: A collection of payloads and scripts labeled as XSS or benign.
  * **Kaggle Malicious URLs Dataset**: A massive list of URLs classified as malicious or benign.

The final prepared dataset is saved to `prepared_dataset/master_web_attack_dataset.csv`.

## 📁 Project Structure

```
.
├── prepared_dataset/
│   └── master_web_attack_dataset.csv
├── saved_model_and_tokenizer/
│   ├── best_gru_model.keras
│   └── tokenizer.pickle
├── model.ipynb
├── csic_2010.csv
├── SQL_Injection_Dataset.csv
├── XSS_dataset.csv
├── malicious_urls.csv
└── README.md
```

## 🛠️ Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:

    ```txt
    tensorflow
    pandas
    scikit-learn
    seaborn
    matplotlib
    ipywidgets
    jupyter
    ```

    Then, install them:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets**
    Ensure the four source CSV files (`csic_2010.csv`, `SQL_Injection_Dataset.csv`, `XSS_dataset.csv`, `malicious_urls.csv`) are placed in the root directory of the project.

## ▶️ Usage

The entire project can be run from the `model.ipynb` notebook.

1.  **Launch Jupyter**
    ```bash
    jupyter notebook
    ```
2.  **Open and Run the Notebook**
    Open `model.ipynb` and run the cells sequentially from top to bottom. The notebook will:
      * Load and combine the four datasets into a master file.
      * Preprocess the data (tokenize and pad).
      * Build, train, and evaluate the GRU model.
      * Save the final model and tokenizer to the `saved_model_and_tokenizer/` directory.
      * Provide an interactive dashboard for real-time predictions at the end.

## 📈 Results

The model achieved the following performance on the unseen test set (20% of the total data):

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 97.39%  |
| Precision | 98.09%  |
| Recall    | 94.14%  |
| F1-Score  | 96.00%  |

The confusion matrix confirmed strong performance, although some misclassifications were noted, providing clear areas for future improvement.

## 💡 Future Work

  * **Expand Dataset**: Incorporate additional attack types like Command Injection and Path Traversal.
  * **Model Comparison**: Implement and compare the GRU model against other architectures like LSTMs, Bidirectional RNNs, and 1D-CNNs.
  * **Hyperparameter Tuning**: Use techniques like KerasTuner or Grid Search to find the optimal model hyperparameters.
  * **Full Deployment**: Deploy the model as a standalone REST API using Flask/FastAPI and build a separate web dashboard with Streamlit or React.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
