# Incremental Learning–Based Anomaly Detection of Communication Key Misuse in SDN Environments

## 📌 Project Overview

This project implements an Incremental Learning–Based Anomaly Detection system to detect communication key misuse in Software Defined Networking (SDN) environments.

The system:
- Uses real intrusion dataset (CIC-IDS-2017)
- Converts it into SDN-style key usage logs
- Performs preprocessing and feature engineering
- Trains a base anomaly detection model
- Updates the model using incremental learning (online learning)
- Generates performance metrics and visual proofs

---

## 🧠 Core Concepts

- Software Defined Networking (SDN)
- Communication Key Monitoring
- Anomaly Detection
- Incremental / Online Learning
- Real-world intrusion dataset (CIC-IDS-2017)
- Batch streaming simulation

---

## 📂 Project Structure

AI Project/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── external/      → Place CIC-IDS-2017 CSV files here
│   ├── raw/           → Adapted SDN-format data
│   └── processed/     → Preprocessed data
│
├── experiments/
│   └── results/
│       ├── figures/
│       ├── logs/
│       ├── base_model.pkl
│       ├── detected_anomalies.csv
│       ├── stream_performance.csv
│       └── evaluation_summary.txt
│
├── src/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── models/
│   ├── visualization/
│   ├── evaluation/
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md

---

## 📊 Dataset

Dataset Used: CIC-IDS-2017

Place CSV files inside:

data/external/

Example:
data/external/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

---

## ⚙️ Installation (Windows - PowerShell)

1️⃣ Create virtual environment

python -m venv venv
.\venv\Scripts\activate

2️⃣ Install dependencies

pip install -r requirements.txt

---

## 🚀 Run the Full Pipeline

From project root:

python main.py

The pipeline automatically performs:

1. Dataset adaptation → SDN format
2. Preprocessing & scaling
3. Base model training
4. Incremental learning (batch streaming)
5. Visualization generation
6. Evaluation report creation

---

## 🔄 Incremental Learning Process

- 25% of data used for initial training.
- Remaining data streamed in batches.
- Each batch:
  - Predicts anomalies
  - Calculates Accuracy & F1-score
  - Updates model using partial_fit()

Model Used:
SGDClassifier (log_loss) – supports incremental updates.

---

## 📈 Output Files

After execution:

experiments/results/

- base_model.pkl
- detected_anomalies.csv
- stream_performance.csv
- evaluation_summary.txt
- logs/system.log

---

## 📊 Generated Visualizations (4)

Stored in:

experiments/results/figures/

1. anomaly_counts.png  
2. incremental_performance.png  
3. true_vs_pred.png  
4. pca_scatter.png  

These graphs provide visual proof of model performance.

---

## 📑 Evaluation Metrics

Stored in:

experiments/results/evaluation_summary.txt

Includes:
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🔧 Changing Dataset

Open:

configs/config.yaml

Modify:

external_file: "data/external/Your_File.csv"

Then run:

python main.py

No other code changes required.

---

## 🛠 Technologies Used

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Loguru
- YAML

---

## 🎯 Key Features

✔ Real dataset adaptation to SDN scenario  
✔ Incremental learning implementation  
✔ Batch streaming simulation  
✔ Automatic pipeline execution  
✔ Performance visualization  
✔ Configurable dataset support  

---

## 👨‍💻 Project Title

Incremental Learning–Based Anomaly Detection of Communication Key Misuse in SDN Environments
