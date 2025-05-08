BERT Sentiment Analysis - Final Project
This repository contains a Jupyter notebook that demonstrates the application of BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis on textual data. It was developed as a final project to showcase how transformer-based models can outperform traditional machine learning algorithms in sentiment classification tasks.

📘 Notebook Overview
The notebook, BertSentimental.ipynb, walks through the complete pipeline of fine-tuning a pre-trained BERT model for sentiment analysis, covering:

Loading and preprocessing a labeled sentiment dataset

Tokenization using BERT's tokenizer

Creating a custom PyTorch dataset and data loaders

Fine-tuning the bert-base-uncased model

Evaluation on the test set

Comparison with a traditional ML baseline (e.g., SVM or Logistic Regression)

🔧 Requirements
To run the notebook, make sure the following Python packages are installed:

bash
Copy
Edit
transformers
torch
sklearn
pandas
numpy
matplotlib
seaborn
You can install them using:

bash
Copy
Edit
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
📁 Dataset
The dataset used for this project is a sentiment-labeled dataset (such as IMDb, SST-2, or another binary classification set). If the dataset is not included in the repository, please ensure it is downloaded and available in the correct path as referenced in the notebook.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/fernandoramos14/Final-Project.git
cd Final-Project
Open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook BertSentimental.ipynb
Run each cell in order to train and evaluate the model.

📊 Results
The notebook demonstrates that BERT-based sentiment analysis outperforms classical ML approaches, achieving high accuracy and robust generalization on unseen text samples.

📌 Key Takeaways
BERT significantly improves sentiment classification performance.

Tokenization and preprocessing tailored to BERT are critical.

Fine-tuning can be done efficiently with small labeled datasets.

📄 License
This project is open-source and available under the MIT License, unless otherwise specified.
