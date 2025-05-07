Speech Emotion Recognition Pipeline
This is an open-source project for Speech Emotion Recognition using reinforcement learning with custom emotion environment and audio feature extraction. It includes training, validation, model explainability, and a simple web interface for testing.


How to Use:
1. Run the download_dataset.ipynb or download the dataset from the official website:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

2. Install Dependencies:
Run the following script to install the required libraries:
python install_dependencies.py

3. Update the necessary path name 

4. Run the Pipeline:
Execute the full emotion recognition pipeline:
python main.py

5. Access the Web Interface:
After the pipeline completes, a web app will launch at:
http://127.0.0.1:5000


Requirements:
Python 3.8 or higher
Internet connection (for Wav2Vec2 feature extraction)
Basic GPU support recommended (optional)


License:
This project is open-source and available under the The Unlicense. (For more information, please read the 'LICENSE' file.)
