# ===========
# Step 9: GUI
# ===========

import os
import tempfile
import traceback
import speech_recognition as sr
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import re
import base64
import warnings
import io
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import subprocess
import ffmpeg  # for audio conversion
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import random
from collections import Counter
from stable_baselines3 import PPO, A2C, DQN
try:
    from stable_baselines3.qrdqn import QRDQN  
except ImportError:
    try:
        from sb3_contrib import QRDQN  
    except ImportError:
        QRDQN = None  
import threading
from IPython.display import IFrame, display
import matplotlib.pyplot as plt
from io import BytesIO

class SpeechEmotionRecognition:
    def __init__(self):
        # Set the path to ffmpeg.exe and ffprobe.exe
        AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
        AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

        # Suppress warnings
        warnings.filterwarnings('ignore')

        # Initialize Flask app
        self.app = Flask(__name__)

        # Set FFmpeg paths explicitly
        self.ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
        self.ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"

        AudioSegment.converter = self.ffmpeg_path
        AudioSegment.ffprobe = self.ffprobe_path

        # Configure upload settings
        self.UPLOAD_FOLDER = 'audio_uploads'
        self.ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a'}

        # Ensure upload directory exists
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

        # Initialize agents and models
        self.agents = {}
        self._load_rl_agents()
        self._initialize_wav2vec2_model()
        
        # Define emotion mapping
        self.emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        self.emotion_map_reverse = {v: k for k, v in self.emotion_map.items()}

        # Define HTML template
        self.HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Speech Emotion Recognition</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    display: flex;
                    background: linear-gradient(135deg, #2c3e50, #000000);
                    color: #ecf0f1;
                    transition: background 0.5s ease;
                }
                @keyframes pulse {
                    0% { opacity: 0.8; }
                    50% { opacity: 1; }
                    100% { opacity: 0.8; }
                }
                #history-sidebar {
                    width: 280px;
                    border-right: 2px solid #34495e;
                    padding: 20px;
                    margin-right: 20px;
                    overflow-y: auto;
                    height: 100%;
                    position: fixed;
                    top: 0;
                    left: -280px;
                    background-color: #34495e;
                    box-shadow: 3px 0px 8px rgba(0, 0, 0, 0.4);
                    transition: left 0.3s ease-in-out;
                    z-index: 5;
                    padding-top: 20px;
                }
                #history-sidebar.open {
                    left: 0;
                }
                #history-sidebar:hover {
                    left: 0;
                }
                #main-content {
                    flex-grow: 1;
                    margin-top: 0;
                    padding-left: 20px;
                    transition: margin-left 0.3s ease-in-out;
                    margin-left: 20px;
                }
                #history-sidebar.open ~ #main-content {
                    margin-left: 300px;
                }
                h1 {
                    color: #e67e22;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                }
                #upload-container {
                    border: 3px dashed #e67e22;
                    padding: 30px;
                    text-align: center;
                    cursor: pointer;
                    background-color: #2c3e50;
                    transition: background-color 0.3s ease, border-color 0.3s ease;
                }
                #upload-container.drag-over {
                    background-color: #34495e;
                    border-color: #f39c12;
                }
                #preview-container {
                    margin-top: 30px;
                    display: none;
                    background-color: #34495e;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
                }
                #file-name {
                    font-weight: bold;
                    color: #f39c12;
                }
                #remove-file {
                    color: #e74c3c;
                    cursor: pointer;
                    margin-left: 15px;
                    font-size: 1.2em;
                    transition: color 0.3s ease;
                }
                #remove-file:hover {
                    color: #c0392b;
                }
                #upload-status {
                    margin-top: 15px;
                    font-style: italic;
                    color: #95a5a6;
                }
                #upload-status.error {
                    color: #e74c3c;
                }
                #upload-status.success {
                    color: #2ecc71;
                }
                #model-selection {
                    margin-top: 30px;
                    background-color: #34495e;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
                }
                #model-selection label {
                    display: block;
                    margin-bottom: 8px;
                    color: #f39c12;
                    font-weight: bold;
                }
                #model-select {
                    width: 220px;
                    padding: 10px;
                    border: 1px solid #f39c12;
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    border-radius: 5px;
                }
                #predict-button {
                    padding: 12px 25px;
                    margin-top: 15px;
                    cursor: pointer;
                    background: linear-gradient(to right, #2ecc71, #27ae60);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
                    transition: background 0.3s ease, transform 0.2s ease;
                }
                #predict-button:disabled {
                    background: #7f8c8d;
                    color: #ecf0f1;
                    cursor: not-allowed;
                }
                #predict-button:not(:disabled):hover {
                    background: linear-gradient(to right, #27ae60, #2ecc71);
                    transform: scale(1.05);
                }
                #prediction-results {
                    margin-top: 30px;
                    border: 1px solid #555;
                    padding: 20px;
                    display: none;
                    background-color: #2c3e50;
                    border-radius: 5;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
                }
                #prediction-results h2 {
                    color: #f39c12;
                    margin-top: 0;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
                }
                #prediction-results p {
                    margin-bottom: 10px;
                }
                #prediction-results span {
                    font-weight: bold;
                }
                #predicted-emotion {
                    color: #f39c12;
                }
                #correct-emotion {
                    color: #f39c12;
                }
                .empty-state {
                    text-align: center;
                    color: #777;
                    font-style: italic;
                }
                .correct-prediction {
                    color: #2ecc71;
                    font-weight: bold;
                    animation: pulse 1.5s infinite alternate;
                }
                .incorrect-prediction {
                    color: #e74c3c;
                    font-weight: bold;
                    animation: pulse 1.5s infinite alternate;
                }
                #history-title {
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 15px;
                    color: #f39c12;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
                }
                #history-list {
                    list-style-type: none;
                    padding: 0;
                }
                .history-item {
                    cursor: pointer;
                    margin-bottom: 8px;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    border-bottom: 1px solid #34495e;
                    transition: background-color 0.3s ease, transform 0.2s ease;
                    display: flex;
                    align-items: center;
                }
                .history-item:hover {
                    background-color: #34495e;
                    transform: scale(1.02);
                }
                .history-item.selected {
                    background-color: #e67e22;
                    color: white;
                    font-weight: bold;
                }
                .history-remove-button {
                    color: #e74c3c;
                    cursor: pointer;
                    margin-left: auto;
                    font-size: 1.0em;
                    transition: color 0.3s ease;
                    padding: 5px;
                    border: none;
                    background: none;
                }
                .history-remove-button:hover {
                    color: #c0392b;
                }
                .history-icon {
                    font-size: 1.2em;
                    margin-right: 8px;
                }
                #report-container {
                    margin-top: 30px;
                    background-color: #34495e;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
                }
                #report-title {
                    color: #f39c12;
                    text-align: center;
                    margin-bottom: 15px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
                }
                #bar-chart {
                    width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div id="history-sidebar">
                <h2 id="history-title">History</h2>
                <ul id="history-list">
                    {% if history %}
                        {% for item in history %}
                            <li class="history-item" data-history-id="{{ loop.index - 1 }}">
                                <span class="history-icon">📄</span> {{ item.filename }}
                                <button class="history-remove-button" data-history-index="{{ loop.index - 1 }}">✖</button>
                            </li>
                        {% endfor %}
                    {% else %}
                        <li class="empty-state">No history yet.</li>
                    {% endif %}
                </ul>
            </div>
            <div id="main-content">
                <h1>Speech Emotion Recognition</h1>
                <div id="upload-container">
                    <input type="file" id="file-input" style="display: none;" accept="audio/*">
                    <p style="color: #95a5a6;">Drag and drop your audio file here or <span style="color: #f39c12; font-weight: bold;">click to browse</span>.</p>
                </div>
                <div id="preview-container" style="display: none;">
                    <p>Selected file: <span id="file-name"></span> <span id="remove-file">✖</span></p>
                    <audio id="audio-preview" controls style="width: 300px;"></audio>
                </div>
                <div id="upload-status" class="upload-status"></div>
                <div id="model-selection">
                    <label for="model-select">Select Model:</label>
                    <select id="model-select">
                        {% if 'PPO' in models %} <option value="PPO">PPO</option> {% endif %}
                        {% if 'A2C' in models %} <option value="A2C">A2C</option> {% endif %}
                        {% if 'DQN' in models %} <option value="DQN">DQN</option> {% endif %}
                        {% if 'QRDQN' in models and 'QRDQN' in loaded_models %} <option value="QRDQN">QRDQN</option> {% endif %}
                        {% if 'ensemble' in models %} <option value="ensemble">Ensemble</option> {% endif %}
                    </select>
                    <button id="predict-button" disabled>Predict Emotion</button>
                </div>
                <div id="prediction-results">
                    <h2>Prediction Results</h2>
                    <p>Predicted Emotion: <span id="predicted-emotion"></span></p>
                    <p>Correct Emotion (from filename): <span id="correct-emotion"></span></p>
                    <p>Prediction Status: <span id="prediction-status"></span></p>
                </div>
                <div id="report-container" style="display: none;">
                    <h2 id="report-title">Model Performance Report</h2>
                    <img id="bar-chart" src="" alt="Model Performance Bar Chart">
                </div>
            </div>
            <script>
                const uploadBox = document.getElementById('upload-container');
                const fileInput = document.getElementById('file-input');
                const fileNameDisplay = document.getElementById('file-name');
                const removeFileButton = document.getElementById('remove-file');
                const previewContainer = document.getElementById('preview-container');
                const audioPreview = document.getElementById('audio-preview');
                const uploadStatus = document.getElementById('upload-status');
                const modelSelect = document.getElementById('model-select');
                const predictButton = document.getElementById('predict-button');
                const predictionResultsDiv = document.getElementById('prediction-results');
                const predictedEmotionSpan = document.getElementById('predicted-emotion');
                const correctEmotionSpan = document.getElementById('correct-emotion');
                const predictionStatusSpan = document.getElementById('prediction-status');
                const historySidebar = document.getElementById('history-sidebar');
                const historyList = document.getElementById('history-list');
                const mainContent = document.getElementById('main-content');
                const reportContainer = document.getElementById('report-container');
                const barChartImage = document.getElementById('bar-chart');
                let selectedFile = null;
                let predictionHistory = [];
                let selectedHistoryItem = null;
                function adjustMainContentMargin() {
                    if (historySidebar.classList.contains('open')) {
                        mainContent.style.marginLeft = '300px';
                    } else {
                        mainContent.style.marginLeft = '20px';
                    }
                }
                historySidebar.addEventListener('mouseenter', () => {
                    historySidebar.classList.add('open');
                    adjustMainContentMargin();
                });
                historySidebar.addEventListener('mouseleave', () => {
                    historySidebar.classList.remove('open');
                    adjustMainContentMargin();
                });
                function highlight(e) {
                    uploadBox.classList.add('drag-over');
                }
                function unhighlight(e) {
                    uploadBox.classList.remove('drag-over');
                }
                function handleDrop(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    unhighlight(e);
                    const dt = e.dataTransfer;
                    const file = dt.files[0];
                    handleFile(file);
                }
                uploadBox.addEventListener('dragenter', highlight, false);
                uploadBox.addEventListener('dragover', highlight, false);
                uploadBox.addEventListener('dragleave', unhighlight, false);
                uploadBox.addEventListener('drop', handleDrop, false);
                uploadBox.addEventListener('click', () => {
                    fileInput.click();
                });
                fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length) {
                        handleFile(e.target.files[0]);
                    }
                });
                function handleFile(file) {
                    if (!file || !file.type.startsWith('audio/')) {
                        uploadStatus.textContent = 'Invalid audio file type.';
                        uploadStatus.className = 'upload-status error';
                        previewContainer.style.display = 'none';
                        selectedFile = null;
                        predictButton.disabled = true;
                        predictionResultsDiv.style.display = 'none';
                        return;
                    }
                    selectedFile = file;
                    fileNameDisplay.textContent = file.name;
                    audioPreview.src = URL.createObjectURL(file);
                    previewContainer.style.display = 'block';
                    uploadStatus.textContent = '';
                    uploadStatus.className = 'upload-status';
                    predictButton.disabled = false;
                    predictionResultsDiv.style.display = 'none';
                    reportContainer.style.display = 'none';
                    if (selectedHistoryItem) {
                        selectedHistoryItem.classList.remove('selected');
                        selectedHistoryItem = null;
                    }
                }
                removeFileButton.addEventListener('click', () => {
                    fileInput.value ='';
                    selectedFile = null;
                    previewContainer.style.display = 'none';
                    uploadStatus.textContent = '';
                    uploadStatus.className = 'upload-status';
                    predictButton.disabled = true;
                    predictionResultsDiv.style.display = 'none';
                    reportContainer.style.display = 'none';
                    if (selectedHistoryItem) {
                        selectedHistoryItem.classList.remove('selected');
                        selectedHistoryItem = null;
                    }
                });
                predictButton.addEventListener('click', () => {
                    if (!selectedFile) {
                        uploadStatus.textContent = 'Please select an audio file first.';
                        uploadStatus.className = 'upload-status error';
                        return;
                    }
                    const selectedModel = modelSelect.value;
                    const formData = new FormData();
                    formData.append('audio_file', selectedFile);
                    formData.append('model_name', selectedModel);
                    fetch('/predict_emotion', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.predicted_emotion) {
                            predictedEmotionSpan.textContent = data.predicted_emotion || 'Unknown';
                            correctEmotionSpan.textContent = data.correct_emotion || 'Unknown';
                            predictionResultsDiv.style.display = 'block';
                            uploadStatus.textContent = 'Prediction successful.';
                            uploadStatus.className = 'upload-status success';
                            const isCorrect = data.predicted_emotion === data.correct_emotion;
                            predictionStatusSpan.textContent = isCorrect ? 'Correct!' : 'Incorrect.';
                            predictionStatusSpan.className = isCorrect ? 'correct-prediction' : 'incorrect-prediction';
                            const historyItem = {
                                filename: selectedFile.name,
                                predicted: data.predicted_emotion,
                                correct: data.correct_emotion,
                                model: selectedModel
                            };
                            predictionHistory.push(historyItem);
                            updateHistoryDisplay();
                            scrollToBottom('history-sidebar');
                            updateReportDisplay();
                        } else if (data.error) {
                            uploadStatus.textContent = `Prediction error: ${data.error}`;
                            uploadStatus.className = 'upload-status error';
                            predictionResultsDiv.style.display = 'none';
                            predictionStatusSpan.textContent = '';
                            predictionStatusSpan.className = '';
                            reportContainer.style.display = 'none';
                        } else {
                            uploadStatus.textContent = 'No prediction result received.';
                            uploadStatus.className = 'upload-status error';
                            predictionResultsDiv.style.display = 'none';
                            predictionStatusSpan.textContent = '';
                            predictionStatusSpan.className = '';
                            reportContainer.style.display = 'none';
                        }
                    })
                    .catch(error => {
                        console.error('Error during prediction:', error);
                        uploadStatus.textContent = 'An error occurred during prediction.';
                        uploadStatus.className = 'upload-status error';
                        predictionResultsDiv.style.display = 'none';
                        predictionStatusSpan.textContent = '';
                        predictionStatusSpan.className = '';
                        reportContainer.style.display = 'none';
                    });
                });
                function updateHistoryDisplay() {
                    historyList.innerHTML = '';
                    if (predictionHistory.length === 0) {
                        const li = document.createElement('li');
                        li.className = 'empty-state';
                        li.textContent = 'No history yet.';
                        historyList.appendChild(li);
                    } else {
                        predictionHistory.forEach((item, index) => {
                            const li = document.createElement('li');
                            li.className = 'history-item';
                            li.dataset.historyId = index;
                            li.innerHTML = `<span class="history-icon">📄</span> ${item.filename} <button class="history-remove-button" data-history-index="${index}">✖</button>`;
                            li.addEventListener('click', () => showHistoryItem(index));
                            const removeButton = li.querySelector('.history-remove-button');
                            removeButton.addEventListener('click', (event) => {
                                event.stopPropagation();
                                removeHistoryItem(index);
                                updateReportDisplay();
                            });
                            historyList.appendChild(li);
                        });
                    }
                }
                function removeHistoryItem(index) {
                    predictionHistory.splice(index, 1);
                    updateHistoryDisplay();
                    if (predictionHistory.length === 0) {
                        predictionResultsDiv.style.display = 'none';
                        reportContainer.style.display = 'none';
                    } else if (selectedHistoryItem && parseInt(selectedHistoryItem.dataset.historyId) === index) {
                        predictionResultsDiv.style.display = 'none';
                        selectedHistoryItem = null;
                    } else if (selectedHistoryItem && parseInt(selectedHistoryItem.dataset.historyId) > index) {
                        selectedHistoryItem.dataset.historyId = parseInt(selectedHistoryItem.dataset.historyId) - 1;
                    }
                }
                function showHistoryItem(index) {
                    const item = predictionHistory[index];
                    predictedEmotionSpan.textContent = item.predicted || 'Unknown';
                    correctEmotionSpan.textContent = item.correct || 'Unknown';
                    predictionResultsDiv.style.display = 'block';
                    const isCorrect = item.predicted === item.correct;
                    predictionStatusSpan.textContent = isCorrect ? 'Correct!' : 'Incorrect.';
                    predictionStatusSpan.className = isCorrect ? 'correct-prediction' : 'incorrect-prediction';
                    const allHistoryItems = document.querySelectorAll('#history-list .history-item');
                    allHistoryItems.forEach(el => el.classList.remove('selected'));
                    const selectedItem = document.querySelector(`#history-list .history-item[data-history-id="${index}"]`);
                    if (selectedItem) {
                        selectedItem.classList.add('selected');
                        selectedHistoryItem = selectedItem;
                    }
                }
                function scrollToBottom(id) {
                    const element = document.getElementById(id);
                    if (element) {
                        element.scrollTop = element.scrollHeight;
                    }
                }
                function updateReportDisplay() {
                    const modelPerformances = {};
                    const usedModels = new Set();
                    predictionHistory.forEach(item => {
                        usedModels.add(item.model);
                        if (!modelPerformances[item.model]) {
                            modelPerformances[item.model] = { correct: 0, total: 0 };
                        }
                        modelPerformances[item.model].total++;
                        if (item.predicted === item.correct) {
                            modelPerformances[item.model].correct++;
                        }
                    });
                    const models = Object.keys(modelPerformances);
                    if (models.length > 0) {
                        const modelNames = [];
                        const accuracyScores = [];
                        models.forEach(model => {
                            modelNames.push(model);
                            accuracyScores.push((modelPerformances[model].correct / modelPerformances[model].total) * 100);
                        });
                        fetch('/generate_report', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ models: modelNames, accuracies: accuracyScores })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.image_base64) {
                                barChartImage.src = `data:image/png;base64,${data.image_base64}`;
                                reportContainer.style.display = 'block';
                            } else if (data.error) {
                                console.error("Error generating report:", data.error);
                                reportContainer.style.display = 'none';
                            } else {
                                reportContainer.style.display = 'none';
                            }
                        })
                        .catch(error => {
                            console.error("Error sending data for report generation:", error);
                            reportContainer.style.display = 'none';
                        });
                    } else {
                        reportContainer.style.display = 'none';
                    }
                }
                updateHistoryDisplay();
                adjustMainContentMargin();
                updateReportDisplay();
            </script>
        </body>
        </html>
        """

    def _load_rl_agents(self):
        best_tuned_models = {
            'PPO': ('./tuned_random_search_models\\best_tuned_model_PPO.zip', {'learning_rate': 0.0005058412416040213, 'gamma': 0.9936511387686254, 'n_steps': 2048, 'ent_coef': 0.01021001741802571, 'clip_range': 0.20799800196716112}),
            'A2C': ('./tuned_random_search_models\\best_tuned_model_A2C.zip', {'learning_rate': 0.0009701995036997713, 'gamma': 0.968258666711713, 'n_steps': 32, 'ent_coef': 0.03805261748923566}),
            'DQN': ('./tuned_random_search_models\\best_tuned_model_DQN.zip', {'learning_rate': 0.00022589129432441278, 'gamma': 0.9437490177792842, 'batch_size': 128, 'buffer_size': 50000, 'exploration_fraction': 0.07620433443270401, 'exploration_final_eps': 0.026506568962077356}),
            'QRDQN': ('./tuned_random_search_models\\best_tuned_model_QRDQN.zip', {'learning_rate': 5.104820568068648e-05, 'gamma': 0.9116347326147005, 'batch_size': 32, 'buffer_size': 50000, 'exploration_fraction': 0.2883581944429394, 'exploration_final_eps': 0.05741793563626612})
        }
        for algo_name, (path, _) in best_tuned_models.items():
            try:
                if not os.path.exists(path):
                    print(f"Warning: Model not found at {path}")
                    continue
                if algo_name.lower() == 'ppo':
                    self.agents[algo_name] = PPO.load(path)
                elif algo_name.lower() == 'a2c':
                    self.agents[algo_name] = A2C.load(path)
                elif algo_name.lower() == 'dqn':
                    self.agents[algo_name] = DQN.load(path)
                elif algo_name.lower() == 'qrdqn' and QRDQN is not None:
                    self.agents[algo_name] = QRDQN.load(path)
                elif algo_name.lower() == 'qrdqn' and QRDQN is None:
                    print(f"Warning: QRDQN not available, skipping loading from {path}")
                print()
            except Exception as e:
                print(f"Error loading {algo_name} model from {path}: {e}")

        # Define EnsembleAgent
        class EnsembleAgent:
            def __init__(self, loaded_agents):
                self.loaded_agents = [agent for agent in loaded_agents if agent is not None]
            
            def predict(self, state):
                if not self.loaded_agents:
                    return None, None
                action = self.ensemble_predict(state, self.loaded_agents)
                return action, None
        
        # Create ensemble agent
        if self.agents:
            ensemble_models = list(self.agents.values())
            self.agents['ensemble'] = EnsembleAgent(ensemble_models)
            print()
        else:
            print("Warning: No RL agents loaded. Ensemble agent not created.")

    def ensemble_predict(self, state, loaded_agents):
        actions = []
        valid_agents = [agent for agent in loaded_agents if agent is not None]
        if not valid_agents:
            return None
        for agent in valid_agents:
            action, _ = agent.predict(state)
            actions.append(int(action.item() if isinstance(action, np.ndarray) else action))
        counts = Counter(actions)
        max_count = max(counts.values())
        candidate_actions = [a for a, count in counts.items() if count == max_count]
        return random.choice(candidate_actions)

    def _initialize_wav2vec2_model(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model.eval()
        if torch.cuda.is_available():
            self.wav2vec2_model.to("cuda")
            print("Wav2Vec2 model moved to CUDA.")
        else:
            print(" ")

    def extract_wav2vec2_features(self, audio_path):
        try:
            waveform, sr = librosa.load(audio_path, sr=16000)
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
            hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return hidden_states.reshape(1, -1)
        except Exception as e:
            print(f"Error extracting Wav2Vec2 features: {e}")
            return None

    def get_correct_emotion(self, filename):
        try:
            file_parts = filename.split('-')
            if len(file_parts) >= 3 and file_parts[0] == '03' and file_parts[1] == '01':
                emotion_code = file_parts[2]
                return self.emotion_map.get(emotion_code, "unknown")
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def _setup_flask_routes(self):
        @self.app.route('/', methods=['GET'])
        def index():
            return render_template_string(self.HTML_TEMPLATE, models=self.agents.keys(), loaded_models=self.agents.keys(), history=[])

        @self.app.route('/predict_emotion', methods=['POST'])
        def predict_emotion():
            if 'audio_file' not in request.files:
                return jsonify({'error': 'No audio file part in the request'}), 400
            file = request.files['audio_file']
            model_name = request.form.get('model_name')

            if file.filename == '':
                return jsonify({'error': 'No selected audio file'}), 400

            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_audio_path = os.path.join(self.UPLOAD_FOLDER, filename)
                file.save(temp_audio_path)

                try:
                    features = self.extract_wav2vec2_features(temp_audio_path)
                    if features is None:
                        return jsonify({'error': 'Error extracting audio features'}), 500

                    predicted_emotion = "unknown"
                    if model_name and model_name in self.agents:
                        agent = self.agents[model_name]
                        action, _ = agent.predict(features)
                        predicted_emotion_index = int(action)
                        predicted_emotion = list(self.emotion_map.values())[predicted_emotion_index]
                    else:
                        return jsonify({'error': f'Model "{model_name}" not found'}), 400

                    correct_emotion = self.get_correct_emotion(filename)

                    return jsonify({'predicted_emotion': predicted_emotion, 'correct_emotion': correct_emotion})
                except Exception as e:
                    traceback.print_exc()
                    return jsonify({'error': str(e)}), 500
                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            else:
                return jsonify({'error': 'Invalid file type'}), 400

        @self.app.route('/generate_report', methods=['POST'])
        def generate_report():
            data = request.get_json()
            models = data.get('models', [])
            accuracies = data.get('accuracies', [])

            if not models or not accuracies or len(models) != len(accuracies):
                return jsonify({'error': 'Invalid data for report generation'}), 400

            try:
                plt.figure(figsize=(10, 6))
                plt.bar(models, accuracies, color='#f39c12')
                plt.xlabel('Model')
                plt.ylabel('Accuracy (%)')
                plt.title('Model Performance Comparison')
                plt.ylim(0, 100)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                img_buf = BytesIO()
                plt.savefig(img_buf, format='png')
                img_buf.seek(0)

                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

                plt.close()

                return jsonify({'image_base64': img_base64})

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def run_app(self):
        self._setup_flask_routes()
        self.app.run(debug=False, use_reloader=False)
