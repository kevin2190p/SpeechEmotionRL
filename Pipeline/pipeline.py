class SpeechEmotionRecognitionPipeline:
    def __init__(self, dataset_path, model_paths, output_dir):
        self.dataset_path = dataset_path
        self.model_paths = model_paths
        self.output_dir = output_dir

        # Step 2 & 3: EmotionEnv and feature extractor
        self.env = EmotionEnv()
        self.extractor = Wav2Vec2FeatureExtractor(dataset_path=dataset_path)

        # Step 5: Validation and Test Evaluation
        self.model_validator = ModelValidation()
        self.test_set_evaluator = TestSetEvaluator()

        # Step 7: SHAP Analyzer
        self.best_agent_analyzer = BestAgentSHAPAnalyzer(
            best_tuned_models=self.model_paths,
            test_results=None,   # You should load actual test_results before use
            test_env=self.env,
            best_models=None,    # Load actual best models if needed
            X_test=None          # Should be assigned actual test features
        )

        # Step 8: Ensemble agent comparison
        self.ensemble_agent = EnsembleAgent(self.model_paths, self.env)

    def step_1_extract_features(self):
        print("Step 1: Extracting and saving features using Wav2Vec2...")
        self.extractor.extract_and_save_features()

    def step_2_initialize_env(self):
        print("Step 2: Initializing EmotionEnv and verifying input...")
        obs, _ = self.env.reset()
        print("Initial observation shape:", obs.shape)

    def step_3_train_all_models(self):
        print("Step 3: Training PPO, A2C, DQN, and QRDQN agents...")
        X = self.env.X
        y = self.env.y
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

        for trainer_class in [PPOTrainer, A2CTrainer, DQNTrainer, QRDQNTrainer]:
            trainer = trainer_class()
            trainer.run(X_train, y_train, X_val, y_val)

        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def step_4_hyperparameter_comparison(self):
        print("Step 4: Running hyperparameter comparison...")
        comparison = HyperparameterComparison(total_timesteps=10000, num_eval_episodes=10, eval_freq=1000)
        comparison.run_comparison()

    def step_5_validate_models(self):
        print("Step 5: Validating models and visualizing results...")
        self.model_validator.X_val = self.X_val
        self.model_validator.y_val = self.y_val
        self.model_validator.best_tuned_models = self.model_paths

        results = self.model_validator.validate_models()
        self.model_validator.visualize_validation_results(results)

    def step_6_test_set_evaluation(self):
        print("Step 6: Evaluating models on the test set...")
        self.test_set_evaluator.run_test_evaluation()

    def step_7_shap_analysis(self):
        print("Step 7: Running SHAP analysis for the best models...")
        self.best_agent_analyzer.X_test = self.X_test
        self.best_agent_analyzer.save_best_agent_and_params()
        self.best_agent_analyzer.run_shap_analysis_for_best_models()

    def step_8_ensemble_comparison(self):
        print("Step 8: Comparing individual models with ensemble agent...")
        individual_results = {}  # Fill this with actual results if available
        self.ensemble_agent.compare_results(individual_results)

    def step_9_run_gui(self):
        print("Step 9: Launching Speech Emotion Recognition GUI...")
        ser = SpeechEmotionRecognition()
        flask_thread = threading.Thread(target=ser.run_app)
        flask_thread.daemon = True
        flask_thread.start()
        print("Flask app running at http://127.0.0.1:5000")
        flask_thread.join()

    def run(self):
        self.step_1_extract_features()
        self.step_2_initialize_env()
        self.step_3_train_all_models()
        self.step_4_hyperparameter_comparison()
        self.step_5_validate_models()
        self.step_6_test_set_evaluation()
        self.step_7_shap_analysis()
        self.step_8_ensemble_comparison()
        self.step_9_run_gui()
