def main():
    dataset_path = r"C:\Users\TARUMT\Downloads\archive"  # Replace with actual path
    model_paths = {
        'PPO': 'path_to_ppo_model',
        'A2C': 'path_to_a2c_model',
        'DQN': 'path_to_dqn_model',
        'QRDQN': 'path_to_qrdqn_model',
    }
    output_dir = './output'

    pipeline = SpeechEmotionRecognitionPipeline(
        dataset_path=dataset_path,
        model_paths=model_paths,
        output_dir=output_dir
    )
    pipeline.run()

if __name__ == "__main__":
    main()
