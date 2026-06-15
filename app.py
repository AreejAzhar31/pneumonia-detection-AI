import sys
import os

def run_preprocess():
    print("\n🔵 Running Module 1 — Data & Preprocessing...")
    from modules import preprocess
    preprocess.run_preprocess()

def run_train():
    print("\n🟠 Running Module 2 — Model Training (CNN)...")
    from modules import train
    # Call the training directly (your train.py runs everything when imported)
    train

def run_predict(image_path):
    print("\n🟢 Running Module 3 — Inference & Recommendation...")
    from modules import inference
    from modules import recommendations

    label, confidence, all_probs = inference.run_inference(image_path)
    report, _ = recommendations.get_recommendation(label, confidence)
    print(report)
    recommendations.save_report(report)

def show_help():
    print("""
╔══════════════════════════════════════════════╗
║   Pneumonia Detection AI — Usage Guide       ║
╠══════════════════════════════════════════════╣
║  python app.py preprocess                    ║
║    → Module 1: Prepare and sort images       ║
║                                              ║
║  python app.py train                         ║
║    → Module 2: Train CNN model               ║
║                                              ║
║  python app.py predict <path_to_image>       ║
║    → Module 3: Classify a chest X-ray        ║
║    Example:                                  ║
║    python app.py predict data/test/normal/sample.jpeg
║                                              ║
║  python app.py all                           ║
║    → Run full pipeline (Modules 1 + 2)       ║
╚══════════════════════════════════════════════╝
    """)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_help()
    elif sys.argv[1] == 'preprocess':
        run_preprocess()
    elif sys.argv[1] == 'train':
        run_train()
    elif sys.argv[1] == 'predict':
        if len(sys.argv) < 3:
            print("❌ Please provide an image path.")
            print("   Example: python app.py predict data/test/normal/sample.jpeg")
        else:
            run_predict(sys.argv[2])
    elif sys.argv[1] == 'all':
        run_preprocess()
        run_train()
    else:
        print(f"❌ Unknown command: {sys.argv[1]}")
        show_help()
