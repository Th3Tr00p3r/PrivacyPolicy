from pathlib import Path

import gradio as gr
import trafilatura

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor()
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")


def classify_url(url: str, member_thresh: float = 0.6):
    try:
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))
        td, removed_ratio = processor.process_document(doc_text, url=url)
        membership_score = processor.membership_test(td.words, removed_ratio)
        membership_result = f"{membership_score:.2f} ({member_thresh:.2f})"
        if membership_score > member_thresh:
            label, scores = classifier.predict([td], get_scores=True)
            class_result = "Good" if label == -1 else "Bad"
            score_result = f"{scores[0]:.2f} ({classifier.threshold:.2f})"
            return class_result, score_result, membership_result, None
        else:
            error_result = f"'{td.tags[0]}' - not a privacy policy!"
            return "N/A", None, membership_result, error_result
    except Exception as e:
        return None, None, f"Error: {str(e)}"


if __name__ == "__main__":
    # Define the GUI
    iface = gr.Interface(
        fn=classify_url,
        inputs=[gr.Textbox(label="URL")],
        outputs=[
            gr.Textbox(label="Class"),
            gr.Textbox(label="Decision Score (threshold)"),
            gr.Textbox(label="Membership Score (threshold)"),
            gr.Textbox(label="Error Message"),
        ],
        title="Privacy Policy Binary Classifier",
        description="Enter privacy-policy URL to classify its contents as 'good' or 'bad'.",
    )
    # Launch locally and get a shareable link
    iface.launch(share=True)
