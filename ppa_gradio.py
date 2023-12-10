from pathlib import Path

import gradio as gr
import trafilatura

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

CORPUS_DIR_PATH = Path.cwd() / "ppa" / "corpus"
MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor(CORPUS_DIR_PATH)
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")


def classify(url):
    try:
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))
        td = processor.process_document(doc_text, url=url)
        label, score = classifier.predict([td], get_scores=True)
        result = {
            "classification": "good" if label == -1 else "bad",
            "score": round(score[0], 3),
            "threshold": round(classifier.threshold, 3),
        }
        return result
    except Exception as e:
        return {"error": str(e)}


iface = gr.Interface(
    fn=classify,
    inputs="text",
    outputs="text",
    title="Privacy Policy Binary Classifier",
    description="Enter a URL of a privacy policy to classify its contents as 'good' or 'bad'.",
)

# Deploy the Gradio interface
iface.launch(share=True)
