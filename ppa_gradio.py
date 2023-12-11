from pathlib import Path

import gradio as gr
import trafilatura

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor()
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")


# TODO: move this general function to a more generally named script, and import it here from here (as in ppa_cli/flask.py)
def classify_url(url: str, member_thresh: float = 0.6):
    try:
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))
        td, removed_ratio = processor.process_document(doc_text, url=url)
        membership_score = processor.membership_test(td.words, removed_ratio)
        if membership_score > member_thresh:
            label, scores = classifier.predict([td], get_scores=True)
            result = f"Class: {'good' if label == -1 else 'bad'}. Score: {scores[0]:.2f} (threshold={classifier.threshold:.2f}). Membership score: {membership_score:.2f} (threshold={member_thresh:.2f})."
        else:
            # Judged not to be a privac policy
            result = f"[Error] '{td.tags[0]}' - not a privacy policy! (membership score={membership_score:.2f})"
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":

    iface = gr.Interface(
        fn=classify_url,
        inputs="text",
        outputs="text",
        title="Privacy Policy Binary Classifier",
        description="Enter a URL of a privacy policy to classify its contents as 'good' or 'bad'.",
    )

    # Deploy the Gradio interface
    iface.launch(share=True)
