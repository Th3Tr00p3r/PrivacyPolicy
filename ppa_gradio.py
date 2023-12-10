from pathlib import Path

import gradio as gr
import trafilatura

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor()
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")


# TODO: move this general function to a more generally named script, and import it here from here (as in ppa_cli/flask.py)
def classify_url(url: str, member_thresh: float = 0.93):
    try:
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))
        td = processor.process_document(doc_text, url=url)
        membership_result = processor.membership_test(td.words)
        if membership_result > member_thresh:
            label, score = classifier.predict([td], get_scores=True)
            result = {
                "classification": "good" if label == -1 else "bad",
                "score": round(score[0], 3),
                "threshold": round(classifier.threshold, 3),
                "membership_result": membership_result,
            }
        else:
            # Judged not to be a privac policy
            result = {
                "error": f"'{td.tags[0]}' - Not a privacy policy! (membership={membership_result:.2f})"
            }
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
