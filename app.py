from pathlib import Path
from typing import Tuple

import gradio as gr
import trafilatura
from gensim.corpora import Dictionary
from matplotlib.figure import Figure

from ppa.display import display_wordcloud
from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor()
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")


def classify_url(
    url: str, member_thresh: float = 0.9, plot: bool = True, verbose=False
) -> Tuple[str, str, str, str, Figure]:
    """Doc."""

    try:
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))
        if doc_text is None:
            raise RuntimeError("Unable to fetch URL. Try pasting in the policy text.")
        td = processor.process_document(doc_text, url=url)
        membership_score = processor.membership_test(td.words, verbose=verbose)
        membership_result = f"{membership_score:.3f} ({member_thresh:.3f})"
        if membership_score > member_thresh:
            label, scores = classifier.predict([td], get_scores=True)
            class_result = "Good" if label == -1 else "Bad"
            score_result = f"{scores[0]:.3f} ({classifier.threshold:.3f})"
            wordcloud_fig = (
                display_wordcloud(Dictionary([td.words]), should_plot=False) if plot else None
            )
            return class_result, score_result, membership_result, None, wordcloud_fig
        else:
            error_result = f"'{td.tags[0]}' - not a privacy policy!"
            return "N/A", None, membership_result, error_result, None
    except Exception as exc:
        return None, None, None, str(exc), None


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
            gr.Plot(label="Word-Cloud"),
        ],
        title="Privacy Policy Binary Classifier",
        description="Enter privacy-policy URL to classify its contents as 'good' or 'bad'.",
    )
    # Launch locally
    iface.launch()
