from pathlib import Path

import trafilatura
import typer
from gensim.models.doc2vec import TaggedDocument

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor.load(MODEL_DIR_PATH / "corpus_processor.pkl")
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")

app = typer.Typer()


@app.command()
def classify_url(url: str):
    """
    Classifies the content fetched from the given URL using the loaded D2VClassifier model.

    Parameters:
    - url (str): The URL of the content to classify.

    Returns:
    - None

    This function fetches the document text from the provided URL using trafilatura, processes
    the document using a trained CorpusProcessor, and then classifies it using a loaded D2VClassifier model.
    The classification result (label) and score are printed to the terminal.
    """

    # get document text from URL using trafilatura
    doc_text = trafilatura.extract(trafilatura.fetch_url(url))

    # process the document using the "trained" CorpusProcessor
    words, tag = processor.process_document(doc_text, url=url)
    td = TaggedDocument(words, [tag])

    # classify using D2VClassifier
    label, score = classifier.predict([td], get_scores=True)
    typer.echo(
        f"'{td.tags[0]}' classified as '{('good' if label == -1 else 'bad')}' (score={score[0]:.2f}, threshold={classifier.threshold:.2f})."
    )


if __name__ == "__main__":
    app()
