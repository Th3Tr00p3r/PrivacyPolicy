from pathlib import Path

import trafilatura
from flask import Flask, jsonify, request

from ppa.estimators import D2VClassifier
from ppa.processing import CorpusProcessor

CORPUS_DIR_PATH = Path.cwd() / "ppa" / "corpus"
MODEL_DIR_PATH = Path.cwd() / "ppa" / "models"

processor = CorpusProcessor(CORPUS_DIR_PATH)
classifier = D2VClassifier.load_model(MODEL_DIR_PATH / "pp_d2v.model")

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is missing"})

    try:
        # get document text from URL using trafilatura
        doc_text = trafilatura.extract(trafilatura.fetch_url(url))

        # process the document using the "trained" CorpusProcessor
        td = processor.process_document(doc_text, url=url)

        if processor.membership_test(td.words):
            # classify using D2VClassifier
            label, score = classifier.predict([td], get_scores=True)
            result = {
                "classification": "good" if label == -1 else "bad",
                "score": round(score[0], 2),
                "threshold": round(classifier.threshold, 2),
            }
        else:
            # Judged not to be a privac policy
            result = f"'{td.tags[0]}': Not a privacy policy!"

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)})


if __name__ == "__main__":
    app.run()
