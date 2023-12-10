import typer

from ppa_gradio import classify_url

app = typer.Typer()


@app.command()
def classify_url_cli(url, member_thresh: float = 0.93):
    typer.echo(classify_url(url, member_thresh))


if __name__ == "__main__":
    app()
