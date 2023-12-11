import typer

from ppa_gradio import classify_url

app = typer.Typer()


@app.command()
def classify_url_cli(url):
    typer.echo(classify_url(url))


if __name__ == "__main__":
    app()
