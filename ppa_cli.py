import typer

from app import classify_url

cli = typer.Typer()


@cli.command()
def classify_url_cli(url):
    class_result, score_result, membership_result, error_result = classify_url(url)
    response = f"Class: {class_result}\nScore: {score_result}\nMembership Score: {membership_result}\nError Message: {error_result}"
    typer.echo(response)


if __name__ == "__main__":
    cli()
