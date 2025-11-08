import typer

app = typer.Typer()


@app.command()
def bert(pretrained_model: str = "google-bert/bert-base-uncased"):
    from tuning.bert import main

    main(pretrained_model)


@app.command()
def gpt2(pretrained_model: str = "google-bert/bert-base-uncased", epochs: int = 5):
    from tuning.gpt2 import main

    main(pretrained_model, epochs)


if __name__ == "__main__":
    app()
