import logging

import typer

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def command(model: str = "jhu-clsp/mmBERT-small"):
    logger.info(model)


if __name__ == "__main__":
    app()
