import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    From [MultiNERD](https://github.com/Babelscape/multinerd/tree/master?tab=readme-ov-file#data)

    | Dataset Version                                                                                         | Sentences | Tokens | PER   | ORG   | LOC    | ANIM  | BIO  | CEL  | DIS   | EVE  | FOOD  | INST | MEDIA | MYTH | PLANT | TIME  | VEHI | OTHER |
    | :------------------------------------------------------------------------------------------------------ | --------: | -----: | ----: | ----: | -----: | ----: | ---: | ---: | ----: | ---: | ----: | ---: | ----: | ---: | ----: | ----: | ---: | ----: |
    | [MultiNERD EN](https://drive.google.com/drive/folders/1MvEsk6eiayAnWzAejcNYrEHXVVtbKUH1?usp=share_link) | 164.1K    | 3.6M   | 75.8K | 33.7K | 78.5K  | 15.5K | 0.2K | 2.8K | 11.2K | 3.2K | 11.0K | 0.4K | 7.5K  | 0.7K | 9.5K  | 3.2K  | 0.5K | 3.1M  |
    | [MultiNERD ES](https://drive.google.com/drive/folders/1uHREA8Acq65nIbSZB_5iyExwhijoc6R1?usp=share_link) | 173.2K    | 4.3M   | 70.9K | 20.6K | 90.2K  | 10.5K | 0.3K | 2.4K | 8.6K  | 6.8K | 7.8K  | 0.6K | 8.0K  | 1.6K | 7.6K  | 45.3K | 0.3K | 3.8M  |
    | [MultiNERD NL](https://drive.google.com/drive/folders/1THo0jsrUfGE9V560Sg-ax9FPHZ7FlSZ9?usp=share_link) | 171.7K    | 3.0M   | 56.9K | 21.4K | 78.7K  | 34.4K | 0.1K | 2.1K | 6.1K  | 4.7K | 5.6K  | 0.2K | 3.8K  | 1.3K | 6.3K  | 31.0K | 0.4K | 2.7M  |
    | [MultiNERD DE](https://drive.google.com/drive/folders/1giPAe4f_hl5RDToJYwffvF791C9_drr4?usp=share_link) | 156.8K    | 2.7M   | 79.2K | 31.2K | 72.8K  | 11.5K | 0.1K | 1.4K | 5.2K  | 4.0K | 3.6K  | 0.1K | 2.8K  | 0.8K | 7.8K  | 3.3K  | 0.5K | 2.4M  |
    | [MultiNERD RU](https://drive.google.com/drive/folders/1V82ZRAFUBaSM2oBbUOJ8_TtIsEGog-cv?usp=share_link) | 129.0K    | 2.3M   | 43.4K | 21.5K | 75.2K  | 7.3K  | 0.1K | 1.2K | 1.9K  | 2.8K | 3.2K  | 1.1K | 11.3K | 0.6K | 4.8K  | 22.8K | 0.5K | 2.0M  |
    | [MultiNERD IT](https://drive.google.com/drive/folders/1KEzz4dCTl1jqCd-gU9MMXGpXpoqPQMoe?usp=share_link) | 181.9K    | 4.7M   | 75.3K | 19.3K | 98.5K  | 8.8K  | 0.1K | 5.2K | 6.5K  | 5.8K | 5.8K  | 0.8K | 8.6K  | 1.8K | 5.1K  | 71.2K | 0.6K | 4.2M  |
    | [MultiNERD FR](https://drive.google.com/drive/folders/1Yqgm-BKB5vlO8TZGYulunRaoe0xvj6Cs?usp=share_link) | 176.2K    | 4.3M   | 89.6K | 28.2K | 90.9K  | 11.4K | 0.1K | 2.3K | 3.1K  | 7.4K | 3.2K  | 0.7K | 8.0K  | 2.0K | 4.4K  | 27.4K | 0.6K | 3.8M  |
    | [MultiNERD PL](https://drive.google.com/drive/folders/1rVIXP5qx2vOoFKUgf4XMRYTBFy2nCLLk?usp=share_link) | 195.0K    | 3.0M   | 66.5K | 29.2K | 100.0K | 19.7K | 0.1K | 3.3K | 6.5K  | 6.7K | 3.3K  | 0.6K | 4.9K  | 1.3K | 6.6K  | 44.1K | 0.7K | 2.5M  |
    | [MultiNERD PT](https://drive.google.com/drive/folders/1MKLlG4cMkBWhWcdhPMrY3g2TvRfyDgT3?usp=share_link) | 177.6K    | 3.9M   | 54.0K | 13.2K | 124.8K | 14.7K | 0.1K | 4.2K | 6.8K  | 5.9K | 5.4K  | 0.6K | 9.1K  | 1.6K | 9.2K  | 48.6K | 0.3K | 3.4M  |
    | [MultiNERD ZH](https://drive.google.com/drive/folders/1wI9ngtA_FK9H4vPWApEmHUZUAG0Tyb_Z?usp=share_link) | 195.3K    | 5.8M   | 68.3K | 20.8K | 49.6K  | 26.1K | 0.4K | 0.8K | 0.1K  | 5.1K | 1.9K  | 1.1K | 55.9K | 1.8K | 6.1K  | 0.4K  | 0.3K | 3.4M  |
    """)
    return


if __name__ == "__main__":
    app.run()
