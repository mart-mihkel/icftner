import marimo

__generated_with = "0.19.4"
app = marimo.App()

with app.setup:
    import marimo as mo
    import polars as pl


@app.cell
def _():
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


@app.cell
def _():
    _data = {
        "lang": ["EN", "ES", "NL", "DE", "RU", "IT", "FR", "PL", "PT", "ZH"],
        "sentences": [
            164.1,
            173.2,
            171.7,
            156.8,
            129.0,
            181.9,
            176.2,
            195.0,
            177.6,
            195.3,
        ],
        "tokens": [3.6, 4.3, 3.0, 2.7, 2.3, 4.7, 4.3, 3.0, 3.9, 5.8],
        "PER": [75.8, 70.9, 56.9, 79.2, 43.4, 75.3, 89.6, 66.5, 54.0, 68.3],
        "ORG": [33.7, 20.6, 21.4, 31.2, 21.5, 19.3, 28.2, 29.2, 13.2, 20.8],
        "LOC": [78.5, 90.2, 78.7, 72.8, 75.2, 98.5, 90.9, 100.0, 124.8, 49.6],
        "ANIM": [15.5, 10.5, 34.4, 11.5, 7.3, 8.8, 11.4, 19.7, 14.7, 26.1],
        "BIO": [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4],
        "CEL": [2.8, 2.4, 2.1, 1.4, 1.2, 5.2, 2.3, 3.3, 4.2, 0.8],
        "DIS": [11.2, 8.6, 6.1, 5.2, 1.9, 6.5, 3.1, 6.5, 6.8, 0.1],
        "EVE": [3.2, 6.8, 4.7, 4.0, 2.8, 5.8, 7.4, 6.7, 5.9, 5.1],
        "FOOD": [11.0, 7.8, 5.6, 3.6, 3.2, 5.8, 3.2, 3.3, 5.4, 1.9],
        "INST": [0.4, 0.6, 0.2, 0.1, 1.1, 0.8, 0.7, 0.6, 0.6, 1.1],
        "MEDIA": [7.5, 8.0, 3.8, 2.8, 11.3, 8.6, 8.0, 4.9, 9.1, 55.9],
        "MYTH": [0.7, 1.6, 1.3, 0.8, 0.6, 1.8, 2.0, 1.3, 1.6, 1.8],
        "PLANT": [9.5, 7.6, 6.3, 7.8, 4.8, 5.1, 4.4, 6.6, 9.2, 6.1],
        "TIME": [3.2, 45.3, 31.0, 3.3, 22.8, 71.2, 27.4, 44.1, 48.6, 0.4],
        "VEHI": [0.5, 0.3, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.3, 0.3],
        "OTHER": [3.1, 3.8, 2.7, 2.4, 2.0, 4.2, 3.8, 2.5, 3.4, 3.4],
    }

    df = pl.DataFrame(_data).with_columns(
        (pl.col("OTHER") * 1000).alias("OTHER"),
        (pl.col("tokens") * 1000).alias("tokens"),
    )

    df = pl.concat(
        [
            df,
            df.select(pl.all().exclude("lang"))
            .sum()
            .with_columns(
                pl.all().round(3),
                pl.lit("TOTAL").alias("lang"),
            )
            .select(df.columns),
        ]
    )

    df
    return (df,)


@app.cell
def _(df):
    # relative sizes

    _tags = df.columns[df.columns.index("PER") :]
    df.with_columns(pl.sum_horizontal(pl.col(_tags)).alias("_row_sum")).select(
        ["lang"] + [(pl.col(c) / pl.col("_row_sum")).round(2).alias(c) for c in _tags]
    )
    return


if __name__ == "__main__":
    app.run()
