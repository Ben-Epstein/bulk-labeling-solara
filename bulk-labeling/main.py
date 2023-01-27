import io
import os
from typing import Callable, Optional, cast

import numpy as np
import pandas as pd
import plotly.express as px
import solara
from solara.components.file_drop import FileInfo

DIR = f"{os.getcwd()}/bulk-labeling"
PATH = f"{DIR}/conv_intent.csv"


@solara.component
def table(df: pd.DataFrame):
    solara.Markdown("## Data")
    solara.DataFrame(df)


@solara.component
def embeddings(
    df: pd.DataFrame, color: str, point_size: int, set_selection_data: Callable
):
    solara.Markdown("## Embeddings")
    p = px.scatter(df, x="x", y="y", color=color or None)
    p.update_layout(showlegend=False)
    p.update_xaxes(visible=False)
    p.update_yaxes(visible=False)

    p.update_traces(marker_size=point_size)
    solara.FigurePlotly(p, on_selection=set_selection_data)


@solara.component
def df_view(
    df: pd.DataFrame, point_size: int, color: str, hard_reset: bool = False
) -> None:
    selection_data, set_selection_data = solara.use_state(None)
    if hard_reset:
        set_selection_data(None)
    if selection_data:
        df = df[df["id"].isin(selection_data["points"]["point_indexes"])]

    with solara.Columns([1, 1]):
        table(df)
        embeddings(df, color, point_size, set_selection_data)


@solara.component
def no_df() -> None:
    with solara.Columns([1, 1]):
        solara.Markdown("## DataFrame (Load Data)")
        solara.Markdown("## Embeddings (Load Data)")


@solara.component
def Page():
    df, set_df = solara.use_state(
        cast(Optional[pd.DataFrame], None), eq=lambda *args: False
    )
    hard_reset, set_reset = solara.use_state(False)
    point_size, set_point_size = solara.use_state(2)

    color, set_color = solara.use_state("")
    avl_cols, set_cols = solara.use_state([])

    def update_chosen_color(new_color: str):
        set_color(new_color)

    def update_point_size(new_point_size: int):
        set_point_size(new_point_size)

    def load_demo_df():
        new_df = pd.read_csv(PATH)
        set_df(new_df)
        set_cols(list(new_df.columns))

    def load_file_df(file: FileInfo):
        data = io.BytesIO(file["data"])
        new_df = pd.read_csv(data)
        new_df["x"] = np.random.rand(len(new_df))
        new_df["y"] = np.random.rand(len(new_df))
        new_df["text_length"] = new_df.text.str.len()
        set_df(new_df)
        set_cols(list(new_df.columns))

    def reset():
        # Force the child page (dataframe view) to reset selected points
        set_reset(True)
        # Set it back to False in order to preserve future lasso selections
        set_reset(False)

    with solara.Column():
        solara.Title("Bulk Labeling!")
        with solara.Sidebar():
            solara.FileDrop(
                label="Drop CSV here (`text` col required)!",
                on_file=load_file_df,
                lazy=False,
            )
            solara.Button(label="Or load demo dataset", on_click=load_demo_df)
            solara.Button(label="Reset view", on_click=reset)
            solara.Markdown("**Set point size**")
            solara.SliderInt("", point_size, on_value=update_point_size)
            solara.Select("Color by", "", avl_cols, on_value=update_chosen_color)
        with solara.Column():
            if df is not None:
                df["id"] = list(range(len(df)))
                df_view(df.copy(), point_size, color, hard_reset)
            else:
                no_df()
