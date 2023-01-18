import os
from typing import Optional, cast

import pandas as pd
import plotly.express as px
import solara

DIR = f"{os.getcwd()}/bulk-labeling"
PATH = f"{DIR}/conv_intent.csv"


@solara.component
def dataframe_view(
    df: pd.DataFrame, point_size: int, color: str, hard_reset: bool = False
) -> solara.HBox:
    selection_data, set_selection_data = solara.use_state(None)
    if hard_reset:
        set_selection_data(None)
    if selection_data:
        df = df[df["id"].isin(selection_data["points"]["point_indexes"])]
    with solara.HBox() as main:
        df_col = solara.VBox()
        embs_col = solara.VBox()

        with df_col:
            solara.Markdown("## Data")
            solara.DataFrame(df)
        with embs_col:
            solara.Markdown("## Embeddings")
            p = px.scatter(df, x="x", y="y", color=color or None)
            p.update_layout(showlegend=False)
            p.update_xaxes(visible=False)
            p.update_yaxes(visible=False)

            p.update_traces(marker_size=point_size)
            solara.FigurePlotly(p, on_selection=set_selection_data)
    return main


@solara.component
def no_df():
    with solara.HBox() as main:
        df_col = solara.VBox()
        embs_col = solara.VBox()
        with df_col:
            solara.Markdown("Load a dataset to see some cool stuff")
        with embs_col:
            solara.Markdown("Load a dataset to see some cool stuff")

    return main


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

    def load_df():
        new_df = pd.read_csv(PATH)
        set_df(new_df)
        set_cols(list(new_df.columns))

    def reset():
        # Force the child page (dataframe view) to reset selected points
        set_reset(True)
        # Set it back to False in order to preserve future lasso selections
        set_reset(False)

    main = solara.HBox()

    with main:
        solara.Title("Bulk Labeling!")
        with solara.VBox():
            solara.Button(label="Load demo dataset", on_click=load_df)
            solara.Button(label="Reset view", on_click=reset)
            solara.Markdown("**Set point size**")
            solara.SliderInt("", point_size, on_value=update_point_size)
            solara.Select("Color by", "", avl_cols, on_value=update_chosen_color)
        if df is not None:
            df["id"] = list(range(len(df)))
            dataframe_view(df.copy(), point_size, color, hard_reset)
        else:
            no_df()
    return main
