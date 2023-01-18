import solara
import plotly.express as px
# import solara.express as px
import pandas as pd
from typing import cast, Optional, List

PATH = "conv_intent.csv"


@solara.component
def dataframe_view(df: pd.DataFrame, point_size: int, color: str) -> solara.HBox:
    with solara.HBox() as main:
        df_col = solara.VBox()
        embs_col = solara.VBox()

        with df_col:
            solara.DataFrame(df)
        with embs_col:
            p = px.scatter(df, x="x", y="y", color=color or None)
            p.update_layout(showlegend=False)
            p.update_xaxes(visible=False)
            p.update_yaxes(visible=False)

            p.update_traces(marker_size=point_size)
            solara.FigurePlotly(p)
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
    point_size, set_point_size = solara.use_state(2)

    color, set_color = solara.use_state("")
    avl_cols, set_cols = solara.use_state([])

    def update_chosen_color(new_color: str):
        set_color(new_color)

    # def update_avl_cols(new_cols: List[str]):
    #     set_cols(new_cols)

    def update_point_size(new_point_size: int):
        set_point_size(new_point_size)

    def load_df():
        new_df = pd.read_csv(PATH)
        set_df(new_df)
        set_cols(list(new_df.columns))

    def reset():
        set_df(None)

    main = solara.HBox()

    with main:
        solara.Title("Bulk Labeling!")
        with solara.VBox():
            solara.Button(label="Load demo dataset", on_click=load_df)
            solara.Button(label="Reset", on_click=reset)
            solara.SliderInt(
                "Set embedding point size", point_size, on_value=update_point_size
            )
            solara.Select(
                "Color embeddings by column", "", avl_cols, on_value=update_chosen_color
            )
        if df is not None:
            dataframe_view(df, point_size, color)
        else:
            no_df()
    return main
