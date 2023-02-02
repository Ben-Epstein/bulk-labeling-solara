import pandas as pd
import solara
from reacton import ipyvuetify as V

from bulk_labeling.state import PlotState, State
from bulk_labeling.utils.df import INTERNAL_COLS, filtered_df
from bulk_labeling.utils.plotly import create_plotly_figure, find_row_ids


@solara.component
def _emb_loading_state() -> None:
    solara.Markdown("## Embeddings")
    solara.Markdown("Loading your embeddings. Enjoy this fun animation for now")
    V.ProgressLinear(indeterminate=True)


@solara.component
def no_embs(df: pd.DataFrame) -> None:
    with solara.Columns([1, 1]):
        table(filtered_df(df))
        _emb_loading_state()


@solara.component
def embeddings(df: pd.DataFrame, color: str, point_size: int):
    solara.Markdown("## Embeddings")
    fig = create_plotly_figure(df, color, point_size)

    # Plotly returns data in a weird way, we just want the ids
    # TODO: Solara to handle :)
    set_point_ids = lambda data: State.filtered_ids.set(find_row_ids(fig, data))
    solara.FigurePlotly(fig, on_selection=set_point_ids)


@solara.component
def table(df: pd.DataFrame):
    solara.Markdown(f"## Data ({len(df):,} points)")
    solara.DataFrame(df[[i for i in df.columns if i not in INTERNAL_COLS]])


@solara.component
def df_view(df: pd.DataFrame) -> None:
    # TODO: Remove when solara updates
    PlotState.point_size.use()
    PlotState.color.use()
    State.filtered_ids.use()
    State.filter_text.use()

    fdf = filtered_df(df)

    with solara.Columns([1, 1]):
        table(fdf)
        embeddings(fdf, PlotState.color.value, PlotState.point_size.value)
