from typing import cast

import pandas as pd
import solara

from bulk_labeling.components.df import df_view, no_embs
from bulk_labeling.components.menu import assigned_label_view, menu
from bulk_labeling.state import PlotState
from bulk_labeling.utils.df import has_df


@solara.component
def no_df() -> None:
    with solara.Columns([1, 1]):
        solara.Markdown("## DataFrame (Load Data)")
        solara.Markdown("## Embeddings (Load Data)")


@solara.component
def Page() -> None:
    # TODO: Remove when solara updates
    # PlotState.loading.use()

    # This `eq` makes it so every time we set the dataframe, solara thinks it's new
    df, set_df = solara.use_state(
        cast(pd.DataFrame, pd.DataFrame({})), eq=lambda *args: False
    )
    solara.Title("Bulk Labeling!")
    # TODO: Why cant i get this view to render?
    assigned_label_view()
    with solara.Sidebar():
        menu(df, set_df)
    if has_df(df) and PlotState.loading.value:
        no_embs(df)
    elif has_df(df):
        df_view(df)
    else:
        no_df()


if __name__ == "__main__":
    Page()
