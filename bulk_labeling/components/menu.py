import io
import itertools
from functools import partial
from time import sleep
from typing import Callable, List

import pandas as pd
import solara
from solara.components.file_drop import FileInfo

from bulk_labeling.state import PlotState, State, reset
from bulk_labeling.utils.common import BUTTON_KWARGS
from bulk_labeling.utils.df import INTERNAL_COLS, apply_df_edits, filtered_df, load_df
from bulk_labeling.utils.menu import (
    PATH,
    add_new_label,
    assign_labels,
    get_assign_label_button_text,
)
from bulk_labeling.utils.ml import add_embeddings_to_df

NO_COLOR_COLS = INTERNAL_COLS + ["text"]


@solara.component
def assigned_label_view() -> None:
    State.assigned_new_label.use()
    State.filtered_ids.use()

    if State.assigned_new_label.value:
        solara.Info(f"{len(State.filtered_ids.value)} labeled!")
        sleep(2)
        State.assigned_new_label.set(False)


@solara.component
def assign_label_button(df: pd.DataFrame) -> None:
    fdf = filtered_df(df)
    btn_label, button_enabled = get_assign_label_button_text(df)
    solara.Button(
        btn_label,
        on_click=partial(assign_labels, fdf),
        disabled=not button_enabled,
        **BUTTON_KWARGS,
    )


@solara.component
def register_new_label_button() -> None:
    # TODO: Remove when solara updates
    State.available_labels.use()
    State.chosen_label.use()

    # TODO: Make a State.available_labels.append
    solara.InputText("Register new label", on_value=add_new_label)
    if State.available_labels.value:
        solara.Select("Available labels", list(State.available_labels.value)).connect(
            State.chosen_label
        )


@solara.component
def export_edits_button(df: pd.DataFrame) -> None:
    # TODO: Remove when solara updates
    State.labeled_ids.use()

    def export_edited_df() -> None:
        """Assigns the label and downloads the df to the user"""
        # TODO: Last thing! Allow the user to download the df
        exp_df = apply_df_edits(df)
        print(f"{len(exp_df)} rows edited")

    if State.labeled_ids.value:
        # Flatten all of the edits into a single set, so we know how many were edited
        num_edited = len(set(itertools.chain(*State.labeled_ids.value.values())))
        solara.Button(
            f"Export {num_edited} labeled points",
            on_click=export_edited_df,
            **BUTTON_KWARGS,
        )


@solara.component
def label_manager(df: pd.DataFrame) -> None:
    register_new_label_button()
    assign_label_button(df)
    export_edits_button(df)


@solara.component
def file_manager(set_df: Callable) -> None:
    PlotState.color.use()
    PlotState.loading.use()

    def load_demo_df() -> None:
        new_df = load_df(PATH)
        set_df(new_df)
        set_df(new_df)

    def load_file_df(file: FileInfo) -> None:
        if not file["data"]:
            return
        new_df = load_df(io.BytesIO(file["data"]))
        # Set it before embeddings so the user can see the df while embeddings load
        PlotState.loading.set(True)
        set_df(new_df)
        new_df = add_embeddings_to_df(new_df)
        # Set it again after embeddings so we can render the plotly graph
        set_df(new_df)
        PlotState.loading.set(False)

    solara.FileDrop(
        label="Drop CSV here (`text` col required)!",
        on_file=load_file_df,
        lazy=False,
    )
    with solara.Column():
        solara.Button(label="Load demo dataset", on_click=load_demo_df, **BUTTON_KWARGS)
        solara.Button(label="Reset view", on_click=reset, **BUTTON_KWARGS)


@solara.component
def view_controller(avl_cols: List[str]) -> None:
    # TODO: Remove when solara updates
    PlotState.color.use()
    PlotState.point_size.use()
    State.filter_text.use()

    solara.InputText(
        "Filter by search", State.filter_text.value, on_value=State.filter_text.set
    )
    solara.Markdown("**Set point size**")
    solara.SliderInt("", PlotState.point_size.value, on_value=PlotState.point_size.set)
    # TODO: A drop down should have "remove selection" option
    #  (esp if default state is None)
    solara.Select(
        "Color by",
        [None] + avl_cols,
        PlotState.color.value,
        on_value=PlotState.color.set,
    )


@solara.component
def menu(df: pd.DataFrame, set_df: Callable) -> None:
    State.reset_on_assignment.use()

    # avl_cols is dependent on df, so any time it changes,
    # this will automatically update
    set_cols = lambda: [i for i in df.columns if i not in NO_COLOR_COLS]
    avl_cols = solara.use_memo(set_cols, [df])

    file_manager(set_df)
    label_manager(df)
    view_controller(avl_cols)
    solara.Markdown(f"**Reset view on label assignment?**")
    if State.reset_on_assignment.value:
        label = "Reset"
    else:
        label = "Keep state"
    # solara.Checkbox(
    #     label=label,
    #     value=State.reset_on_assignment.value,
    #     on_value=State.reset_on_assignment.set
    # )
    solara.Checkbox(label=label).connect(State.reset_on_assignment)
    # solara.ToggleButtonsSingle(State.reset_on_assignment.value, values=[True, False], on_value=State.reset_on_assignment.set)
