import io
import itertools
import os
from collections import defaultdict
from time import sleep
from typing import Callable, Dict, List, Optional, Set, cast

import numpy as np
import pandas as pd
import plotly.express as px
import solara

# from sentence_transformers import SentenceTransformer
from reacton import ipyvuetify as V
from solara.components.file_drop import FileInfo
from solara.lab import Reactive
from umap import UMAP

DIR = f"{os.getcwd()}/bulk-labeling"
PATH = f"{DIR}/conv_intent.csv"

UMAP_MODEL = UMAP(n_neighbors=15, random_state=42, verbose=True)
# ENCODER = SentenceTransformer("paraphrase-MiniLM-L3-v2")

INTERNAL_COLS = ["x", "y", "hovertext", "id"]
NO_COLOR_COLS = INTERNAL_COLS + ["text"]
BUTTON_KWARGS = dict(color="primary", text=True, outlined=True)


class State:
    available_labels = Reactive[Set[str]](set())
    labeled_ids = Reactive[Dict[str, List[int]]](defaultdict(list))
    filtered_ids = Reactive[List[int]]([])
    chosen_label = Reactive[Optional[str]](None)
    assigned_new_label = Reactive[bool](False)
    filter_text = Reactive[str]("")
    df = Reactive[pd.DataFrame](pd.DataFrame({}))

    @staticmethod
    def filtered_df() -> pd.DataFrame:
        df = State.df.value.copy()
        if State.filtered_ids.value:
            df = df[df["id"].isin(State.filtered_ids.value)]
        if State.filter_text.value:
            df = df[df["text"].str.contains(State.filter_text.value)]
        return df

    @staticmethod
    def has_df() -> bool:
        return len(State.df.value) != 0




class PlotState:
    point_size = Reactive[int](2)
    color = Reactive[str]("")
    # While we calculate embeddings and UMAP, we can manage the loading state
    loading = Reactive[bool](False)


def encode_inputs(samples: List[str]) -> np.ndarray:
    # return ENCODER.encode(samples)
    return np.random.rand(len(samples), 20)


def get_xy(embs: np.ndarray) -> np.ndarray:
    return UMAP_MODEL.fit_transform(embs)


def get_text_embeddings(samples: List[str]) -> np.ndarray:
    return get_xy(encode_inputs(samples))


def find_row_ids(fig, click_data) -> List[int]:
    """A very annoying function to get row IDs because Plotly is unhelpful

    Solara is going to do this for us in the future!
    """
    # goes from trace index and point index to row index in a dataframe
    # requires passing df.index as to custom_data
    trace_index = click_data["points"]["trace_indexes"]
    point_index = click_data["points"]["point_indexes"]
    point_ids = []
    for t, p in zip(trace_index, point_index):
        point_trace = fig.data[t]
        point_ids.append(point_trace.customdata[p][0])
    return point_ids


def reset():
    """Removes any filters applied to the data"""
    State.filtered_ids.set([])
    State.filter_text.set("")
    State.chosen_label.set(None)


@solara.component
def assigned_label_view() -> None:
    State.assigned_new_label.use()
    if State.assigned_new_label.value:
        print("Should be an info!")
        solara.Info(f"{len(State.filtered_ids.value)} labeled!")
        sleep(2)
        State.assigned_new_label.set(False)


@solara.component
def table():
    State.df.use()

    df = State.filtered_df()
    solara.Markdown(f"## Data ({len(df):,} points)")
    solara.DataFrame(df[[i for i in df.columns if i not in INTERNAL_COLS]])


@solara.component
def embeddings():
    # TODO: Remove once solara handles this for me
    State.df.use()
    PlotState.color.use()
    PlotState.point_size.use()

    solara.Markdown("## Embeddings")
    # We pass in df.id to custom_data so we can get back the correct points on a
    # lasso selection. Plotly makes this difficult
    # TODO: Solara will wrap and handle all of this logic for us in the future
    df = State.filtered_df()
    p = px.scatter(
        df,
        x="x",
        y="y",
        color=PlotState.color,
        custom_data=[df["id"]],
        hover_data=["hovertext"],
    )
    p.update_layout(showlegend=False)
    p.update_xaxes(visible=False)
    p.update_yaxes(visible=False)
    p.update_traces(marker_size=PlotState.point_size)

    # Plotly returns data in a weird way, we just want the ids
    # TODO: Solara to handle :)
    set_point_ids = lambda data: State.filtered_ids.set(find_row_ids(p, data))
    solara.FigurePlotly(p, on_selection=set_point_ids)


@solara.component
def df_view() -> None:
    with solara.Columns([1, 1]):
        table()
        embeddings()


@solara.component
def no_df() -> None:
    with solara.Columns([1, 1]):
        solara.Markdown("## DataFrame (Load Data)")
        solara.Markdown("## Embeddings (Load Data)")


@solara.component
def _emb_loading_state() -> None:
    solara.Markdown("## Embeddings")
    solara.Markdown("Loading your embeddings. Enjoy this fun animation for now")
    V.ProgressLinear(indeterminate=True)


@solara.component
def no_embs() -> None:
    with solara.Columns([1, 1]):
        table()
        _emb_loading_state()


@solara.component()
def label_manager() -> None:
    State.chosen_label.use()
    State.filtered_ids.use()
    State.filter_text.use()
    State.labeled_ids.use()

    df = State.filtered_df()

    def add_new_label(new_label: str) -> None:
        all_labels = State.available_labels.value.copy()
        all_labels.add(new_label)
        State.available_labels.set(all_labels)
        # So the "assign points" button is already pre-populated with your new label =]
        State.chosen_label.set(new_label)

    def assign_labels() -> None:
        print(
            f"Setting {State.chosen_label.value} for "
            f"{len(State.filtered_ids.value)} points"
        )

        labeled_ids = State.labeled_ids.value.copy()
        new_ids = State.filtered_ids.value.copy()
        # In the event that they've loaded a dataframe but haven't selected any points
        # to label, they are labeling all of the points. So set IDs to all df ids
        if not new_ids and df is not None:
            new_ids = list(range(len(df)))
        labeled_ids[State.chosen_label.value].extend(new_ids)
        State.labeled_ids.set(labeled_ids)
        # State.assigned_new_label.set(True)
        # Reset the view so no points are selected
        reset()

    def export_edited_df() -> None:
        """Assigns the label and downloads the df to the user"""
        # TODO: Last thing! Allow the user to download the df
        print("Should be downloading!")
        df2 = df.copy()
        labeled_ids = State.labeled_ids.value
        # Map every ID to it's assigned labels
        # TODO: We can be smarter with conflicts and pick the label that an ID is
        #  assigned to most frequently
        id_label = {id_: label for label, ids in labeled_ids.items() for id_ in ids}
        df2["label"] = df2["id"].apply(lambda id_: id_label.get(id_, "-1"))
        df2 = df2[df2["label"] != "-1"]
        cols = [c for c in df2.columns if c not in INTERNAL_COLS]
        return df2[cols]

    # TODO: Make a State.available_labels.append
    solara.InputText("Register new label", on_value=add_new_label)
    if State.available_labels.value:
        solara.Select("Available labels", list(State.available_labels.value)).connect(
            State.chosen_label
        )
    if State.chosen_label.value and State.available_labels.value:
        button_disabled = State.has_df()
        btn_label = (
            "Load a df to label"
            if button_disabled
            else f"Assign {len(df)} points to label {State.chosen_label.value}"
        )
        solara.Button(
            btn_label, on_click=assign_labels, disabled=button_disabled, **BUTTON_KWARGS
        )
    if State.labeled_ids.value and State.has_df():
        # Flatten all of the edits into a single set, so we know how many were edited
        num_edited = len(set(itertools.chain(*State.labeled_ids.value.values())))
        solara.Button(
            f"Export {num_edited} labeled points",
            on_click=export_edited_df,
            **BUTTON_KWARGS,
        )

@solara.component()
def menu() -> None:
    # TODO: Remove when solara updates
    PlotState.point_size.use()
    PlotState.color.use()
    State.available_labels.use()
    State.chosen_label.use()
    State.filtered_ids.use()
    State.filter_text.use()
    State.labeled_ids.use()
    State.df.use()

    # avl_cols is dependent on df, so any time it changes,
    # this will automatically update
    df = State.filtered_df()
    # set_cols = lambda: [i for i in df.columns if i not in NO_COLOR_COLS]
    # avl_cols = solara.use_memo(set_cols, [df])

    def _set_default_cols(df: pd.DataFrame) -> pd.DataFrame:
        df["text_length"] = df.text.str.len()
        df["id"] = list(range(len(df)))
        df["hovertext"] = df.text.str.wrap(30).str.replace("\n", "<br>")
        return df

    def load_demo_df():
        new_df = pd.read_csv(PATH)
        new_df = _set_default_cols(new_df)
        State.df.set(new_df)

    def load_file_df(file: FileInfo):
        data = io.BytesIO(file["data"])
        new_df = pd.read_csv(data)
        new_df = _set_default_cols(new_df)
        # Set it before embeddings so the user can see the df while embeddings load
        PlotState.loading.set(True)
        State.df.set(new_df)
        embs = get_text_embeddings(new_df["text"].tolist())
        new_df["x"] = embs[:, 0]
        new_df["y"] = embs[:, 1]
        # Set it again after embeddings so we can render the plotly graph
        State.df.set(new_df)
        PlotState.loading.set(False)

    solara.FileDrop(
        label="Drop CSV here (`text` col required)!",
        on_file=load_file_df,
        lazy=False,
    )
    with solara.Column():
        solara.Button(label="Load demo dataset", on_click=load_demo_df, **BUTTON_KWARGS)
        solara.Button(label="Reset view", on_click=reset, **BUTTON_KWARGS)
    label_manager()
    solara.InputText(
        "Filter by search", State.filter_text.value, on_value=State.filter_text.set
    )
    solara.Markdown("**Set point size**")
    solara.SliderInt("", PlotState.point_size.value, on_value=PlotState.point_size.set)
    # TODO: A drop down should have "remove selection" option
    #  (esp if default state is None)
    avl_cols=[i for i in df.columns if i not in NO_COLOR_COLS]
    solara.Select(
        "Color by",
        [None] + avl_cols,
        PlotState.color.value,
        on_value=PlotState.color.set,
    )


@solara.component
def Page():
    # TODO: Remove when solara updates
    PlotState.loading.use()
    State.df.use()
    # This `eq` makes it so every time we set the dataframe, solara thinks it's new
    # df, set_df = solara.use_state(
    #     cast(Optional[pd.DataFrame], None), eq=lambda *args: False
    # )
    solara.Title("Bulk Labeling!")
    # TODO: Why cant i get this view to render?
    assigned_label_view()
    with solara.Sidebar():
        menu()
    if State.has_df() and PlotState.loading.value:
        no_embs()
    elif State.has_df():
        df_view()
    else:
        no_df()
