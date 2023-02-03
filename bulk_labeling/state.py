from collections import defaultdict
from typing import Dict, List, Optional, Set

from solara.lab import Reactive


class State:
    available_labels = Reactive[Set[str]](set())
    labeled_ids = Reactive[Dict[str, List[int]]](defaultdict(list))
    filtered_ids = Reactive[List[int]]([])
    chosen_label = Reactive[Optional[str]](None)
    assigned_new_label = Reactive[bool](False)
    filter_text = Reactive[str]("")


class PlotState:
    point_size = Reactive[int](2)
    color = Reactive[str]("")
    # While we calculate embeddings and UMAP, we can manage the loading state
    loading = Reactive[bool](False)


def reset() -> None:
    """Removes any filters applied to the data"""
    State.filtered_ids.set([])
    State.filter_text.set("")
    State.chosen_label.set(None)
