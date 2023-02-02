import os

from bulk_labeling.state import State

DIR = f"{os.getcwd()}/bulk-labeling"
PATH = f"{DIR}/conv_intent.csv"


def add_new_label(new_label: str) -> None:
    if not new_label:
        return
    all_labels = State.available_labels.value.copy()
    all_labels.add(new_label)
    State.available_labels.set(all_labels)
    # So the "assign points" button is already pre-populated with your new label =]
    State.chosen_label.set(new_label)
