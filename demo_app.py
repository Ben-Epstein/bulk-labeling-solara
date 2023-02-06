from typing import Callable 
from functools import partial
import solara as sl
import pandas as pd
from solara.components.file_drop import FileInfo
import io

def load_file_df(file: FileInfo, set_df: Callable):
    if not file["data"]:
        return
    new_df = pd.read_csv(io.BytesIO(file["data"]))
    # Set it before embeddings so the user can see the df while embeddings load
    set_df(new_df)
    
@sl.component
def df_table(df: pd.DataFrame, num_samples: int, search: str) -> None:
    if not len(df):
        return
    filtered = df.copy()
    if "text" in df.columns:
        filtered = df[df["text"].str.contains(search)]
    sl.DataFrame(filtered[:num_samples])
    

@sl.component
def file_manager(set_df: Callable) -> None:
    sl.FileDrop(
        label="Drop CSV here (`text` col required)!",
        on_file=partial(load_file_df, set_df=set_df),
        lazy=False,
    )


@sl.component
def menu():
    df, set_df = sl.use_state(pd.DataFrame({}))
    num_samples, set_samples = sl.use_state(10)
    search, set_search = sl.use_state("")
    
    with sl.Sidebar():
        sl.InputText("What would you like to see?", search, on_value=set_search)
        sl.SliderInt("How many samples?", num_samples, max=100, on_value=set_samples)
        file_manager(set_df)
    df_table(df, num_samples, search)
    

@sl.component
def Page():
    menu()
