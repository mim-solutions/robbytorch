from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import matplotlib.pyplot as plt

from .input_transforms import PGD


class MatplotLabel:
    """A text label with matplotlib attributes such as color etc.

    This is essentially a dictionary of kwargs for `matplotlib.text.Text`.
    Adding a string concatenates. Adding another label concatenates text and updates other kwargs.
    Note the same kwargs have to apply to the whole label.
    Thus `MatplotLabel('a', color='red') + MatplotLabel('b', color='blue')` is just
         `MatplotLabel('ab', color='blue')`.

    Some popular kwargs are:
        - color, background-color: eg. 0.2, (1.0, 0.2, 0.3), 'red', '#ff0f0faa' ...
        - fontfamily or family: eg. 'sans-serif', 'monospace', 'cursive', 'fantasy', FONTNAME
        - fontsize or size: eg. 12.5, 'xx-small', 'x-small', 'small', 'large', ...
        - fontstretch or stretch: eg. 500 (a value 0-1000), 'ultra-condensed', 'semi-expanded', ...
        - fontstyle or style: 'normal', 'italic', 'oblique'
        - fontvariant or variant: 'normal', 'small-caps'
        - fontweight or weight: eg. 800 (a value 0-1000), 'ultralight', 'semibold', 'heavy', ...
        - horizontalalignment or ha: 'center', 'right', 'left'
    See `https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text`.
    For colors see `https://matplotlib.org/stable/tutorials/colors/colors.html`.
    """

    def __init__(self, text: Union[str, "MatplotLabel"] = "", **kwargs):
        if isinstance(text, str):
            self.text: str = text
        else:
            self.__dict__.update(text.__dict__)
        self.__dict__.update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def __add__(self, other: Union["MatplotLabel", str]) -> "MatplotLabel":
        if isinstance(other, str):
            return MatplotLabel(**{**self.__dict__, "text": self.text + other})
        elif isinstance(other, MatplotLabel):
            text = self.text + other.text
            return MatplotLabel(**{**self.__dict__, **other.__dict__, "text": text})
        else:
            raise NotImplementedError

    def __radd__(self, other: str) -> "MatplotLabel":
        if isinstance(other, str):
            return MatplotLabel(**{**self.__dict__, "text": other + self.text})
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.text)

    def __bool__(self) -> bool:
        return bool(self.text)


Label = Union[str, MatplotLabel]


def get_image_table(
    *items: Union[Label, Iterable[Label], Iterable[torch.Tensor], Iterable[int], Iterable[bool]],
    size=(2.5, 2.5),
    fontsize=12,
    trim_rows=0,
    classes: Optional[List[str]] = None
):
    """Return a matplotlib table figure of images and labels.

    Example usage: ```
        get_image_table("gnd truth: ",    labels,
                        "\nprediction: ", pred_labels, labels == pred_labels,
                        "Orig",           orig_images,
                        "mod prediction: ", mod_labels, labels == mod_labels,
                        "Modified",         mod_images,
                        classes=dataset.classes).show()
    ```
    The positional arguments are either:
    - a single string, used as a label for the next row
    - an iterable of strings, used as column labels atop of the next row
    - an iterable of images, displayed as a row
    - an iterable of bools, used to color column labels green/red
    - an iterable of integers i, changed to classes[i]
    - a single special string "---", attaching column labels to the bottom of the previous row
      (rather than the top of the next row)
    A row label before a list of strings is prepended to all those strings.
    A row label before a list of images is displayed left of it.
    Consecutive lists of strings concatenated (labels are added as `MatplotLabel`).

    Keyword arguments:
        - `size`: a (height, width) pair for individual image cells.
    """
    Cell = Tuple[Label, Optional[torch.Tensor], Label]  # top label, img, bottom label
    Row = Tuple[Label, List[Cell]]  # left label for the whole row, list of cells

    row_label: Label = ""
    col_labels: List[Label] = []

    rows: List[Row] = []
    for item in [*items, "---"]:  # Always end with an implicit "---".
        # A single string
        if isinstance(item, (str, MatplotLabel)):
            # A row label
            if item != "---":
                assert not row_label, (
                    "Unexpected row-label followed by row-label."
                    " Did you mean to give a list of column-labels?"
                )
                row_label = item
            # A single "---" special string: use col_labels as bottom labels.
            else:
                assert not row_label, (
                    "Unexpected row-label followed by '---'."
                    " Did you mean to give a list of column-labels?"
                )
                if not col_labels:
                    continue
                if not rows:
                    label: Label = ""
                    cells: List[Cell] = [("", None, "")] * len(col_labels)
                else:
                    label, cells = rows.pop()
                assert len(cells) == len(col_labels), "All rows should have the same length."
                cells = [(top, img, b1 + b2) for (top, img, b1), b2 in zip(cells, col_labels)]
                rows.append((label, cells))
                col_labels = []
            continue

        lst: List[Any] = list(item)
        if trim_rows:
            lst = lst[:trim_rows]
        # Cast scalar tensors to their value.
        if isinstance(lst[0], torch.Tensor) and lst[0].numel() == 1:
            lst = [i.item() for i in lst]
        # Cast bools to red/green font color.
        if isinstance(lst[0], bool):
            lst = [MatplotLabel(color="green" if b else "red") for b in lst]
        # Cast ints to strings by mapping i to classes[i].
        if isinstance(lst[0], int):
            assert classes, "Argument `classes` is required for integer rows."
            lst = [classes[i] for i in lst]

        # A list of strings.
        if isinstance(lst[0], (str, MatplotLabel)):
            if row_label:
                lst = [row_label + s for s in lst]
                row_label = ""
            if not col_labels:
                col_labels = [""] * len(lst)
            else:
                assert len(lst) == len(col_labels), "All rows should have the same length."
            col_labels = [a + b for a, b in zip(col_labels, lst)]
        # A list of images.
        else:
            if not col_labels:
                col_labels = [""] * len(lst)
            else:
                assert len(lst) == len(col_labels), "All rows should have the same length."
            cells = [(top, img, "") for top, img in zip(col_labels, lst)]
            rows.append((row_label, cells))
            row_label = ""
            col_labels = []

    n_rows = len(rows)
    n_cols = len(rows[0][1])

    figsize = (size[0] * n_cols, size[1] * n_rows)
    figure, axarr = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for r in range(n_rows):
        row_label, row = rows[r]
        assert len(row) == n_cols, "All rows should have the same length."
        for c in range(n_cols):
            top, img, bot = row[c]
            ax = axarr[r][c]
            if img is not None:
                ax.imshow(img.permute(1, 2, 0).cpu())  # CHW to HWC
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            if row_label and c == 0:
                row_label = MatplotLabel(fontsize=fontsize) + row_label
                ax.set_ylabel(row_label.text, **row_label.as_dict())
            if top:
                top = MatplotLabel(fontsize=fontsize) + top
                ax.set_title(top.text, **top.as_dict())
            if bot:
                bot = MatplotLabel(fontsize=fontsize) + bot
                ax.set_xlabel(bot.text, **bot.as_dict())
    return figure


def explain(model, dataitem, forward, cols: int=8, rows: int=4, 
           classes: Optional[List[str]]=None, labels: Optional[List[int]]=None, 
           loss_key: str='loss', eps: float=80, step_size: float=20, Nsteps: int=60, use_tqdm: bool=True, minimize: bool=True):
    """Prints images modified toward minimization/maximization of value under "loss" key of the dict 
    returned by the `forward` callable.

    Pass `classes` and `labels` to match int targets to str labels. Other params are forwarded to the PGD.
    """

    rows = min(len(dataitem["data"]) // cols, rows)
    
    advs = PGD(
        model, dataitem, forward, loss_key=loss_key,
        eps=eps, step_size=step_size, Nsteps=Nsteps, use_tqdm=use_tqdm, minimize=minimize
    )

    for start, end in [(i*cols, (i+1)*cols) for i in range(rows)]:
        args = []
        if labels is not None:
            args += ["label: ", labels[start:end]]
        args += [
            "original:", dataitem['data'][start:end],
            ("targeted:" if minimize else "avoided:"), advs[start:end]
        ]
        _ = get_image_table(*args, classes=classes)