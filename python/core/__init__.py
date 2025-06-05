from pyiganet import pyiganet_core as _pyiganet_core


def Options():
    """
    Creator function for Options objects. Based on the input, a
    corresponding backend class will be created
    """
    return eval(f"_pyiganet_core.core.Options")()


def BlockTensor(rows, cols=None):
    """
    Creator function for block tensors. Based on the input, a corresponding
    backend class will be created

    Parameters
    ----------
    rows: int
      number of row blocks
    cols: int
      number of column blocks

    Returns
    -------
    block_tensor: BlockTensor
    """
    rows_str = f"_{rows}"

    if cols is None:
        cols = 1

    cols_str = f"_{cols}"

    return eval(f"_pyiganet_core.core.BlockTensor{rows_str}{cols_str}")()
