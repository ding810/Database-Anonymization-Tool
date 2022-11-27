import pandas as pd
import warnings
from typing import Tuple

def filter_sens_attr(table: pd.DataFrame, sens_attr: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    if sens_attr in table.columns:
        sens_attr_column = table[sens_attr]
    else:
        if sens_attr is None:
            warnings.warn("no sensitive attribute given")
        else:
            warnings.warn("sensitive attribute {} not found in dataset".format(sens_attr))
        sens_attr_column = table[table.columns[-1]]
        sens_attr = sens_attr_column.name
        warnings.warn("using {} as sensitive attribute".format(sens_attr))
    table = table.drop(sens_attr, axis=1)
    return (table, sens_attr_column)