import traceback
from concurrent.futures import Future
from typing import Any, TypedDict

import pandas as pd


class RasterProperties(TypedDict):
    filename: str
    filesize: int
    hash: str
    nodata: int | float
    bands: int
    compression: str
    epsg: int
    xres: float
    yres: float
    minval: int | float
    maxval: int | float
    has_voids: bool
    void_ratio: float
    nodata_internal: int
    nodata_boundary: int
    is_valid_cog: bool
    cog_errors: str
    cog_warnings: str


def write_report(filename: str, results: list[RasterProperties]) -> None:
    """
    Write RasterProperties to CSV file.

    Args:
        filename (str): Filename to write to.
        results (list[RasterProperties]): List of RasterProperties.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)


def write_failed(
    filename: str, failed: list[Future], future_to_kwargs: dict[Future, Any]
) -> None:
    """List the raster filename that resulted in an exception"""
    rows = []
    for future in failed:
        kwargs = future_to_kwargs[future]
        raster_filename = kwargs.get("raster_filename")
        rows.append({"path": raster_filename})
    failed = pd.DataFrame(rows, columns=["path"])
    failed.to_csv(filename, index=False)


def dump_exceptions(filename: str, failed: list[Future]) -> None:
    """Dump exceptions raised from failed futures"""
    exceptions = []
    for future in failed:
        exceptions.extend(traceback.format_exception(future.exception()))
        with open(filename, "w") as f:
            f.writelines(exceptions)
