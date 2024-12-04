import hashlib
import logging
import os
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
import shapely.ops
from rio_cogeo import cog_info
from tqdm import tqdm

from timing import timeit
from walk import list_files

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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


def crc_checksum(file_path):
    prev = 0
    for eachLine in open(file_path, "rb"):
        prev = zlib.crc32(eachLine, prev)
    return "%X" % (prev & 0xFFFFFFFF)


def md5_checksum(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_void_mask(
    dataset: rasterio.DatasetReader,
    boundary_geom: shapely.geometry.Polygon | list[shapely.geometry.Polygon],
    raster_difference: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """
    Returns a boolean mask array where True indicates void pixels i.e. pixels of nodata contained

    Args:
        dataset (rasterio.DatasetReader): DatasetReader of a raster image to detect voids.
        boundary_geom (shapely.geometry.Polygon): Geometry describing the valid data regions.
        raster_difference (bool): If True (default), return the XOR between the raster mask
            and the boundary mask. If False, return the boundary mask.

    Returns:
        np.ndarray: Boolean np.ndarray where True indicates void pixels
    """
    dtype = kwargs.pop("dtype", "uint8")
    fill = kwargs.pop("fill", 0)
    default_value = kwargs.pop("default_value", 255)
    all_touched = kwargs.pop("all_touched", True)

    if isinstance(boundary_geom, shapely.geometry.Polygon):
        boundary_geom = [boundary_geom]
    boundary_arr = rasterio.features.rasterize(
        boundary_geom,
        out_shape=dataset.shape,
        transform=dataset.transform,
        dtype=dtype,
        fill=fill,
        default_value=default_value,
        all_touched=all_touched,
        **kwargs,
    )

    # mask array where True indicates void pixels i.e. pixels of nodata contained
    # within the boundary geometry
    if raster_difference:
        void_mask = (dataset.read_masks(1) ^ boundary_arr).astype("bool")
    else:
        void_mask = boundary_arr.astype("bool")

    return void_mask


def compute_void_ratio(void_mask: np.ndarray) -> float:
    void_pixel_count = void_mask.sum()  # only True values are counted
    total_pixel_count = void_mask.size
    void_ratio = void_pixel_count / total_pixel_count

    return void_ratio


def contains_voids(
    void_mask: np.ndarray, void_threshold: float = 0.01, strict: bool = False
) -> bool:
    """
    True if void mask array contains void pixels. False otherwise.

    Args:
        void_mask (np.ndarray): Void mask array, generally the output of get_void_mask function.
        void_threshold (float, optional): Allowed ratio of void pixels in void mask array.
            Computed as the ratio of void pixels / total pixels. Defaults to 0.01 (1%).
        strict (bool, optional): Return False if any void pixels in void mask array. Defaults to False.

    Returns:
        bool: True if void mask array contains void pixels. False otherwise.
    """
    if strict:
        return bool(void_mask.any())

    void_ratio = compute_void_ratio(void_mask)
    if void_ratio > void_threshold:
        return True
    return False


def compute_void_classification(
    void_mask: np.ndarray,
    boundary_mask: np.ndarray,
    shapes_options: Optional[dict[str, Any]] = None,
) -> tuple:
    if shapes_options is None:
        shapes_options = dict()
    connectivity = shapes_options.pop("connectivity", 8)

    assert void_mask.dtype == "bool", "void_mask must be a boolean array"
    assert boundary_mask.dtype == "bool", "boundary_mask must be a boolean array"

    count_internal = 0
    count_boundary = 0
    for geom, value in rasterio.features.shapes(
        void_mask.astype("uint8"), connectivity=connectivity, **shapes_options
    ):
        # void pixels are True
        if value:
            points = geom["coordinates"][0]
            count_line = 0
            for point in points:
                if boundary_mask[int(point[1] - 1), int(point[0] - 1)]:
                    count_line += 1
            if count_line == 0:
                count_internal += 1
            else:
                if count_line != len(points):
                    count_boundary += 1
    return (count_internal, count_boundary)


@timeit
def get_raster_properties(
    raster_filename: str,
    boundary: gpd.GeoDataFrame,
    strict: bool = False,
    void_threshold: float = 0.01,
    classify_voids: bool = False,
    geom_buffer_size: float = 1,
    env_options: Optional[dict[str, Any]] = None,
    rasterize_options: Optional[dict[str, Any]] = None,
) -> RasterProperties:
    if env_options is None:
        env_options = dict()
    if rasterize_options is None:
        rasterize_options = dict()

    raster_filename = Path(raster_filename)
    hashval = md5_checksum(raster_filename)
    filesize = raster_filename.stat().st_size
    info = cog_info(raster_filename)
    nodata = info.Profile.Nodata
    bands = info.Profile.Bands
    compression = info.Compression
    xres, yres = info.GEO.Resolution
    is_valid_cog = info.COG
    cog_errors = info.COG_errors
    cog_warnings = info.COG_warnings

    with rasterio.Env(**env_options):
        with rasterio.open(raster_filename) as src:
            # if unset, assume same crs as raster, else reproject to raster crs
            if boundary.crs is None:
                boundary.crs = src.crs
            elif boundary.crs != src.crs:
                boundary = boundary.to_crs(src.crs)
            boundary_geom = boundary.geometry

            epsg = src.crs.to_epsg()
            stats = src.stats(indexes=1)[0]
            minval = stats.min
            maxval = stats.max
            void_arr = get_void_mask(src, boundary_geom, **rasterize_options)
            has_voids = contains_voids(
                void_arr, void_threshold=void_threshold, strict=strict
            )
            void_ratio = compute_void_ratio(void_arr)
            nodata_internal = 0
            nodata_boundary = 0
            if classify_voids and has_voids:
                # rasterize buffered perimeter of the boundary geometry
                boundary_arr = get_void_mask(
                    src,
                    boundary.exterior.buffer(geom_buffer_size),
                    raster_difference=False,
                    **rasterize_options,
                )
                nodata_internal, nodata_boundary = compute_void_classification(
                    void_arr, boundary_arr, shapes_options=dict(connectivity=8)
                )

    return RasterProperties(
        filename=str(raster_filename),
        filesize=filesize,
        hash=hashval,
        nodata=nodata,
        bands=bands,
        compression=compression,
        epsg=epsg,
        xres=xres,
        yres=yres,
        minval=minval,
        maxval=maxval,
        has_voids=has_voids,
        void_ratio=void_ratio,
        nodata_internal=nodata_internal,
        nodata_boundary=nodata_boundary,
        is_valid_cog=is_valid_cog,
        cog_errors=cog_errors,
        cog_warnings=cog_warnings,
    )


@timeit
def main():
    MAX_RETRY_FAILED_FUTURES = 3  # maximum number of retries for failed futures
    MAX_WORKERS = (
        os.cpu_count()
    )  # numbers of workers to use for parallel processing, default is number of CPU cores.
    DEFAULT_ENV_OPTIONS = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_PAM_ENABLED": "NO",
    }
    MAX_ITEMS = 100  # short-circuit for testing

    data_root = Path("./data/17811_US_LP_2024_Enbridge_test")
    boundary_file = next(data_root.rglob("*Boundary*.geojson"))
    boundary_df = gpd.read_file(boundary_file)
    flist = list_files(data_root, extensions=["tif", "tiff"])
    if MAX_ITEMS:
        flist = flist[:MAX_ITEMS]
    future_to_kwargs = dict()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for f in flist:
            kwargs = dict(
                raster_filename=str(f),
                boundary=boundary_df,
                strict=True,
                classify_voids=True,
                env_options=DEFAULT_ENV_OPTIONS,
            )
            future_to_kwargs[executor.submit(get_raster_properties, **kwargs)] = kwargs

        results = []
        failed_futures = []
        logger.info("processing %s futures", len(future_to_kwargs))
        for future in tqdm(as_completed(future_to_kwargs.keys())):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("%s raised an exception: %s", future, e)
                failed_futures.append(future)

        # retry failed futures up to N times
        retry_count = 0
        while retry_count < MAX_RETRY_FAILED_FUTURES and failed_futures:
            logger.info("retrying %s failed futures, attempt %s", len(failed_futures), retry_count)
            _futures = []
            for f in failed_futures:
                kwargs = future_to_kwargs[f]
                _f = executor.submit(get_raster_properties, **kwargs)
                _futures.append(_f)
                future_to_kwargs[_f] = kwargs
            failed_futures = []
            for future in as_completed(_futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("%s raised an exception: %s", future, e)
                    failed_futures.append(future)
            retry_count += 1

        # log filename for failed and full exceptions
        if failed_futures:
            logger.info("failed futures: %s", failed_futures)
            rows = []
            exceptions = []
            for future in failed_futures:
                kwargs = future_to_kwargs[future]
                raster_filename = kwargs.get("raster_filename")
                rows.append((raster_filename,))
                exceptions.extend(traceback.format_exception(future.exception()))
            failed = pd.DataFrame(rows, columns=["path"])
            failed.to_csv("failed.csv", index=False)
            with open("exceptions.txt", "w") as f:
                f.writelines(exceptions)

    df = pd.DataFrame(results)
    df.to_csv("out.csv", index=False)


if __name__ == "__main__":
    main()
