import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TypedDict
import logging
import traceback

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
import shapely.ops
from rasterio.plot import show
from rio_cogeo import cog_info

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
    cellsize: tuple[float, float]
    minval: int | float
    maxval: int | float
    has_voids: bool
    nodata_internal: int
    nodata_boundary: int


def get_void_mask(
    dataset: rasterio.DatasetReader,
    boundary_geom: shapely.geometry.Polygon | list[shapely.geometry.Polygon],
    **kwargs: Any,
) -> np.ndarray:
    """
    Returns a boolean mask array where True indicates void pixels i.e. pixels of nodata contained

    Args:
        dataset (rasterio.DatasetReader): DatasetReader of a raster image to detect voids.
        boundary_geom (shapely.geometry.Polygon): Geometry describing the valid data regions.


    Returns:
        np.ndarray: Boolean np.ndarray where True indicates void pixels
    """
    if isinstance(boundary_geom, shapely.geometry.Polygon):
        boundary_geom = [boundary_geom]
    boundary_arr = rasterio.features.rasterize(
        boundary_geom,
        out_shape=dataset.shape,
        transform=dataset.transform,
        dtype="uint8",
        fill=0,
        default_value=255,
        **kwargs,
    )

    # mask array where True indicates void pixels i.e. pixels of nodata contained
    # within the boundary geometry
    void_mask = (dataset.read_masks(1) ^ boundary_arr).astype("bool")

    return void_mask


def contains_voids(
    void_mask: np.ndarray, allowed_threshold: float = 0.05, strict: bool = False
) -> bool:
    """
    True if void mask array contains void pixels. False otherwise.

    Args:
        void_mask (np.ndarray): Void mask array, generally the output of get_void_mask function.
        allowed_threshold (float, optional): Allowed ratio of void pixels in void mask array.
            Computed as the ratio of void pixels / total pixels. Defaults to 0.05 (5%).
        strict (bool, optional): Return False if any void pixels in void mask array. Defaults to False.

    Returns:
        bool: True if void mask array contains void pixels. False otherwise.
    """
    if strict:
        return bool(void_mask.any())

    void_pixel_count = void_mask.sum()  # only True values are counted
    total_pixel_count = void_mask.size
    void_ratio = void_pixel_count / total_pixel_count
    if void_ratio > allowed_threshold:
        return True
    return False


def classify_voids(
    void_mask: np.ndarray, boundary_geom: shapely.geometry.Polygon
) -> tuple:
    raise NotImplementedError


@timeit
def get_raster_properties(
    raster_filename: str,
    boundary: gpd.GeoDataFrame,
    env_options: Optional[dict[str, Any]] = None,
) -> RasterProperties:
    if env_options is None:
        env_options = dict()

    raster_filename = Path(raster_filename)
    hashval = uuid.uuid4()
    filesize = raster_filename.stat().st_size
    info = cog_info(raster_filename)
    nodata = info.Profile.Nodata
    bands = info.Profile.Bands
    compression = info.Compression
    cellsize = info.GEO.Resolution

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
            void_arr = get_void_mask(src, boundary_geom)
            has_voids = contains_voids(void_arr, strict=True)
            nodata_internal = 1 if has_voids else 0
            nodata_boundary = 1 if has_voids else 0

    return RasterProperties(
        filename=str(raster_filename),
        filesize=filesize,
        hash=hashval,
        nodata=nodata,
        bands=bands,
        compression=compression,
        epsg=epsg,
        cellsize=cellsize,
        minval=minval,
        maxval=maxval,
        has_voids=has_voids,
        nodata_internal=nodata_internal,
        nodata_boundary=nodata_boundary,
    )


@timeit
def main():
    MAX_RETRY_FAILED_FUTURES = 3  # maximum number of retries for failed futures
    MAX_WORKERS = os.cpu_count()  # numbers of workers to use for parallel processing, default is number of CPU cores.
    DEFAULT_ENV_OPTIONS = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

    data_root = Path("./data/17811_US_LP_2024_Enbridge_test")
    lidar_products = ["DSM"]
    boundary_file = next(data_root.rglob("*Boundary*.geojson"))
    boundary_df = gpd.read_file(boundary_file)
    flist = list_files(data_root, extensions=["tif", "tiff"])
    future_to_kwargs = dict()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for f in flist:
            kwargs = dict(raster_filename=str(f), boundary=boundary_df, env_options=DEFAULT_ENV_OPTIONS)
            future_to_kwargs[executor.submit(get_raster_properties, **kwargs)] = kwargs

        results = []
        failed_futures = []
        logger.info("processing %s futures", len(future_to_kwargs))
        for future in as_completed(future_to_kwargs.keys()):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("%s raised an exception: %s", future, e)
                failed_futures.append(future)
        
        # retry failed futures up to N times
        logger.info("retrying %s failed futures", len(failed_futures))
        retry_count = 0
        while retry_count < MAX_RETRY_FAILED_FUTURES and failed_futures:
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
