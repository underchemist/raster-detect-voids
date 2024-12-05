import hashlib
import logging
import os
import zlib
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
from rio_cogeo import cog_info
from tqdm import tqdm

from report import RasterProperties, dump_exceptions, write_failed, write_report
from timing import timeit
from walk import list_files

# ignore logging from these modules, remove to see debug messages from all modules
LOGGING_MODULE_IGNORE = ["rasterio", "pandas", "geopandas"]
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
for module in LOGGING_MODULE_IGNORE:
    logging.getLogger(module).setLevel(logging.WARNING)


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


def compute_void_ratio(void_mask: np.ndarray, raster_mask: np.ndarray) -> float:
    void_pixel_count = void_mask.sum()  # only True values are counted
    total_pixel_count = raster_mask.astype(
        "bool"
    ).sum()  # perhaps should be the count of masked raster pixels?
    void_ratio = void_pixel_count / total_pixel_count

    return void_ratio


def contains_voids(
    void_mask: np.ndarray,
    raster_mask: np.ndarray,
    void_threshold: float = 0.01,
    strict: bool = False,
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

    void_ratio = compute_void_ratio(void_mask, raster_mask)
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
    void_threshold: float = 0.01,  # 1% void pixel threshold
    classify_voids: bool = False,
    geom_buffer_size: float = 1.0,
    env_options: Optional[dict[str, Any]] = None,
    rasterize_options: Optional[dict[str, Any]] = None,
) -> RasterProperties:
    """
    Compute raster properties for a given raster file and boundary geometry.

    Args:
        raster_filename (str): Filename for raster dataset.
        boundary (gpd.GeoDataFrame): A vector dataset describing the data
            collection Boundary, returned by geopandas.read_file.
        strict (bool, optional): If True, voids will be detected in raster
            if there is >=1 nodata pixel contained in the boundary geometry. Defaults to False.
        void_threshold (float, optional): The ratio of detected void pixels
            to the total number of pixels in a raster dataset. Raster datasets
            that have a void pixel ratio above this value will be determined as
            having "voids", less than will be determined to not have "voids". Defaults to 0.01.
        classify_voids (bool, optional): If the raster dataset has voids,
            attempt to classify the voids as either nodata_internal or nodata_boundary. Defaults to False.
        geom_buffer_size (float, optional): Buffer the boundary exterior ring
            geometry in pixel units. Only used to classify_voids. Defaults to 1.
        env_options (Optional[dict[str, Any]], optional): Options to pass to rasterio.Env.
            See DEFAULT_ENV_OPTIONS. Defaults to None.
        rasterize_options (Optional[dict[str, Any]], optional): Options to pass to
            rasterio.features.Rasterize. Defaults to None.

    Returns:
        RasterProperties: Mapping of raster properties
    """
    if env_options is None:
        env_options = dict()
    if rasterize_options is None:
        rasterize_options = dict()

    raster_filename = Path(raster_filename)
    hashval = md5_checksum(raster_filename)  # use your preferred hashing algorithm
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
            # can pass approx=True to compute stats from overviews,
            # I don't expect this to significatly impact performance however
            stats = src.stats(indexes=1)[0]
            minval = stats.min
            maxval = stats.max

            # void computations
            raster_mask = src.read_masks(1)
            void_arr = get_void_mask(src, boundary_geom, **rasterize_options)
            has_voids = contains_voids(
                void_arr,
                raster_mask,
                void_threshold=void_threshold,
                strict=strict,
            )
            void_ratio = compute_void_ratio(void_arr, raster_mask)
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


def process_rasters(
    flist: list[Path],
    boundary_df: gpd.GeoDataFrame,
    max_workers: int,
    max_retries: int,
    env_options: dict[str, Any],
    void_options: dict[str, Any],
) -> tuple[list[RasterProperties], list[Future], dict[Future, Any]]:
    """
    Wrapper function to process raster files in parallel using ProcessPoolExecutor.

    Args:
        flist (list[Path]): List of raster file paths.
        boundary_df (gpd.GeoDataFrame): Vector dataset describing the data collection Boundary.
        max_workers (int): Number of processes to use in ProcessPoolExecutor.
        max_retries (int): Number of retries for failed get_raster_properties.
        env_options (dict[str, Any]): Options to pass to rasterio.Env
        void_options (dict[str, Any]): keyword arguments to pass to get_raster_properties

    Returns:
        tuple[list[RasterProperties], list[Future], dict[Future, Any]]
    """
    future_to_kwargs = dict()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        logger.info("launching ProcessPoolExecutor...")
        for f in flist:
            kwargs = dict(
                raster_filename=str(f),
                boundary=boundary_df,
                **void_options,
                env_options=env_options,
            )
            future_to_kwargs[executor.submit(get_raster_properties, **kwargs)] = kwargs

        results = []
        failed_futures = []
        logger.info("processing %s futures", len(future_to_kwargs))
        for future in tqdm(
            as_completed(future_to_kwargs.keys()), total=len(future_to_kwargs.keys())
        ):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("%s raised an exception: %s", future, e)
                failed_futures.append(future)

        # retry failed futures up to N times
        retry_count = 0
        while retry_count < max_retries and failed_futures:
            logger.info(
                "retrying %s failed futures, attempt %s",
                len(failed_futures),
                retry_count,
            )
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
    return (results, failed_futures, future_to_kwargs)


@timeit
def main():
    MAX_RETRY_FAILED_FUTURES = 3  # maximum number of retries for failed futures
    MAX_WORKERS = (
        os.cpu_count()
        # 1  # numbers of workers to use for parallel processing, default is number of CPU cores.
    )
    OUTPUT_DIR = Path(".")  # output directory for csv files
    DEFAULT_ENV_OPTIONS = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # avoid reading all files in directory looking for external .ovr files
        "GDAL_PAM_ENABLED": "NO",  # avoid writing *.aux.xml files which might cache incorrect stats
        # default is 5% of physical memory. Since rasterize is memory intensive increasing this value can help.
        # See notes in https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.rasterize
        # Value must be integer number of bytes.
        # Example showcases 25% of 8GB converted to bytes
        # "GDAL_CACHEMAX": int(0.25 * (8 * 1e9))
    }
    MAX_ITEMS = 100  # short-circuit for testing

    data_root = Path("./data/17811_US_LP_2024_Enbridge_test")
    boundary_file = next(data_root.rglob("*Boundary*.geojson"))
    boundary_df = gpd.read_file(boundary_file)
    flist = list_files(data_root, extensions=["tif", "tiff"])
    if MAX_ITEMS:
        flist = flist[:MAX_ITEMS]

    # work
    void_options = dict(
        strict=False,
        void_threshold=0.01,
        classify_voids=True,
        geom_buffer_size=1,
    )
    results, failed_futures, future_to_kwargs = process_rasters(
        flist=flist,
        boundary_df=boundary_df,
        max_workers=MAX_WORKERS,
        max_retries=MAX_RETRY_FAILED_FUTURES,
        env_options=DEFAULT_ENV_OPTIONS,
        void_options=void_options,
    )

    # write results, failed, and exceptions
    write_report(
        OUTPUT_DIR.joinpath("report.csv"),
        results,
    )

    write_failed(OUTPUT_DIR.joinpath("failed.csv"), failed_futures, future_to_kwargs)
    dump_exceptions(OUTPUT_DIR.joinpath("exceptions.txt"), failed_futures)


if __name__ == "__main__":
    main()
