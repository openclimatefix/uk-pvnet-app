"""Runtime configuration loaded from environment."""

from typing import Literal

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Runtime configuration loaded from environment.

    Input data paths

        - NWP_UKV_ZARR_PATH: Path to the UKV NWP data in zarr format
        - NWP_ECMWF_ZARR_PATH: Path to the ECMWF NWP data in zarr format
        - CLOUDCASTING_ZARR_PATH: Path to the Cloudcasting data in zarr format
        - SATELLITE_ICECHUNK_PATH_5: Path on s3 to the 5-minute satellite icechunk data
        - SATELLITE_ICECHUNK_PATH_15: Path on s3 to the 15-minute satellite icechunk data
        - SATELLITE_S3_REGION: The AWS region for the satellite data S3 bucket

    Options for choosing which models to run and how to validate forecasts

        - RUN_CRITICAL_MODELS_ONLY: Option to run critical models only
        - FILTER_BAD_FORECASTS: Option to filter out bad forecasts. If set to true and the forecast
          fails the validation checks, it will not be saved. If set to false, the forecast will be
          saved even if it fails validation.
        - RAISE_MODEL_FAILURE: Option to raise an exception if a model fails to run. If set to
          "any" it will raise an exception if any model fails. If set to "critical" it will raise
          an exception if any critical model fails. If not set, it will not raise an exception.

        - FORECAST_VALIDATE_ZIG_ZAG_WARNING_THRESHOLD: threshold for forecast zig-zag warning
        - FORECAST_VALIDATE_ZIG_ZAG_ERROR_THRESHOLD: threshold for forecast zig-zag error
        - FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT: when the solar elevation is above this,
          we expect positive forecast values

    Other settings

        - LOG_LEVEL: logging level for the application
        - SENTRY_DSN: link to sentry
        - ENVIRONMENT: the environment this is running in.
        - DATA_PLATFORM_HOST: Hostname of the data platform gRPC server.
        - DATA_PLATFORM_PORT: Port of the data platform gRPC server.
        - HUGGINGFACE_TOKEN: Huggingface token, required if any of the models being run are in
          private repositories.
        - SAVE_BATCHES_DIR: If set, the batches will be saved to this path.
        - SCRATCH_DIR: If set, the scratch directory will be used for temporary files. Else, the
          system temp directory will be used.
    """

    # Input data paths
    nwp_ukv_zarr_path: str | None = None
    nwp_ecmwf_zarr_path: str | None = None
    cloudcasting_zarr_path: str | None = None
    satellite_icechunk_path_5: str | None = None
    satellite_icechunk_path_15: str | None = None
    satellite_s3_region: str | None = None

    # Options for choosing which models to run and how to validate forecasts
    run_critical_models_only: bool = False
    filter_bad_forecasts: bool = False
    raise_model_failure: Literal["any", "critical"] | None = None

    forecast_validate_zig_zag_warning_threshold: float = 250.0
    forecast_validate_zig_zag_error_threshold: float = 500.0
    forecast_validate_sun_elevation_lower_limit: float = 10.0

    # Other settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    sentry_dsn: str | None = None
    environment: str = "local"
    data_platform_host: str
    data_platform_port: int = 50051
    huggingface_token: str | None = None
    save_batches_dir: str | None = None
    scratch_dir: str | None = None
