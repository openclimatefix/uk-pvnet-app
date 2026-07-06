"""Functions to validate data batches."""

from ocf_data_sampler.numpy_sample.common_types import NumpyBatch
from pvnet.utils import validate_batch_against_config
from pvnet.models.base_model import BaseModel as PVNetBaseModel


def check_nwp_sources_not_all_zero(batch: NumpyBatch) -> None:
    """Check that the NWP data in the batch is not all zeros.

    Args:
        batch: The batch to check

    Raises:
        ValueError: If the NWP data is all zeros
    """
    if "nwp" in batch:
        for nwp_source in batch["nwp"]:
            if (batch["nwp"][nwp_source]["nwp"] == 0).all():
                raise ValueError(
                    f"NWP data for {nwp_source} is all zeros. This is probably an error. "
                    "To fix this check raw NWP data, and the nwp-consumer",
                )


def _check_batch(batch: NumpyBatch, model: PVNetBaseModel) -> None:
    """Check the batch for any defined issues.

    Args:
        batch: The batch to check

    Raises:
        ValueError: If the batch is invalid
    """
    check_nwp_sources_not_all_zero(batch)
    validate_batch_against_config(
        batch=batch,
        model=model,
    )


def get_batch_validation_error(
    batch: NumpyBatch,
    model: PVNetBaseModel,
) -> str | None:
    """Return the validation error message, or None if the batch is valid."""
    try:
        _check_batch(batch, model=model)
    except ValueError as exc:
        return str(exc)

    return None