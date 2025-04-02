from ocf_data_sampler.numpy_sample.common_types import NumpyBatch

def check_nwp_sources_not_all_zero(batch: NumpyBatch) -> None:
    """Check that the NWP data in the batch is not all zeros
    
    Args:
        batch: The batch to check

    Raises:
        ValueError: If the NWP data is all zeros
    """

    for nwp_source in batch["nwp"].keys():
        if (batch["nwp"][nwp_source]["nwp"] == 0).all():
            raise ValueError(
                f"NWP data for {nwp_source} is all zeros. This is probably an error. "
                "To fix this check raw NWP data, and the nwp-consumer"
            )


def check_batch(batch: NumpyBatch) -> None:
    """Check the batch for any defined issues
    
    Args:
        batch: The batch to check
    
    Raises:
        ValueError: If the batch is invalid
    """
    check_nwp_sources_not_all_zero(batch)
