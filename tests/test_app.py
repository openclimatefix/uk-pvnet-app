import tempfile
import zarr
import os
import logging

from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)


def _test_app(db_session, nwp_data, sat_5_data, sat_15_data, gsp_yields_and_systems, me_latest):
    # Environment variable DB_URL is set in engine_url, which is called by db_session
    # set NWP_ZARR_PATH
    # save nwp_data to temporary file, and set NWP_ZARR_PATH
    # SATELLITE_ZARR_PATH
    # save sat_data to temporary file, and set SATELLITE_ZARR_PATH
    # GSP data

    with tempfile.TemporaryDirectory() as tmpdirname:
        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables 
        temp_nwp_path = f"{tmpdirname}/nwp.zarr"
        os.environ["NWP_ZARR_PATH"] = temp_nwp_path
        nwp_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        temp_sat_path = f"{tmpdirname}/sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        store = zarr.storage.ZipStore(temp_sat_path, mode="x")
        sat_5_data.to_zarr(store)
        store.close()

        # Maybe save the 15-minute data too
        if sat_15_data is not None:
            temp_sat_path = os.environ["SATELLITE_ZARR_PATH"].replace("sat.zarr", "sat_15.zarr")
            store = zarr.storage.ZipStore(temp_sat_path, mode="x")
            sat_15_data.to_zarr(store)
            store.close()
        
        # Set model version
        os.environ["SAVE_GSP_SUM"] = "True"

        # Run prediction
        # This import needs to come after the environ vars have been set
        from pvnet_app.app import app
        app(gsp_ids=list(range(1, 318)), num_workers=2)
        
    # Check forecasts have been made
    # (317 GSPs + 1 National + GSP-sum) = 319 forecasts
    # Doubled for historic and forecast
    forecasts = db_session.query(ForecastSQL).all()
    assert len(forecasts) == 319 * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 318 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == 319 * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 319 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 319 * 16
    
    # Clean up
    db_session.query(ForecastSQL).delete()
    db_session.commit()


def test_app_5(db_session, nwp_data, sat_5_data, gsp_yields_and_systems, me_latest):
    
    _test_app(
        db_session=db_session, 
        nwp_data=nwp_data, 
        sat_5_data=sat_5_data, 
        sat_15_data=None,
        gsp_yields_and_systems=gsp_yields_and_systems, 
        me_latest=me_latest
    )
    

def test_app_15(
    db_session, nwp_data, sat_5_data_delayed, sat_15_data, gsp_yields_and_systems, me_latest
):
    _test_app(
        db_session=db_session, 
        nwp_data=nwp_data, 
        sat_5_data=sat_5_data_delayed, 
        sat_15_data=sat_15_data,
        gsp_yields_and_systems=gsp_yields_and_systems, 
        me_latest=me_latest
    )
