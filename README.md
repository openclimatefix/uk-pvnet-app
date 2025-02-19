# pvnet_app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-12-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

Internal OCF application to run [PVNet](https://github.com/openclimatefix/PVNet) models and (optionally) [PVNet summation](https://github.com/openclimatefix/PVNet-summation) models for the UK in a live environment. This involves accessing weather data stored in AWS S3 buckets, processing and loading this data using either our `ocf-data-sampler` or `ocf_datapipes` libraries, pulling pre-trained models from HuggingFace and then producing solar PV power forecasts for the UK by feeding the processed weather data into the model. 

The app supports multiple model versions being deployed to live environments and these can be ran with specific configurations which are set via environment variables.

## Environment Variables

The following environment variables are used in the app:

### Required Environment Variables

- `DB_URL`: The URL for the database connection.
- `NWP_UKV_ZARR_PATH`: The path to the UKV NWP data in Zarr format.
- `NWP_ECMWF_ZARR_PATH`: The path to the ECMWF NWP data in Zarr format.
- `SATELLITE_ZARR_PATH`: The path to the satellite data in Zarr format.

### Optional Environment Variables

- `PVNET_V2_VERSION`: The version of the PVNet V2 model to use. Default is a version above.
- `USE_ADJUSTER`: Option to use adjuster. Defaults to true.
- `SAVE_GSP_SUM`: Option to save GSP sum for PVNet V2. Defaults to false.
- `RUN_EXTRA_MODELS`: Option to run extra models. Defaults to false.
- `DAY_AHEAD_MODEL`: Option to use day ahead model. Defaults to false.
- `SENTRY_DSN`: Optional link to Sentry.
- `ENVIRONMENT`: The environment this is running in. Defaults to local.
- `USE_ECMWF_ONLY`: Option to use ECMWF only model. Defaults to false.
- `USE_OCF_DATA_SAMPLER`: Option to use OCF data sampler. Defaults to true.

### Examples

Here are some examples of how to set these environment variables:

```sh
export DB_URL="postgresql://user:password@localhost:5432/dbname"
export NWP_UKV_ZARR_PATH="s3://bucket/path/to/ukv.zarr"
export NWP_ECMWF_ZARR_PATH="s3://bucket/path/to/ecmwf.zarr"
export SATELLITE_ZARR_PATH="s3://bucket/path/to/satellite.zarr"
export PVNET_V2_VERSION="v2.0.0"
export USE_ADJUSTER="true"
export SAVE_GSP_SUM="false"
export RUN_EXTRA_MODELS="false"
export DAY_AHEAD_MODEL="false"
export SENTRY_DSN="https://examplePublicKey@o0.ingest.sentry.io/0"
export ENVIRONMENT="production"
export USE_ECMWF_ONLY="false"
export USE_OCF_DATA_SAMPLER="true"
```

## Validation Checks

We run a number of different validation checks on the data and the forecasts that are made. 
These are in place to ensure quality forecasts are made and saved to the database.

Before feeding data into the model(s) we check whether the data avilable is compatible with the 
data that the model expects.

### Satellite data

We check:
- Whether 5 minute and/or 15 minute satellite data is available
- We check if there are any NaNs in the satellite data, if there are, an error is raised
- We check if there are more that 10% zeros in the satellite data, if there are, an error is raised
- We check whether there are any missing timestamps in the satellite data. We linearly interpolate
any gaps less that 15 minutes.
- We check whether the exact timestamps that the model expects are all available after infilling

### NWP data

We check:
- Whether the exact timestamps that the model expects from each NWP are available


### ML batch checks

Just before the batch data goes into the ML models, we check that 
- All the NWP are not zeros. We raise an error if, for any nwp provider, all the NWP data is zero. 
- TODO: https://github.com/openclimatefix/PVNet/issues/324

### Forecast checks

After the ML models have run, we check the following
- The forecast is not above 110% of the national capacity. An error is raised if any forecast value is above 110% of the national capacity.
- The forecast is not above 100 GW, any forecast value above 30 GW we get a warning but any forecast value above 100 GW we raise an error. 
- If the forecast goes up and then down more than 500 MW we raise an error. A warning is made for 250 MW. This stops zig-zag forecasts. 
- TODO: Check positive values in day: https://github.com/openclimatefix/uk-pvnet-app/issues/200

## Development

### Running tests locally

To be able to run the tests locally it is recommended to use conda & pip and follow the steps from the Install requirements section onwards in the [Dockerfile](Dockerfile) or the install steps in the [conda-pytest.yaml](.github/workflows/conda-pytest.yaml) file and run tests the usual way via `python -m pytest`. Note if using certain macs you may need to install python >= 3.11 to get this to work.

### Running the app locally

It is possbile to run the app locally by setting the required environment variables listed at the top of the [app](pvnet_app/app.py), these should point to the relevant data sources and DBs for the environment you want to run the app in. You will need to make sure you have opened a connection to the DB, as well as authenticating against any cloud providers where data may be stored (e.g if using AWS S3 then can do this via the AWS CLI command `aws configure`), a simple [notebook](scripts/run_app_local_example.ipynb) has been created as an example.  


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DubraskaS"><img src="https://avatars.githubusercontent.com/u/87884444?v=4?s=100" width="100px;" alt="Dubraska Solórzano"/><br /><sub><b>Dubraska Solórzano</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=DubraskaS" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zakwatts"><img src="https://avatars.githubusercontent.com/u/47150349?v=4?s=100" width="100px;" alt="Megawattz"/><br /><sub><b>Megawattz</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=zakwatts" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DivyamAgg24"><img src="https://avatars.githubusercontent.com/u/142659327?v=4?s=100" width="100px;" alt="DivyamAgg24"/><br /><sub><b>DivyamAgg24</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=DivyamAgg24" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://github.com/aryanbhosale"><img src="https://avatars.githubusercontent.com/u/36108149?v=4?s=100" width="100px;" alt="Aryan Bhosale"/><br /><sub><b>Aryan Bhosale</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=aryanbhosale" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/felix-e-h-p"><img src="https://avatars.githubusercontent.com/u/137530077?v=4?s=100" width="100px;" alt="Felix"/><br /><sub><b>Felix</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=felix-e-h-p" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ADIMANV"><img src="https://avatars.githubusercontent.com/u/68527614?v=4?s=100" width="100px;" alt="Aditya Sawant"/><br /><sub><b>Aditya Sawant</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=ADIMANV" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sukh-P"><img src="https://avatars.githubusercontent.com/u/42407101?v=4?s=100" width="100px;" alt="Sukhil Patel"/><br /><sub><b>Sukhil Patel</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=Sukh-P" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alirashidAR"><img src="https://avatars.githubusercontent.com/u/110668489?v=4?s=100" width="100px;" alt="Ali Rashid"/><br /><sub><b>Ali Rashid</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=alirashidAR" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mahmoud-40"><img src="https://avatars.githubusercontent.com/u/116794637?v=4?s=100" width="100px;" alt="Mahmoud Abdulmawlaa"/><br /><sub><b>Mahmoud Abdulmawlaa</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=mahmoud-40" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/meghana-0211"><img src="https://avatars.githubusercontent.com/u/136890863?v=4?s=100" width="100px;" alt="Meghana Sancheti"/><br /><sub><b>Meghana Sancheti</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=meghana-0211" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
