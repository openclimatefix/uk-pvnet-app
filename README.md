# pvnet_app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-16-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![ease of contribution: medium](https://img.shields.io/badge/ease%20of%20contribution:%20medium-f4900c)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

Internal OCF application to run [PVNet](https://github.com/openclimatefix/PVNet) models and (optionally) [PVNet summation](https://github.com/openclimatefix/PVNet-summation) models for the UK in a live environment. This involves accessing weather data stored in AWS S3 buckets, processing and loading this data using either our `ocf-data-sampler`, pulling pre-trained models from HuggingFace and then producing solar PV power forecasts for the UK by feeding the processed weather data into the model. 

The app supports multiple model versions being deployed to live environments and these can be ran with specific configurations which are set via environment variables.

## Environment Variables

The following environment variables are used in the app:

### Required Environment Variables

- `DB_URL`: The URL for the database connection.
- `NWP_UKV_ZARR_PATH`: The path to the UKV NWP data in Zarr format.
- `NWP_ECMWF_ZARR_PATH`: The path to the ECMWF NWP data in Zarr format.
- `CLOUDCASTING_ZARR_PATH`: The path to the cloudcasting forecast data in Zarr format.
- `SATELLITE_ZARR_PATH`: The path to the satellite data in Zarr format.

### Optional Environment Variables

#### These control the data sources

- `SATELLITE_15_ZARR_PATH`: The path to the 15 minute satellite data in Zarr format. If 
this is not set then the `SATELLITE_ZARR_PATH` is used by `.zarr` is repalced with `_15.zarr`

#### These control the model(s) run

- `RUN_CRITICAL_MODELS_ONLY`: Option to run critical models only. Defaults to false.

#### These control the saved results

- `ALLOW_ADJUSTER`: Option to allow the adjuster to be used. If false this overwrites the adjuster 
  option in the model configs so it is not used. Defaults to true.
- `ALLOW_SAVE_GSP_SUM`: Option to allow model to save the GSP sum. If false this overwrites the
  model configs so saving of the GSP sum is not used. Defaults to false.

#### These extra variables control validation and logging

- `SENTRY_DSN`: Optional link to Sentry.
- `ENVIRONMENT`: The environment this is running in. Defaults to local.
- `FORECAST_VALIDATE_ZIG_ZAG_WARNING`: Threshold for warning on forecast zig-zag, defaults to 250MW.
- `FORECAST_VALIDATE_ZIG_ZAG_ERROR`: Threshold for error on forecast zig-zag, defaults to 500MW.
- `FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT`, when the solar elevation is above this,
  we expect positive forecast values. Defaults to 10 degrees.
- `FILTER_BAD_FORECASTS`: If set to true and the forecast fails validation checks, it will not be 
  saved. Defaults to false, where all forecasts are saved even if they fail the checks.
- `RAISE_MODEL_FAILURE`: Option to raise an exception if a model fails to run. If set to "any" it 
  will raise an exception if any model fails. If set to "critical" it will raise an exception if any
  critical model fails. If not set, it will not raise an exception.
- `DATA_PLATFORM_HOST`: The host address for the data platform, default is localhost.
- `DATA_PLATFORM_PORT`: The port for the data platform, default is 50051. 

### Examples

Here are some examples of how to set these environment variables:

```sh
export DB_URL="postgresql://user:password@localhost:5432/dbname"
export NWP_UKV_ZARR_PATH="s3://bucket/path/to/ukv.zarr"
export NWP_ECMWF_ZARR_PATH="s3://bucket/path/to/ecmwf.zarr"
export CLOUDCASTING_ZARR_PATH="s3://bucket/path/to/cloudcasting.zarr"
export SATELLITE_ZARR_PATH="s3://bucket/path/to/satellite.zarr"
export ALLOW_ADJUSTER="true"
export ALLOW_SAVE_GSP_SUM="false"
export RUN_CRITICAL_MODELS_ONLY="true"
export SENTRY_DSN="https://examplePublicKey@o0.ingest.sentry.io/0"
export ENVIRONMENT="production"
```

## ML Models Used for PVNet

<!-- START model-config-table -->
| Model Name | Uses satellite | Uses UKV | Uses ECMWF | Uses cloudcasting | PVNet Hugging Face Link | PVNet Summation Hugging Face Link |
| ----|----|----|----|----|----|---- |
| pvnet_intra_allbells30 | yes | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/1ff82a88e890441e4497a59b7d5e6d6836769ff7) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/71acc4c990cf0d0cd9f01dbd1375ec4fa7e1e750) |
| pvnet_intra_allbells0 | yes | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/cc486ad6af6628a5d7c604c5730be748b243bc15) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/474a138e4d8b188e07ef0d34520c896281132bf0) |
| pvnet_da_ecmwf | - | - | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/8cd379ad7769aecd532017976e9658d3fea76e02) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/62d13dbe1458003995f671656e4db5c7b36dc912) |
| pvnet_intra_sat30 | yes | - | - | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/5fdc7ec353668d4758455430eea8e10d4c638f1e) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/eb779e7aaa9e0fb5813889a8f9a02d6c518b432a) |
| pvnet_da_ukv | - | yes | - | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/4831fc755042b7b311a72b206d62032bfd606c9f) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/aff32fb46f8ecbaf6d5b2664c444a27f6bb160f4) |
| pvnet_da_2nwp | - | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/03924afeef8c83fb8daf45d555f51f7647c91e6d) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/0b597e230753c7f0e24fb8906cce23b500f96627) |

<!-- END model-config-table -->

## Validation Checks

We run a number of different validation checks on the data and the forecasts that are made. 
These are in place to ensure quality forecasts are made and saved to the database.

Before feeding data into the model(s) we check whether the data avilable is compatible with the 
data that the model expects.

### Satellite data

We check:
- Whether 5 minute and/or 15 minute satellite data is available
- If more than 5% of satellite data is NaN - if so the satellite data is treated as missing
- If more that 10% of satellite data is zero - if so the satellite data is treated as missing
- Whether there are any missing timestamps in the satellite data. We linearly interpolate
any gaps less that 15 minutes.
- Whether the exact timestamps that the model expects are all available after infilling and checks

### NWP data

We check:
- If the NWP data contains any NaNs - if so that NWP source is treated as missing
- Whether the exact timestamps that the model expects from each NWP are available

### ML batch checks

Just before the batch data goes into the ML models, we check that 
- All the NWP are not zeros. We raise an error if, for any nwp provider, all the NWP data is zero. 
- TODO: https://github.com/openclimatefix/PVNet/issues/324

### Forecast checks

After the ML models have run, we check the following
- The forecast is not above 110% of the national capacity. An error is raised if any forecast value is above 110% of the national capacity.
- The forecast is not above 100 GW, any forecast value above 30 GW we get a warning but any forecast value above 100 GW we raise an error. 
- If the forecast goes up, then down, then up, more than 500 MW we raise an error. A warning is made for 250 MW. This stops zig-zag forecasts. 
- Check positive values in day. If the sun is up, we expect positive values. 

## Development

### Running tests locally

To be able to run the tests locally it is recommended to use `conda` and `uv`. The easiest way to to create the required environment and run the pytest is to run `make test`. This will create a new environment in your working directory under `.venv`.

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mukiralad"><img src="https://avatars.githubusercontent.com/u/67241568?v=4?s=100" width="100px;" alt="Dheeraj Mukirala"/><br /><sub><b>Dheeraj Mukirala</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=mukiralad" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/utsav-pal"><img src="https://avatars.githubusercontent.com/u/159793156?v=4?s=100" width="100px;" alt="utsav-pal"/><br /><sub><b>utsav-pal</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=utsav-pal" title="Code">💻</a> <a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=utsav-pal" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/r-ram-kumar-71bb75144/"><img src="https://avatars.githubusercontent.com/u/114728749?v=4?s=100" width="100px;" alt="RAM KUMAR R"/><br /><sub><b>RAM KUMAR R</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=ram-from-tvl" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rdsingh120"><img src="https://avatars.githubusercontent.com/u/82333889?v=4?s=100" width="100px;" alt="Ripudaman Singh"/><br /><sub><b>Ripudaman Singh</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=rdsingh120" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
