# pvnet_app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-15-orange.svg?style=flat-square)](#contributors-)
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

#### These control the model(s) run

- `RUN_CRITICAL_MODELS_ONLY`: Option to run critical models only. Defaults to false.
- `DAY_AHEAD_MODEL`: Option to use day ahead model. Defaults to false.

#### These control the saved results

- `ALLOW_ADJUSTER`: Option to allow the adjuster to be used. If false this overwrites the adjuster 
  option in the model configs so it is not used. Defaults to true.
- `ALLOW_SAVE_GSP_SUM`: Option to allow model to save the GSP sum. If false this overwrites the
  model configs so saving of the GSP sum is not used. Defaults to false.

#### These extra varaibles control validation and logging

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
export DAY_AHEAD_MODEL="false"
export SENTRY_DSN="https://examplePublicKey@o0.ingest.sentry.io/0"
export ENVIRONMENT="production"
```

## ML Models Used for PVNet

| Model Name                | Satellite | NWP UKV | NWP ECMWF | Legacy | PVNet Hugging Face Link | PVNet Summation Hugging Face Link |
|---------------------------|-----------|---------|-----------|--------|--------------------------|----------------------------------|
| **pvnet_v2**              | yes       | yes     | yes       | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/f3135c47eb0f21320dbd8c590bdd03dfafc39bca) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/01ca2b6e37a71deb446bb90471b44a1851d3e43f) |
| **pvnet_ecmwf**           | no        | no      | yes       | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/20b882bd4ceaee190a1c994d861f8e5d553ea843) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/b40867abbc2e5163c9a665daf511cbf372cc5ac9) |
| **pvnet-sat0**            | yes       | no      | no        | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/d81a9cf8adca49739ea6a3d031e36510f44744a1) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/7057e8c2baa065aa4024dd6b6381b71ac4879c87) |
| **pvnet-sat0-only**       | yes       | no      | no        | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/158f9aeb006dddc10ef67612a91e7175a87b8dd0) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/c5371880120503646327dc2df2da2698de82982e) |
| **pvnet-ukv-only**        | no        | yes     | no        | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/4009df82e63e30546e2000728bff34b9c0520617) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/1789cd9bdaded3896587efd54d3e9a257762fb63) |
| **pvnet_day_ahead**       | no        | yes     | yes       | no     | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region_day_ahead/tree/263741ebb6b71559d113d799c9a579a973cc24ba) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_summation_uk_national_day_ahead/tree/7a2f26b94ac261160358b224944ef32998bd60ce) |
| **Legacy pvnet_v2**       | yes       | yes     | yes       | yes    | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/aa73cdafd1db8df3c8b7f5ecfdb160989e7639ac) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/a7fd71727f4cb2b933992b2108638985e24fa5a3) |
| **Legacy pvnet_ecmwf**    | no        | no      | yes       | yes    | [HF Link](https://huggingface.co/openclimatefix/pvnet_uk_region/tree/c14f7427d9854d63430aa936ce45f55d3818d033) | [Summation HF Link](https://huggingface.co/openclimatefix/pvnet_v2_summation/tree/4fe6b1441b6dd549292c201ed85eee156ecc220c) |


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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mukiralad"><img src="https://avatars.githubusercontent.com/u/67241568?v=4?s=100" width="100px;" alt="Dheeraj Mukirala"/><br /><sub><b>Dheeraj Mukirala</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=mukiralad" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/utsav-pal"><img src="https://avatars.githubusercontent.com/u/159793156?v=4?s=100" width="100px;" alt="utsav-pal"/><br /><sub><b>utsav-pal</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=utsav-pal" title="Code">💻</a> <a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=utsav-pal" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/r-ram-kumar-71bb75144/"><img src="https://avatars.githubusercontent.com/u/114728749?v=4?s=100" width="100px;" alt="RAM KUMAR R"/><br /><sub><b>RAM KUMAR R</b></sub></a><br /><a href="https://github.com/openclimatefix/uk-pvnet-app/commits?author=ram-from-tvl" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
