# pvnet_app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-16-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![ease of contribution: medium](https://img.shields.io/badge/ease%20of%20contribution:%20medium-f4900c)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

Internal OCF application to run [PVNet](https://github.com/openclimatefix/PVNet) models and (optionally) [PVNet summation](https://github.com/openclimatefix/PVNet-summation) models for the UK in a live environment. This involves accessing weather data stored in AWS S3 buckets, processing and loading this data using either our `ocf-data-sampler`, pulling pre-trained models from HuggingFace and then producing solar PV power forecasts for the UK by feeding the processed weather data into the model. 

The app supports multiple model versions being deployed to live environments and these can be ran with specific configurations which are set via environment variables.

## Environment Variables

The app is configured at runtime with many environmental variables. See [settings.py](src/pvnet_app/settings.py).


## ML Models Used for PVNet

<!-- START model-config-table -->
| Model Name | Uses satellite | Uses UKV | Uses ECMWF | Uses cloudcasting | PVNet Hugging Face Link | PVNet Summation Hugging Face Link |
| ----|----|----|----|----|----|---- |
| pvnet_intra_allbells30 | yes | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/b88eae46dd40d8670de00b0c0e64d6363886aadf) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/b10af61ef303566dee6647f0b9eba07432d91957) |
| pvnet_intra_allbells0 | yes | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/0f6a00de0c6a12b36f6c050ba7d9916949803ad8) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/f5b6b7fab06d762e900f6caeb3ec6af07db3ec4c) |
| pvnet_da_ecmwf | - | - | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/bd376303f753dd869dd0fd194906451b91b2dcd1) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/2e301b599ab8a15ed02d09aceea43c7f7c6f4516) |
| pvnet_intra_sat30 | yes | - | - | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/14f2223681c9741163a3099493d97bcd1c4e0025) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/ac6d5dc380628f15ad02c53a2ddb8727e1e2907e) |
| pvnet_da_ukv | - | yes | - | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/b242aad4ed243efe4e701b9ae61136615795faa8) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/2263afba597d231c0699782b9e44c7208a751345) |
| pvnet_da_2nwp | - | yes | yes | - | [HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_region/tree/723603423d1b2b74fe49715a2d52fa7593a9451e) | [Summation HF Link](https://huggingface.co/openclimatefix-models/pvnet_uk_summation/tree/fe1c826c4141d8cefc5c5b60618ad1654a06f9b0) |

<!-- END model-config-table -->

## Validation Checks

We run a number of validation checks on the input data and on the forecasts that are
produced.

Before feeding data into the model(s) we check whether the available data is compatible
with what each model expects.

### Satellite data

We check:
- Whether 5-minute and/or 15-minute satellite data is available.
- If more than 5% of the satellite data in the required area is NaN. If so, the satellite data is 
  treated as completely missing.
- Whether there are any missing timestamps in the satellite data. Gaps of up to 15 minutes are 
  linearly interpolated.
- After interpolation, the most recent missing timestamps are extended forward with NaNs up to the 
  delay the model allows.
- Whether the exact timestamps the model expects are all available after infilling.

### NWP data

We check:
- If the NWP data contains any NaNs — if so, that NWP source is treated as missing.
- Whether the exact timestamps the model expects from each NWP source are available.

### ML batch checks

Just before the batch goes into the ML models, we check that:
- The NWP data for each provider is not entirely zeros. We raise an error if, for any
  NWP provider, all the NWP data is zero.

### Forecast checks

After the ML models have run, we check the national forecast only:
- It does not exceed 110% of the national capacity. The forecast fails validation if any
  value is above this.
- It does not exceed an absolute ceiling of 20 GW. The forecast fails validation if any
  value is above this.
- It does not zig-zag (go up, then down, then up). Swings above 500\* MW fail validation;
  swings above 250\* MW produce a warning. \*configurable - see setting.py
- Forecast values are positive when the sun is up (above the configured elevation).

## Development

### Running tests locally

To be able to run the tests locally it is recommended to use `conda` and `uv`. The easiest way to to create the required environment and run the pytest is to run `make test`. This will create a new environment in your working directory under `.venv`.

### Running the app locally

It is possbile to run the app locally by setting the required environment variables listed in [settings.py](pvnet_app/settings.py), these should point to the relevant data sources and the data platform instance for the environment you want to run the app in. You will need to make sure the data platform at `DATA_PLATFORM_HOST:DATA_PLATFORM_PORT` is reachable, as well as authenticating against any cloud providers where data may be stored (e.g if using AWS S3 then can do this via the AWS CLI command `aws configure`).


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
