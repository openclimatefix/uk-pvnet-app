# pvnet_app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Application to run [PVNet](https://github.com/openclimatefix/PVNet) and (optionally) [PVNet summation](https://github.com/openclimatefix/PVNet-summation) in live environments e.g Dev/Prod.

The app supports multiple model versions being deployed to live environments, the models are pulled from HuggingFace and ran with specific configurations which are set via environment variables. 

## Running tests locally

To be able to run the tests locally it is recommended to use conda & pip and follow the steps from the Install requirements section onwards in the [Dockerfile](Dockerfile) or the install steps in the [conda-pytest.yaml](.github/workflows/conda-pytest.yaml) file and run tests the usual way via `python -m pytest`. Note if using certain macs you may need to install python 3.11 to get this to work.

## Running the app locally

It is possbile to run the app locally by setting the required environment variables listed at the top of the [app](pvnet_app/app.py), these should point to the relevant data sources and DBs for the environment you want to run the app in. You will need to make sure you have opened a connection to the DB, as well as authenticating against any cloud providers where data may be stored (e.g if using AWS S3 then can do this via the AWS CLI command `aws configure`), a simple [notebook](scripts/run_app_local_example.ipynb) has been created as an example.  


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DubraskaS"><img src="https://avatars.githubusercontent.com/u/87884444?v=4?s=100" width="100px;" alt="Dubraska Solórzano"/><br /><sub><b>Dubraska Solórzano</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet_app/commits?author=DubraskaS" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet_app/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zakwatts"><img src="https://avatars.githubusercontent.com/u/47150349?v=4?s=100" width="100px;" alt="Megawattz"/><br /><sub><b>Megawattz</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet_app/commits?author=zakwatts" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet_app/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DivyamAgg24"><img src="https://avatars.githubusercontent.com/u/142659327?v=4?s=100" width="100px;" alt="DivyamAgg24"/><br /><sub><b>DivyamAgg24</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet_app/commits?author=DivyamAgg24" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!