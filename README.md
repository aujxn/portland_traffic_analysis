# Portland Traffic Analysis

This repo contains code and analysis done in collaboration between PSU math/stats and ODOT for a graduate statistics consulting course.
The data analyzed is publicly available ATR data which counts vehicles over time at different locations on highways.
The primary goal of the project is to understand changes in the transportation system by analyzing traffic trends before and after the pandemic.

## Usage

### Initial Setup

Cloning the project, creating a virtual environment, and installing only has to be done once per computing system. The steps to do so are:

1. Clone project, create python environment, activate, and update `pip`.
    ```sh
    git clone https://github.com/aujxn/portland_traffic_analysis.git
    cd portland_traffic_analysis
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ```
2. Install the analysis tooling to the environment. By default, the optional `neural_network` module is disabled because it requires `torch` which is quite a large dependency.

    From the project root, install without the `neural_network` module by:
    ```sh
    pip install -e traffic_data_analysis
    ```
    or with the `neural_network` module by:
    ```sh
    pip install -e traffic_data_analysis[nn]
    ```
    Then download the datasets into the project by running:
    ```sh 
    fetch_data
    ```
3. Then you should be able to run various analyses. Some examples used to generate figures for the presentation and project report can be found in the `traffic_data_analysis/examples` folder. One such example could be run with:
    ```sh 
    python3 examples/smooth_quantiles.py
    ```

### Each Development Session

Although the virtual environment and package only have to be installed once, the environment must be activated each session with
```sh 
source venv/bin/activate
```
This could be automated with an environment manager like `conda` or migrating this package to use `poetry` but requires more tooling and additional complexity.

### Web Application

Included in this project is source for a web application built with the `dash` framework from `plotly`. I'm currently hosting this app at [traffic dashboard](https://traffic.aujxn.dev/dashboard). 

To host your own version or run the application locally the container configuration files `compose.yaml` and `Containerfile` provide a simple way to deploy the application using a container. I suggest the open source container runtime `podman`, but `docker` is the well established proprietary option. Hosting the app would additionally require setting up a reverse-proxy and SSL certificates, but the containerized application can be built and run with:
```sh 
podman-compose build
podman-compose up -d 
```

For hosting locally without containers, activate environment and install the web-app if you haven't already:
```sh
source venv/bin/activate
pip install -e traffic_webapp
```
and run the local debug web-server with:
```sh 
python3 traffic_webapp/local.py
```
and then navigate to <localhost:8096> in a browser window.

## Results

Analysis can be found in the `reports` folder [here](reports/README.md).

## License

See [license](LICENSE).
