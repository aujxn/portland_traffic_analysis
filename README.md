# Portland Traffic Analysis

This repo contains code and analysis done in collaboration between PSU and ODOT for a graduate statistic consulting course.
The data analyzed is publicly available ATR data which counts vehicles over time at different locations on highways.
The primary goal of the project is to understand changes in the transportation system by analyzing traffic trends before and after the pandemic.

## Usage

Clone the repo, copy the data files, create a virtual python environment, install dependencies, and run the code.

ODOT has provided us with 4 data files which I have renamed to `ATR10_18-24.pq`,  `ATR26004_2018-2024.csv`,  `ATR26024_2018-2024.csv`, and `ATR_metadata.csv` which are not committed to the repo. `main.py` expects these files to be in a folder called `data/` located in the project root.

```
git clone https://github.com/aujxn/portland_traffic_analysis.git
cd portland_traffic_analysis
mkdir data
cp <the data files listed above> data
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd src
python3 main.py   <or>   python3 app.py
```

`python3 main.py` generates the hourly plots for weekends vs weekdays for all of the locations. For a more interactive experience, running `python3 app.py` will start a simple web-server which can be accessed by going to `localhost:8050` in your browser. Click on one of the ATR locations on the rendered map to generate the associated plots.

## Results

Weekly progress and analysis can be found in the `weekly_reports` folder. See the first in progress report [here](weekly_reports/week3.md).

## License

See [license](LICENSE).
