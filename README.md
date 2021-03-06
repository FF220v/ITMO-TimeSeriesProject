# ITMO-TimeSeriesProject
A repo for teamwork on a time series project.
Our project is *SuperMegaProTimeSeriesAnalyser*!

[Ссылка на презентацию](https://www.figma.com/proto/D5aEWf299NAx0yXznDOFHR/TSA-T-course?node-id=28%3A2&viewport=6048%2C664%2C2.2322583198547363&frame-preset-name=Desktop&scaling=contain)

SuperMegaProTimeSeriesAnalyser is a tool for time series analysis. It allows you to use 6 different methods 
of prediction with your datasets in a pretty graphical browser-based interface.

To start SuperMegaProTimeSeriesAnalyser in docker:
- `./build.sh && ./run.sh` will build docker image and start SuperMegaProTimeSeriesAnalyser on port 8050.
- OR just use `./run.sh` to use the latest version from dockerhub.
- Open a browser and go to http://localhost:8050

To start SuperMegaProTimeSeriesAnalyser locally on own machine: 
- Python 3.8 is required.  
- Once you acquired python 3.8, install dependencies `pip install -r src/requiremens.txt`  
- Then run `python src/main.py`, it will start the tool on 8050 port.   
- Open browser and go to http://localhost:8050
