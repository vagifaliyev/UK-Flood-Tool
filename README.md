## Flood Risk Prediction tool

In environmental hazard assessment, risk is defined as the probability of a hazardous event occuring multiplied by the consequences of that event. Thus high risk can be the result of frequent events of low consequence or rare events of high consequence; the highest risk scenarios are often not obvious.

The Environment Agency routinely collects data on rainfall, sea level and river discharge and has published flooding probability maps that indicate areas of England that fall within four flood probability bands, based on the recurrence interval of flood levels that cause property damage and threat to life. These bands are very low probability (flooding expected once per 1000 years), low probability (flooding expected once per 100 years), medium probability (flooding expected once per 50 years), and high probability (flooding expected once per 10 years).

This tool calculates flood probabilities and risks for postcodes in England.

![](https://github.com/vagifaliyev/UK-Flood-Tool/blob/master/Presentation.pdf)

### Installation Guide

```
python -m pip install -r requirements.txt
```

### User instructions

For the extension 3, the required input from the user is the postcodes that are to be analysed. The output will be alert levels which are depended on flood risk and rainfall levels. 

For extensions 4, the required input is the date to be analysed in the format of yyyy/mm/dd. This function will access historical rainfall data across all the stations avaliable and generate a visual map with colour-coded alert levels. 

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be build by running

```
python -m sphinx docs html
```

then viewing the `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `FloodTool.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to checki its operation on your system. With [[pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```
python -m pytest flood_tool
``
The current version also includes a speed scoring algorithm. This can be run with.
```
python -m score
```
in the main repository directory.
