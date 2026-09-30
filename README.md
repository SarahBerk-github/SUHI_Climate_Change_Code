# Projected changes in surface urban heat islands under 2 °C climate change

This repository contains the code used to model present-day and future changes in the surface urban heat island (SUHI) across cities globally.

The analysis combines satellite-derived observations of urban and rural land surface temperature and environmental characteristics with a Regression-Enhanced Random Forest (RERF) model. Changes in climate and vegetation derived from CMIP6 Earth System Models (ESMs) are then used to estimate changes in SUHI intensity under future climate conditions.

## Workflow

The analysis consists of five main stages:

1. **City selection** – calculate the criteria used to identify cities suitable for inclusion in the analysis.
2. **Present-day predictor calculation** – calculate city-scale variables, including SUHI magnitude, vegetation (EVI) and surface albedo.
3. **Model fitting** – fit the Regression-Enhanced Random Forest model used to represent present-day SUHI intensity.
4. **Climate projections** – calculate changes in relevant predictor variables from CMIP6 Earth System Model simulations.
5. **Future SUHI projections** – apply the projected changes in predictor variables to the fitted RERF model to estimate future changes in SUHI intensity.

## Repository contents

| File | Description |
|---|---|
| `City_Selection_Criteria_Calculator.ipynb` | Calculates the criteria used to select cities for inclusion in the analysis. |
| `EVI_Means_Calculator.py` | Calculates mean Enhanced Vegetation Index (EVI) values for the urban and surrounding rural areas. |
| `Albedo_Means_Calculator.py` | Calculates mean surface albedo for the urban and surrounding rural areas. |
| `RERF_Fitting.ipynb` | Fits and evaluates the Regression-Enhanced Random Forest model used to model SUHI intensity. |
| `ESM_Projection_Calculator_run_on_JASMIN.ipynb` | Processes CMIP6 Earth System Model output and calculates projected changes in model predictor variables. This analysis was run using the JASMIN computing environment. |
| `RERF_ESM_Projections.ipynb` | Applies the ESM-derived changes in predictor variables to the fitted RERF model to estimate projected changes in SUHI intensity. |

## Data

The analysis uses a combination of satellite observations and CMIP6 Earth System Model output.

Due to the size of the input datasets, the raw data are not stored directly in this repository. File paths in the scripts therefore need to be updated to point to the corresponding datasets on the user's system.

City population and location data is available at https://www.un.org/en/desa/2018-revision-world-urbanization-prospects. Coastal distance data is available at https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/. Water proximity data is available at https://catalogue.ceda.ac.uk/uuid/84d4f66b668241328df0c43f8f3b3e16. Topography data is available at https://www.ngdc.noaa.gov/mgg/topo/globe.html. Landcover data is available at https://catalogue.ceda.ac.uk/uuid/b382ebe6679d44b8b0e68ea4ef4b701c. LST data is available at https://lpdaac.usgs.gov/products/myd11a2v006/. Vegetation Index data is available at  https://catalogue.ceda.ac.uk/uuid/d65aab04c69b4df391e6e7fc4b901aef and https://catalogue.ceda.ac.uk/uuid/f10420bececd447eb1b74db9d66ef12a. Albedo data is available at https://catalogue.ceda.ac.uk/uuid/48efa9b67d69435caffd2d06cf8406d3 and https://catalogue.ceda.ac.uk/uuid/cf8e3e801a114979a160e809deb5cc9f. CMIP6 ESM data is available at https://catalogue.ceda.ac.uk/uuid/b96ce180077f4810abc4eef0e48901d9. 

## Requirements

The analysis is written primarily in Python and uses common scientific, geospatial, and machine-learning packages.

Key packages include:

- `numpy`
- `pandas`
- `xarray`
- `scikit-learn`
- `matplotlib`
- `geopandas`
- `rasterio`

Additional packages are imported by individual scripts and notebooks.

## Running the analysis

The approximate workflow is:

```text
City selection
      ↓
Present-day variable calculation
      ↓
RERF model fitting
      ↓
CMIP6 ESM predictor changes
      ↓
Future SUHI projections
```

The scripts contain file paths specific to the original computing environment. These paths should be changed before running the analysis on another system.

Processing of the CMIP6 ESM output was originally performed on JASMIN, the UK collaborative data analysis facility (NERC, UKRI). 

## Associated publication

This repository contains code associated with:

[Berk, S. et al. (2026) “Amplified warming in tropical and subtropical cities under 2 °C climate change,” Proceedings of the National Academy of Sciences](https://doi.org/10.1073/pnas.2502873123)
