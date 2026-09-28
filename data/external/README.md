# NASA POWER enrichment

`nasa_power_features.csv` is generated from the public NASA POWER Daily Point API:

<https://power.larc.nasa.gov/docs/services/api/temporal/daily/point/>

The downloader requests meteorological and environmental variables for the unique 0.5° grid cells covering the competition rows, then computes rolling summaries using days strictly before each death date. It is climate/environmental data only and is joined back to rows by the supplied `ID`.

Regenerate it from the project root with:

```bash
python scripts/download_nasa_power.py
```
