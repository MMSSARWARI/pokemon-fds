# FDS Pokemon Battle Prediction (Gen 2 OU)

Classical machine learning only (logistic regression, LightGBM, XGBoost). Only the competition data and the Gen 2 type chart are used.

## Files
- `src/features.py`: turns each battle into one row of numbers (team stats, type matchups, HP curves, fainted Pokemon, status, boosts, moves).
- `src/run_pipeline.py`: repeated 5-fold cross-validation, final training, and the 3 submission files.
- `notebooks/kaggle_notebook.ipynb`: Kaggle notebook that clones this repo and creates the 3 submissions.
- `outputs/cv_results.csv`: cross-validation accuracy of every model and blend.

## Run
```
pip install -r requirements.txt
python src/run_pipeline.py --data /path/to/competition/files --out outputs
```
The data folder must contain train.jsonl, test.jsonl and type_chart_gen2.csv. Runtime is about 5 minutes on one CPU. All random seeds are fixed, so the results can be reproduced.

## Submissions
| file | model |
|---|---|
| sub1_logreg.csv | logistic regression |
| sub2_gbm_blend.csv | mean of LightGBM and XGBoost probabilities |
| sub3_all_blend.csv | mean of logistic regression, LightGBM and XGBoost probabilities |
