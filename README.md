# Vehicle Budget Recommender

Enter a budget in PKR and the app shows what you can buy - from a cycle to a helicopter -
including combinations such as *"2 cycles + 1 electric bike"*.

## How it works

| Part | What it does |
|------|--------------|
| `vehicles.csv` | 59 vehicles (13 categories) with approximate PKR prices, fuel type, seats and top speed |
| **Random Forest** | Estimates the price of a vehicle from its specs (trained on log10 of the price) |
| **KNN** | Finds vehicles similar to a chosen one (category, fuel, price, speed, seats) |
| Combination search | Tries combinations of up to 4 vehicles that fit the budget and use most of it (knapsack-style search) |
| Streamlit app | `app.py` - the web page the user interacts with |

## Files

```
app.py                      Streamlit app
recommender.py              shared helper code (combos, KNN, price estimate)
train_vehicle_model.ipynb   Kaggle notebook - trains and saves model.joblib
vehicles.csv                dataset
requirements.txt            packages for deployment
model.joblib                created by the Kaggle notebook (download it and add it here)
```

## 1. Train on Kaggle

1. Kaggle -> **Datasets -> New Dataset** -> upload `vehicles.csv`.
2. Kaggle -> **Code -> New Notebook** -> **File -> Import Notebook** -> upload `train_vehicle_model.ipynb`.
3. In the notebook click **Add Input** and add your dataset.
4. **Run All**. At the end the notebook saves `model.joblib`.
5. Open the **Output** tab and download `model.joblib`.

## 2. Upload to GitHub

Create a new repository and upload: `app.py`, `recommender.py`, `vehicles.csv`,
`requirements.txt`, `README.md`, `model.joblib` and the notebook.
All files go in the **root** of the repository (not inside a folder).

## 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> choose your repository, branch `main`, main file `app.py`.
3. Click **Deploy**.

## Run locally (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

## Notes and limitations

- **Prices are approximate estimates**, not live market prices (see the `source` and `date` columns).
  Update `vehicles.csv` with real prices before submitting, and write the date you checked them.
  If you change the number of rows, run the Kaggle notebook again; the app will otherwise retrain
  from the CSV automatically.
- The dataset is small, so the Random Forest price estimate is rough
  (5-fold cross-validation R2 is about 0.85 on the log scale, and a typical estimate is within a factor of ~3 of the real price).
- If `model.joblib` is missing or was made with a different scikit-learn version, the app trains the models again from `vehicles.csv` on start-up.
