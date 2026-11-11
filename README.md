# Farm Price Predictor

## 1. Mission and problem
Smallholder farmers in Rwanda often sell to middlemen without knowing the fair market price of their crops.
This project predicts the expected market price (RWF per unit) of a commodity in a specific Rwandan market,
for a given month and price type (Retail/Wholesale), so farmers can decide where and when to sell and what to ask,
and buyers can make fair offers.

## 2. Dataset
WFP **Rwanda - Food Prices** (World Food Programme, via the Humanitarian Data Exchange):
https://data.humdata.org/dataset/wfp-food-prices-for-rwanda

Monthly market prices from 2000 to 2026: about 158,000 rows covering 65 commodities in 109 markets across all 5 provinces,
with location, commodity category, unit, price type and price in RWF. After cleaning, 157,333 rows (2008–2026) are used
with 11 features (4 numeric, 7 categorical). The best model is a Random Forest (test RMSE ≈ 1,459 RWF, R² = 0.98).

## 3. API (Swagger UI)
https://farmgate-api.onrender.com/docs

The API runs on Render's free plan and sleeps when idle, so the first request may take about 50 seconds to respond.

## 4. Video demo
YOUTUBE_LINK_HERE

## 5. Run the mobile app

**Prerequisites:** Git, the Flutter SDK (stable), and Android Studio with an Android emulator (or an Android phone with USB debugging on).
Run `flutter doctor` to check your setup.

1. Clone the repository:
   ```bash
   git clone https://github.com/Rutijana/linear_regression_model.git
   ```
2. Go to the app folder:
   ```bash
   cd linear_regression_model/summative/FlutterApp
   ```
3. Install the packages:
   ```bash
   flutter pub get
   ```
4. Start an emulator (or plug in your phone):
   ```bash
   flutter emulators                      # list available emulators
   flutter emulators --launch <emulator_id>
   flutter devices                        # wait until the device appears
   ```
5. Run the app:
   ```bash
   flutter run -d <device_id>
   ```
6. In the app, fill in the 5 fields (e.g. `Maize`, `Kimironko`, `Retail`, `2025`, `3`) and tap **Predict**.
   The expected price appears below the button; invalid or missing values show an error message in the same place.
