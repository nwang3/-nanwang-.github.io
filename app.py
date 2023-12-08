from flask import Flask, render_template, request, send_file
import pandas as pd
from io import BytesIO
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from flask.blueprints import Blueprint

app = Flask(__name__)
columns = ['zip_code', 'city', 'bed', 'bath', 'house_size', 'acre_lot', 'status', 'state']
model = joblib.load('linear_regression_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        dfs = []

        for file in files:
            df = pd.read_excel(file)
            dfs.append(df)

        combined = pd.concat(dfs)

        # Add analysis columns
        combined["YOY% Gross Rental Income"] = (
            combined.groupby("Property Name")["Gross Rental Income"].pct_change() * 100
        )
        combined["YOY% NOI"] = (
            combined.groupby("Property Name")["NOI"].pct_change() * 100
        )

        # Round everything to integer
        combined = combined.round(0)

        # Format % columns
        combined["YOY% Gross Rental Income"] = combined[
            "YOY% Gross Rental Income"
        ].apply(lambda x: f"{int(x)}%" if not pd.isna(x) else "")
        combined["YOY% NOI"] = combined["YOY% NOI"].apply(
            lambda x: f"{int(x)}%" if not pd.isna(x) else ""
        )

        # Reorder columns with Property Name as the first column
        combined = combined[
            ["Property Name"]
            + [col for col in combined.columns if col != "Property Name"]
        ]

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine="xlsxwriter")
        combined.to_excel(writer, index=False)
        writer._save()
        output.seek(0)

        return send_file(output, as_attachment=True, download_name="consolidated.xlsx")

    return render_template("index.html")


@app.get("/predict")
def predict_get():
    return render_template("index2.html")


# @app.post("/predict")
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        bed = float(request.form['bed']) 
        bath = float(request.form['bath'])
        
        # Matches what model was trained on 
        input_data = pd.DataFrame({'bed': [bed], 'bath': [bath]})  
        
        predicted_price = model.predict(input_data)[0]

        return render_template('result.html', predicted_price=predicted_price)


if __name__ == "__main__":
    app.run(debug=True)