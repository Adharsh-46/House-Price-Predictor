from flask import Flask, render_template, request
import pandas as pd
import joblib
import sklearn.compose._column_transformer as ct

# sklearn compatibility patch
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

app = Flask(__name__)

# Load model
model = joblib.load("house_price_model.joblib")

# Load dataset
df = pd.read_csv("cleaned_data.csv")


# ------------------ ROUTES ------------------

@app.route("/")   # ✅ Root route
def home():
    return render_template("home.html")


@app.route("/about.html")
def about():
    return render_template("about.html")


@app.route("/explore.html")
def explore():
    return render_template("explore.html")


@app.route("/contact.html")
def contact():
    return render_template("contact.html")


@app.route("/predict.html", methods=["GET", "POST"])
def predict():

    prediction = None

    titles = sorted(df["title"].unique())
    locations = sorted(df["location"].unique())
    status = sorted(df["building_status"].unique())

    if request.method == "POST":
        title = request.form["title"]
        location = request.form["location"]
        rate_persqft = float(request.form["rate_persqft"])
        area_insqft = float(request.form["area_insqft"])
        building_status = request.form["building_status"]

        input_data = pd.DataFrame([{
            "title": title,
            "location": location,
            "rate_persqft": rate_persqft,
            "area_insqft": area_insqft,
            "building_status": building_status
        }])

        prediction = round(model.predict(input_data)[0], 2)

    return render_template(
        "predict.html",
        prediction=prediction,
        titles=titles,
        locations=locations,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)