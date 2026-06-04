from flask import Flask, render_template, request
from src.predict import predict_churn


app = Flask(__name__)

@app.route("/", methods= ["GET", "POST"])
def home():
    result =  None

    if request.method == "POST":
        input_data = {
            "Gender": request.form["Gender"],
            "Senior Citizen": int(request.form["Senior Citizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "Tenure Months": int(request.form["Tenure Months"]),
            "Phone Service": request.form["Phone Service"],
            "Multiple Lines": request.form["Multiple Lines"],
            "Internet Service": request.form["Internet Service"],
            "Online Security": request.form["Online Security"],
            "Online Backup": request.form["Online Backup"],
            "Device Protection": request.form["Device Protection"],
            "Tech Support": request.form["Tech Support"],
            "Streaming TV": request.form["Streaming TV"],
            "Streaming Movies": request.form["Streaming Movies"],
            "Contract": request.form["Contract"],
            "Paperless Billing": request.form["Paperless Billing"],
            "Payment Method": request.form["Payment Method"],
            "Monthly Charges": float(request.form["Monthly Charges"]),
            "Total Charges": float(request.form["Total Charges"]),
            "CLTV": float(request.form["CLTV"])
        }


        result = predict_churn(input_data)

    return render_template("index.html", result= result)

if __name__ == "__main__":
    app.run(debug = True)