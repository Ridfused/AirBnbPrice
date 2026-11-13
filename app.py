
from flask import Flask, render_template_string, request

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Airbnb Price Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #101827;
            color: white;
        }

        header {
            padding: 22px 8%;
            background: #1e293b;
            font-size: 23px;
            font-weight: bold;
            color: #38bdf8;
        }

        .container {
            max-width: 850px;
            margin: 55px auto;
            padding: 35px;
            background: #1e293b;
            border-radius: 18px;
        }

        h1 { text-align: center; }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 35px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        label {
            display: block;
            margin-bottom: 9px;
            color: #cbd5e1;
        }

        input, select {
            width: 100%;
            padding: 13px;
            border: 1px solid #475569;
            border-radius: 8px;
            background: #0f172a;
            color: white;
        }

        button {
            width: 100%;
            padding: 16px;
            margin-top: 28px;
            border: none;
            border-radius: 9px;
            background: #38bdf8;
            color: #0f172a;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover { background: #0ea5e9; }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #0f766e;
            border-radius: 10px;
            text-align: center;
        }

        @media(max-width: 600px) {
            .grid { grid-template-columns: 1fr; }
            .container { margin: 25px 15px; padding: 22px; }
        }
    </style>
</head>

<body>

<header>AirbnbPrice AI</header>

<div class="container">

    <h1>Airbnb Price Predictor</h1>

    <p class="subtitle">
        Estimate the nightly rental price of your property.
    </p>

    <form method="POST">

        <div class="grid">

            <div>
                <label>Location</label>
                <input name="location" placeholder="Enter city or neighbourhood" required>
            </div>

            <div>
                <label>Room Type</label>
                <select name="room_type" required>
                    <option value="Entire home/apt">Entire home</option>
                    <option value="Private room">Private room</option>
                    <option value="Shared room">Shared room</option>
                </select>
            </div>

            <div>
                <label>Accommodates</label>
                <input type="number" name="accommodates" min="1" value="2" required>
            </div>

            <div>
                <label>Bedrooms</label>
                <input type="number" name="bedrooms" min="0" value="1" required>
            </div>

            <div>
                <label>Bathrooms</label>
                <input type="number" name="bathrooms" min="0" step="0.5" value="1" required>
            </div>

            <div>
                <label>Minimum Nights</label>
                <input type="number" name="minimum_nights" min="1" value="1" required>
            </div>

        </div>

        <button type="submit">Predict Airbnb Price</button>

    </form>

    {% if message %}
    <div class="result">
        {{ message }}
    </div>
    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    message = None

    if request.method == "POST":
        message = "Your property details were received! ML prediction engine is coming next."

    return render_template_string(HTML, message=message)


if __name__ == "__main__":
    app.run(debug=True)
