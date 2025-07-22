from flask import Flask,redirect,url_for,render_template,request
from main import get_diet_recommendation

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        user_data = {
            "weight" : float(request.form["weight"]),
            "height" : float(request.form["height"]),
            "age" : int(request.form["age"]),
            "gender" : request.form["gender"].lower(),
            "goal" : request.form["goal"].lower(),
            "diet_type" : request.form["diet_type"].lower(),
            "activity_level" : request.form["activity_level"].lower()
        }
        calories, recommendation = get_diet_recommendation(user_data=user_data)
        return render_template(
            "result.html",
            calories = calories,
            recommendation = recommendation
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
