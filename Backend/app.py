from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/")
def chart():
    labels = []
    values = [10,9,8,7,6]
    return render_template('chart.html', values=values, labels=str(labels), counter)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
