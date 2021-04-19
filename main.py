import RSCCore as core
from flask import Flask
import json
import io
from flask import request
from flask import jsonify

__app__ = Flask(__name__)


@__app__.route("/getSuggestions", methods=["POST"])
def get_suggestions():
    if request.method == "POST":
        content = request.get_data()
        content = content.decode("utf8").replace("[", "").replace("]", "").replace(" ", "").split(",")
        suggestions = core.get_suggestion(content)
        return jsonify(results=suggestions)


@__app__.route("/selectModel/<int:selected_model>")
def select_model(selected_model):
    status = core.select_model(selected_model=selected_model)
    if status == 200:
        return "OK"
    else:
        return "No model loaded"


if __name__ == '__main__':
    __app__.run()
