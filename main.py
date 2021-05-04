import RSCCore as core
from flask import Flask
import json
from flask import request
from flask import jsonify

__app__ = Flask(__name__)


@__app__.route("/getSuggestions", methods=["POST"])
def get_suggestions():
    if request.method == "POST":
        content = request.get_data()
        content = content.decode("utf8").replace("[", "").replace("]", "").replace(" ", "").replace('''"''', "") \
            .replace("'", "").split(",")
        print(content)
        suggestions = core.get_suggestion(content)
        return jsonify(results=suggestions)


@__app__.route("/selectModel/<int:selected_model>")
def select_model(selected_model):
    status = core.select_model(selected_model=selected_model)
    if status == 200:
        return "OK"
    else:
        return "No model loaded"


@__app__.route("/updateDataset", methods=["POST"])
def updateDataset():
    if request.method == "POST":
        content = request.get_data()
        try:
            content = json.loads(content)
            title = content["Title"]
            ID = content["ID"]
            plot = content["Plot"]
            if core.update_dataset(ID=ID, title=title, plot=plot) == 400:
                return "Dataset not updated"
        except Exception:
            return "Format Error"
        return "OK"


@__app__.route("/getSuggestionsFromSentence", methods=["POST"])
def getSuggestionsFromSentence():
    if request.method == "POST":
        content = request.get_data()
        suggestions = core.get_suggestion_from_sentence(content)
        return jsonify(results=suggestions)
    

if __name__ == '__main__':
    __app__.run()
