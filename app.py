# app.py
from flask import Flask, render_template, request
from main import get_qa_chain

app = Flask(__name__)
qa = get_qa_chain()

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = ""
    if request.method == "POST":
        user_query = request.form["query"]
        try:
            response = qa.invoke({"query": user_query})
            answer = response["result"]
        except Exception as e:
            answer = f"❌ Error: {e}"
    return render_template("chat.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
