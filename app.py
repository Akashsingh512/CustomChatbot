from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from main import get_qa_chain

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for React
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

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = qa.invoke({"query": query})
        return jsonify({"answer": response["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)