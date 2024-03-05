from flask import Flask, request, render_template
from Main import generate_poem
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        poem = generate_poem(seed_text)
        return render_template('index.html', generated_poem=poem)
    return render_template('index.html', generated_poem='')

if __name__ == '__main__':
    app.run(debug=True)
