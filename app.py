from flask import Flask,url_for,render_template,request
from convert_model_output_to_html import return_html_from_model

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/extract',methods=["GET","POST"])
def extract():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		html = return_html_from_model(raw_text)
		result = HTML_WRAPPER.format(html)

	return render_template('result.html',rawtext=raw_text,result=result)

@app.route('/about')
def previewer():
	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=True)