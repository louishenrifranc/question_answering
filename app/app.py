import os
import sys; sys.path.append("backend");
import mimetypes

import falcon

from question_answerer import QuestionAnswerer, Static, Index


app = application = falcon.API()

def static(req, res, static_dir="static", index_file="index.html"):
    path = static_dir + req.path
    if req.path == "/":
        path += index_file
    if os.path.isfile(path):
        res.content_type = mimetypes.guess_type(path)[0]
        res.status = falcon.HTTP_200
        res.stream = open(path)
    else:
        res.status = falcon.HTTP_404

app.add_route("/ask", QuestionAnswerer())
app.add_route("/", Index())
app.add_route("/static/{file_name}", Static())




