import numpy as np
import os
import sys; sys.path.append("src")
import uuid
import mimetypes
import json

import falcon

from dmn import DMN


class QuestionAnswerer(object):

    def __init__(self):

        self._answerer = DMN(100, 3, name="test_task_2", vocabulary="load")
        self._answerer.load()

        
    def answer(self, context, question):

        context = context.split("\n")
        
        answer, attention = self._answerer.answer(context, question,
                                                  max_answer_length=1,
                                                  return_attention=True)
        attention = attention / np.sum(attention, axis=0)
        attention = [["{0:.5f}".format(aa) for aa in a] for a in attention]
        context = [{"sentence": c, "attention": a} for c,a in zip(context, attention)]
        
        response = {"answer": answer, "context": context}
        response = json.dumps(response)

        return response

    
    def on_post(self, req, resp):
        data = json.loads(req.stream.read().decode("utf-8"))
        resp.body = self.answer(data["context"], data["question"])
        resp.status = falcon.HTTP_200
        


class Static(object):

    def on_get(self, req, resp, file_name):
        resp.stream = open(os.path.join("static", file_name), "r")
        resp.content_type = mimetypes.guess_type(file_name)[0]
        resp.status = falcon.HTTP_200


class Index(object):

    def on_get(self, req, resp):
        resp.stream = open(os.path.join("static", "index.html"), "r")
        resp.content_type = "text/html"
        resp.status = falcon.HTTP_200
    
