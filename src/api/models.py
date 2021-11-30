from app import db
import json


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    code = db.Column(db.Integer, unique=True, nullable=False)
    encoding = db.Column(db.LargeBinary, unique=False, nullable=False)

    def __repr__(self):
        return str({
            "id": self.id,
            "code": self.email,
            "encoding": self.encoding
        })
