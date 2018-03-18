import os
import shutil


class Model:
    def __init__(self, file_model, auto_clear=True):
        self.file_model = file_model
        self.auto_clear = auto_clear

    def predict(self,ftest,foutput):
        jar_path = os.path.dirname(os.path.realpath(__file__)) 
        command = "java -Xmx3600m -jar " + jar_path + "/tgboost.jar" \
              + " " + "testing" \
              + " " + self.file_model \
              + " " + ftest \
              + " " + foutput

        os.system(command)

    def save(self, path):
        shutil.copy(self.file_model, path)

    def __del__(self):
        if self.auto_clear:
            os.remove(self.file_model)


def load_model(file_model):
    return Model(file_model, auto_clear=False)
