import os


class Model:
    def __init__(self, file_model):
        self.file_model = file_model

    def predict(self,ftest,foutput):
        jar_path = os.path.dirname(os.path.realpath(__file__)) 
        command = "java -Xmx3600m -jar " + jar_path + "/tgboost.jar" \
              + " " + "testing" \
              + " " + self.file_model \
              + " " + ftest \
              + " " + foutput

        os.system(command)

    def __del__(self):
        os.remove(self.file_model)
