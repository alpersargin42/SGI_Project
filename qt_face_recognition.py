import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets
from face_recognition_ui import Ui_Dialog

from analiz import analyze_faces

class FaceRecognitionApp(QWidget, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.start_analysis)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        self.setWindowTitle('Duygu Durum Analizi')

    def start_analysis(self):
        analyze_faces()

    def update_frame(self):
        pass

    def closeEvent(self, event):
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
