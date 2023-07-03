import threading

import matplotlib

matplotlib.use('TkAgg')  # Switch the backend to TkAgg

import re
import queue
import numpy as np
from colorama import Fore
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns

from uart_server import SerialServer
from Pipeline.tf_dataset_builder import RawDatasetBuilder


class DynamicConfusionMatrix:
    # TODO not working
    def __init__(self, classes):
        self.classes = classes
        self.cm = np.zeros((len(classes), len(classes)), dtype=int)
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.lock = threading.Lock()  # Lock to synchronize access to the confusion matrix
        self.prediction_queue = queue.Queue()  # Queue for receiving prediction updates
        self.fig_canvas = FigureCanvas(self.fig)

    def update(self, y_true, y_pred):
        with self.lock:
            self.cm[y_true, y_pred] += 1

    def process_predictions(self):
        while True:
            y_true, y_pred = self.prediction_queue.get()
            self.update(y_true, y_pred)
            self.update_plot()

    def add_predictions(self, y_true, y_pred):
        self.prediction_queue.put((y_true, y_pred))

    def update_plot(self):
        self.ax.cla()  # Clear the current axes
        self.ax.set_title('Confusion Matrix')
        self.im = sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                              xticklabels=self.classes, yticklabels=self.classes, ax=self.ax)
        self.ax.set_xlabel('Predicted label')
        self.ax.set_ylabel('True label')
        self.fig_canvas.draw()

    def start_animation(self):
        ani = FuncAnimation(self.fig, lambda i: None, interval=1000)
        plt.show()


class SerialServerSequential(SerialServer):
    def __init__(self, port):
        super().__init__(port)

    def receive_serial(self, stream):
        count = 0
        predictions = []

        filtering_ex_time = []
        detection_ex_time = []
        spectrogram_ex_time = []
        classification_ex_time = []

        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii').strip('\x00')
                if 'detected' in line:
                    pred = int(line[-2])
                    predictions.append(pred)
                    # dyn_cm.add_predictions(stream[count], pred)

                    print(Fore.CYAN + '[SERVER] Predicted: {} | Actual: {}'.format(pred, stream[count]))
                    if count % 10 == 0 and count != 0:
                        print(Fore.CYAN + '[SERVER] Current report:\n{}'.format(
                            classification_report(stream[:len(predictions)], predictions)))
                    count += 1
                elif 'duration' in line:
                    match = re.search(r'stage duration: (\d+)', line)
                    if match:
                        ex_time = int(match.group(1))
                        if 'F' in line:
                            print(f"Filtering stage duration: {ex_time}")
                            filtering_ex_time.append(ex_time)
                        elif 'D' in line:
                            print(f"Detection stage duration: {ex_time}")
                            detection_ex_time.append(ex_time)
                        elif 'S' in line:
                            print(f"Spectrogram stage duration: {ex_time}")
                            spectrogram_ex_time.append(ex_time)
                        elif 'C' in line:
                            print(f"Classification stage duration: {ex_time}")
                            classification_ex_time.append(ex_time)
                else:
                    print(Fore.WHITE + line)

                if count == len(stream):
                    print(Fore.CYAN + '[SERVER] Final report:\n{}'.format(classification_report(stream, predictions)))
                    break

        self.ser.close()


# Create a dynamic confusion matrix object
# dyn_cm = DynamicConfusionMatrix(['Background', 'Big mandibles', 'Small mandibles'])

if __name__ == "__main__":
    # Find COM port in device manager
    server = SerialServerSequential('COM3')

    dataset_builder = RawDatasetBuilder(
        dataset_dir="../../Dataset/Final dataset/Generation/Generated dataset/**/*.wav",
        class_dict={'Background': 0, 'Big mandibles': 1, 'Small mandibles': 2},
        seed=1,
    )
    _, _, test_dataset = dataset_builder.build(val_split=0.2, test_split=0.2)
    x_test, y_test = test_dataset[0], test_dataset[1]

    # Create a separate thread for processing predictions
    # update_thread = threading.Thread(target=dyn_cm.process_predictions)
    # update_thread.start()

    # listen for serial output from the MCU
    server.receive(y_test)

    # send stream of float values to the MCU
    server.send(x_test)

    # Start the animation in the main thread
    # dyn_cm.start_animation()
