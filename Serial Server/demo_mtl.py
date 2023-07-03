import re
import numpy as np
from colorama import Fore
from sklearn.metrics import classification_report

from uart_server import SerialServer
from Pipeline.evaluation import plot_confusion_matrix, print_detection_metrics
from Pipeline.tf_dataset_builder import RawDatasetBuilder


class SerialServerMTL(SerialServer):
    def __init__(self, port):
        super().__init__(port)

    def receive_serial(self, stream):
        count = 0
        y_pred_detect = []
        y_pred_class = []

        y_true_detect = stream[0]
        y_true_class = stream[1]

        ex_time_list = []

        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii').strip('\x00')
                if 'Detection' in line:
                    pred = int(line[-2])
                    y_pred_detect.append(pred)
                elif 'Classification' in line:
                    pred = int(line[-2])
                    y_pred_class.append(pred)
                    print(Fore.CYAN + '[SERVER] Predicted: {} | Actual: {}'.format(pred, y_true_class[count]))
                    count += 1

                    if count % 10 == 0 and count != 0:
                        print(Fore.CYAN + '[SERVER] Current report:\n')
                        print_detection_metrics(y_true_detect[:len(y_pred_detect)], y_pred_detect, Fore.CYAN)
                        print(Fore.CYAN + classification_report(y_true_class[:len(y_pred_class)], y_pred_class))
                elif 'duration' in line:
                    match = re.search(r'stage duration: (\d+)', line)
                    if match:
                        ex_time = int(match.group(1))
                        ex_time_list.append(ex_time)
                        print(f"Execution time: {ex_time} us")
                else:
                    print(Fore.WHITE + line)

                if count == len(y_true_detect):
                    print(Fore.CYAN + '[SERVER] Final report:\n')
                    print_detection_metrics(y_true_detect, y_pred_detect, Fore.CYAN)
                    print(Fore.CYAN + classification_report(y_true_class, y_pred_class))
                    plot_confusion_matrix(
                        ['Background', 'Big mandibles', 'Small mandibles'],
                        y_true_class,
                        y_pred_class
                    )
                    # compute average execution time +- confidence interval
                    ex_time_list = np.array(ex_time_list)
                    conf_interval = 1.96 * np.std(ex_time_list) / np.sqrt(len(ex_time_list))
                    print(f"Average execution time: {np.mean(ex_time_list)} us")
                    print(f"Confidence interval: {conf_interval} us")
                    break

        self.ser.close()


if __name__ == "__main__":
    # Find COM port in device manager
    server = SerialServerMTL('COM3')

    dataset_builder = RawDatasetBuilder(
        dataset_dir="../../Dataset/Final dataset/Generation/Generated dataset/**/*.wav",
        class_dict={'Background': 0, 'Big mandibles': 1, 'Small mandibles': 2},
        seed=1,
    )
    _, _, test_dataset = dataset_builder.build(val_split=0.2, test_split=0.2)
    x_test, y_test_classification = test_dataset[0], test_dataset[1]
    y_test_detection = np.array(y_test_classification, copy=True)
    y_test_detection[y_test_detection != 0] = 1

    y_test = [y_test_detection, y_test_classification]

    # listen for serial output from the MCU
    server.receive(y_test)

    # send stream of float values to the MCU
    server.send(x_test)
