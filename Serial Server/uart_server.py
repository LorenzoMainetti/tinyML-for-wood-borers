import time
from threading import Thread

import serial
import struct
from colorama import Fore


class SerialServer:
    def __init__(self, port):
        self.port = port
        self.reset_serial()
        self.ser = self.init_serial()

    def init_serial(self):
        try:
            ser = serial.Serial(port=self.port, baudrate=115200, timeout=0.5)
            print(Fore.GREEN + "[SERVER] Connection to " + self.port + " established successfully!\n")
        except serial.SerialException as var:
            print(Fore.GREEN + "[SERVER] Connection to " + self.port + " failed!\n")
            print(Fore.GREEN + 'Exception Details-> ', var)
            return None

        return ser

    def reset_serial(self):
        # TODO fix
        # reset and clean UART channel
        ser = self.init_serial()
        ser.write(struct.pack('<f', 0.0))
        ser.flush()
        ser.close()
        print(Fore.GREEN + "[SERVER] UART channel has been reset\n")

    def send_serial(self, stream):
        stream = stream.astype('float32')
        count = 0

        for i in stream:
            self.ser.write(struct.pack('<f', i))
            time.sleep(0.005)  # takes around 30sec per file (2205 samples)
            self.ser.flush()
            count += 1
        # print(Fore.GREEN + '[SERVER] Sent {} bytes'.format(count))

    def send_serial_loop(self, stream):
        for count, file in enumerate(stream):
            self.send_serial(file)
            print(Fore.GREEN + '[SERVER] Sent {} files'.format(count + 1))

    def send(self, stream):
        sending_thread = Thread(target=self.send_serial_loop, args=(stream,))
        sending_thread.start()

    def receive_serial(self, stream):
        # default implementation, override in child class
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii').strip('\x00')
                print(line)

    def receive(self, stream):
        receiving_thread = Thread(target=self.receive_serial, args=(stream,))
        receiving_thread.start()

