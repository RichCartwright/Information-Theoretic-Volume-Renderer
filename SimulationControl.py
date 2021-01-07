import socket
import struct
import csv

class SimulationControl():

    def __init__(self):
        self.PORT, self.HOST_IP = 8888, '127.0.0.1'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = (self.HOST_IP, self.PORT)
        print('\nStarting up on %s port %s\n' % self.server_address)
        self.sock.bind(self.server_address)
        self.sock.listen(1)
        self.state = None
        self.connection, client_address = self.sock.accept()
        #while True:
        #    recv_control()
        #    print >>sys.stderr, 'Closing connection'

        self.__rotstepsize = 1
        self.__zoombounds = (-10.0,-5.0)
        self.__zoomstepsize = 0.1
        self.reward = 0
        self.rotation = [0.0,0.0]
        self.zoom = 0.0
        self.MI = 0.0
        #self.stepnumber = 0
        self.done = False

    ## Action spaces
    # The actual movement of the model should have a negative reward
    # I'll have to tune this
    # TODO: All of the clamping is a bit shit and messy, probably no point in fixing it but...
    def RotateCW(self):
        if (self.rotation[0] + self.__rotstepsize) >= 360:
            self.rotation[0] = self.__rotstepsize
        else:
            self.rotation[0] += self.__rotstepsize
        self.reward -= 1

    def RotateCCW(self):
        if (self.rotation[0] - self.__rotstepsize) <= 0:
            self.rotation[0] = 360 - self.__rotstepsize
        else:
            self.rotation[0] -= self.__rotstepsize
        self.reward -= 1

    def RotatePitchN(self):
        if (self.rotation[1] + self.__rotstepsize) >= 360:
            self.rotation[1] = self.__rotstepsize
        else:
            self.rotation[1] += self.__rotstepsize
        self.reward -= 1

    def RotatePitchS(self):
        if (self.rotation[1] - self.__rotstepsize) <= 0:
            self.rotation[1] = 360 - self.__rotstepsize
        else:
            self.rotation[1] -= self.__rotstepsize
        self.reward -= 1

    def ZoomIn(self):
        if (self.zoom - self.__zoomstepsize) <= self.__zoombounds[0]:
            self.zoom = self.__zoombounds[0]
        else:
            self.zoom -= self.__zoomstepsize
        self.reward -= 2

    def ZoomOut(self):
        if (self.zoom + self.__zoomstepsize) >= self.__zoombounds[1]:
            self.zoom = self.__zoombounds[1]
        else:
            self.zoom += self.__zoomstepsize
        self.reward -= 2

    def send_control(self, array):
        buff = struct.pack('f' * len(array), *array)
        self.connection.sendall(bytes(buff))

    def recv_control(self):
        data = self.connection.recv(2000)
        if data:
            if data[0] == '[':
                print >>sys.stderr, '%s' % data
            else:
                data = struct.unpack('fffff', data)
                currentstate = (data[0], data[1], data[2], data[3], data[4]);
                print(data)
                with open('LearningResults.csv', 'a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([data[0], data[1], data[2], data[3], data[4]])
                #send_control(self, data)
        return currentstate

    def run_frame(self):
        # We need to get the new data first
        self.connection, client_address = self.sock.accept()
        retdata = self.recv_control()

        if (self.step == 0):
            self.rotation = (retdata[1], retdata[2])
            self.zoom = retdata[3]
            self.MI = retdata[4]
            return
        else:
            self.reward += (retdata[4] - self.MI) * 20
            self.MI = retdata[4]

    def step(self, action):
        self.send_control([self.rotation[0], self.rotation[1], self.zoom])
        self.reward = 0

        # Action == 0 == DO NOTHING
        if action == 1:
            self.RotateCW()
        if action == 2:
            self.RotateCCW()
        if action == 3:
            self.ZoomIn()
        if action == 4:
            self.ZoomOut()
        if action == 5:
            self.RotatePitchN()
        if action == 6:
            self.RotatePitchS()

        self.run_frame()

        state = (self.rotation[0], self.rotation[1], self.zoom, self.MI)
        print(self.reward)
        return self.reward, state, self.done
