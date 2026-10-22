import socket
import json


class Connection:
    def __init__(self):
        self.socket = socket.create_connection(("localhost", 5555))
        self.fileObject = self.socket.makefile("rw")

    def sendrecv(self, command):
        self.socket.sendall((json.dumps(command) + "\n").encode())
        newMapEncoding = self.fileObject.readline()
        return json.loads(newMapEncoding)

    def reset(self):
        newMapEncoding = self.sendrecv({"cmd": "reset"})
        return newMapEncoding

    def step(self):
        newMapEncoding = self.sendrecv({"cmd": "step"})
        return newMapEncoding
    
    def close(self):
        self.fileObject.close()
        self.socket.close()

if __name__ == "__main__":
    connectionObj = Connection()
    print(connectionObj.step())
    print(connectionObj.reset())
    connectionObj.close()
