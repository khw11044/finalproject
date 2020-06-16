
import time
import atexit
import sys
import termios
import contextlib
import threading
import imutils
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor


@contextlib.contextmanager
def raw_mode(file):
    """
    Magic function that allows key presses.
    :param file:
    :return:
    """
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)




class ServoMotor():
    """
    Class used for turret control.
    """
    def __init__(self):

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)

        self.p = GPIO.PWM(18, 50)
        self.p.start(0)
        time.sleep(0.5)
        #self.p.ChangeDutyCycle(5.5)
        
    
    def right(self):
        self.p.ChangeDutyCycle(6)
        time.sleep(0.04)
        self.p.ChangeDutyCycle(100)
        print("right")
        
    def left(self):
        self.p.ChangeDutyCycle(90)
        time.sleep(0.04)
        self.p.ChangeDutyCycle(100)
        print("left")
        
    def stop(self):
        self.p.ChangeDutyCycle(100)
        time.sleep(0.005)
        print("stop")
        
    
    def interactive(self):
        """
        Starts an interactive session. Key presses determine movement.
        :return:
        """

        print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break
          
                    if ch == "a":
                        self.p.ChangeDutyCycle(5)
                        time.sleep(0.05)
                        print("left")
                    elif ch == "d":
                        self.p.start(7.5)
                        time.sleep(0.05)
                        print("right")
                        
                    elif ch == "s":
                        self.p.ChangeDutyCycle(0)
                        time.sleep(0.05)
                        print("stop")
                        
                    elif ch == "\n":
                        print('sdsdsdsdsd')

            except (KeyboardInterrupt, EOFError):
                pass



if __name__ == "__main__":
    '''
    s = ServoMotor()
    print('input commend (w) or (s) or (a) or (d)')
    s.interactive()
    
    #or
    '''
    s = ServoMotor()
    print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == "q":
                    break

                if ch == "a":
                    s.left()
                elif ch == "d":
                    s.right() 
                elif ch == "s":
                    s.stop()
                elif ch == "\n":
                    print('sdsdsdsdsd')

        except (KeyboardInterrupt, EOFError):
            pass
        





