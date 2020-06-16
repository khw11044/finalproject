
import time
import atexit
import sys
import termios
import contextlib
import threading
import imutils
import RPi.GPIO as GPIO

# Import the PCA9685 module.
import Adafruit_PCA9685


pwm = Adafruit_PCA9685.PCA9685()
servo_min = 150  # Min pulse length out of 4096
servo_max = 600  # Max pulse length out of 4096

# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

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
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.servo_min = 5  # Min pulse length out of 4096
        self.servo_max = 600  # Max pulse length out of 4096
        self.pwm.set_pwm_freq(10)
        self.pulse = 580
        #self.p.ChangeDutyCycle(5.5)
        
        
    
    def right(self):
        pwm.set_pwm(0, 0, 387) #389
        time.sleep(0.01)
        #pwm.set_pwm(0, 0, 0)
        print("right")
        
    def left(self):
        pwm.set_pwm(0, 0, 400) #399
        time.sleep(0.01)
        #pwm.set_pwm(0, 0, 0)
        print("left")
        
    def stop(self):
        pwm.set_pwm(0, 0, 0)
        time.sleep(0.005)
        print("stop")
        
    def up(self):
        self.pulse -= 6
        self.pwm.set_pwm(1, 0, self.pulse)
        time.sleep(0.01)
        print("up")
        print(self.pulse)
        if self.pulse < 200:
            self.pulse = 200
        
    def down(self):
        self.pulse += 6
        self.pwm.set_pwm(1, 0, self.pulse)
        time.sleep(0.01)
        print("down")
        print(self.pulse)
        if self.pulse > 700:
            self.pulse = 700
     
    
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
                elif ch == "w":
                    s.up()
                elif ch == "x":
                    s.down()
                elif ch == "\n":
                    print('sdsdsdsdsd')

        except (KeyboardInterrupt, EOFError):
            pass
        






