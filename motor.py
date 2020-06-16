
import time
import atexit
import sys
import termios
import contextlib
import threading
import imutils
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
import wiringpi

### User Parameters ###

MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False

MAX_STEPS_X = 1501
MAX_STEPS_Y = 15

RELAY_PIN = 22


wiringpi.wiringPiSetupGpio()

# set #18 to be a PWM output
wiringpi.pinMode(18, wiringpi.GPIO.PWM_OUTPUT)

# set the PWM mode to milliseconds stype
wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

# divide down clock
wiringpi.pwmSetClock(192)
wiringpi.pwmSetRange(2000)

delay_period = 0.0000001
pulse = 180


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




class StepMotor():
    """
    Class used for turret control.
    """
    def __init__(self):

        # create a default object, no changes to I2C address or frequency
        self.mh = Adafruit_MotorHAT()
        atexit.register(self.__turn_off_motors)

        # Stepper motor 1
        self.sm_x = self.mh.getStepper(200, 1)      # 200 steps/rev, motor port #1
        self.sm_x.setSpeed(1500)                       # 5 RPM
        self.current_x_steps = 0

        # Stepper motor 2
        self.sm_y = self.mh.getStepper(200, 2)      # 200 steps/rev, motor port #2
        self.sm_y.setSpeed(5)                       # 5 RPM
        self.current_y_steps = 0

        # Relay
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)
        
    def calibrate(self):
        """
        Waits for input to calibrate the turret's axis
        :return:
        """
        print("Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
              "(s) moves down. Press (enter) to finish.\n")
        self.__calibrate_y_axis()

        print("Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
              "(d) moves right. Press (enter) to finish.\n")
        self.__calibrate_x_axis()

        print("Calibration finished.")

    def __calibrate_x_axis(self):
        """
        Waits for input to calibrate the x axis
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Turret.move_backward(self.sm_x, 35)
                        else:
                            Turret.move_forward(self.sm_x, 35)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Turret.move_forward(self.sm_x, 35)
                        else:
                            Turret.move_backward(self.sm_x, 35)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)

    def __calibrate_y_axis(self):
        """
        Waits for input to calibrate the y axis.
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Turret.move_forward(self.sm_y, 4)
                            pulse += 2
                        else:
                            Turret.move_backward(self.sm_y, 4)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Turret.move_backward(self.sm_y, 4)
                            pulse -= 2
                        else:
                            Turret.move_forward(self.sm_y, 4)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)
    
    def right(self):
        if MOTOR_X_REVERSED:
            StepMotor.move_forward(self.sm_x, 5)
        else:
            StepMotor.move_backward(self.sm_x, 5)
        print("right")
        
    def left(self):
        if MOTOR_X_REVERSED:
            StepMotor.move_backward(self.sm_x, 5)
        else:
            StepMotor.move_forward(self.sm_x, 5)
        print("left")
        
    def up(self):
        if MOTOR_Y_REVERSED:
            StepMotor.move_forward(self.sm_y, 5)
            pulse += 2
        else:
            StepMotor.move_backward(self.sm_y, 5)
            pulse -= 2
        print("up")
        
    def down(self):
        if MOTOR_Y_REVERSED:
            StepMotor.move_backward(self.sm_y, 5)
            pulse -= 2
        else:
            StepMotor.move_forward(self.sm_y, 5)
            pulse += 2
            
    def stop(self):
        StepMotor.move_forward(self.sm_x, 0)
        StepMotor.move_backward(self.sm_x, 0)

    def interactive(self):
        """
        Starts an interactive session. Key presses determine movement.
        :return:
        """

        StepMotor.move_forward(self.sm_x, 1)
        StepMotor.move_forward(self.sm_y, 1)

        print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            StepMotor.move_forward(self.sm_y, 4)
                        else:
                            StepMotor.move_backward(self.sm_y, 4)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            StepMotor.move_backward(self.sm_y, 4)
                        else:
                            StepMotor.move_forward(self.sm_y, 4)
                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            StepMotor.move_backward(self.sm_x, 35)
                        else:
                            StepMotor.move_forward(self.sm_x, 35)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            StepMotor.move_forward(self.sm_x, 35)
                        else:
                            StepMotor.move_backward(self.sm_x, 35)
                    elif ch == "\n":
                        print('sdsdsdsdsd')

            except (KeyboardInterrupt, EOFError):
                pass


    @staticmethod
    def move_forward(motor, steps):
        """
        Moves the stepper motor forward the specified number of steps.
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.DOUBLE) #INTERLEAVE

    @staticmethod
    def move_backward(motor, steps):
        """
        Moves the stepper motor backward the specified number of steps
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)

    def __turn_off_motors(self):
        """
        Recommended for auto-disabling motors on shutdown!
        :return:
        """
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

if __name__ == "__main__":
    '''
    s = StepMotor()
    print('input commend (w) or (s) or (a) or (d)')
    s.interactive()
    
    #or
    '''
    s = StepMotor()
    print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == "q":
                    break

                if ch == "w":
                    s.up()
                elif ch == "s":
                    s.down()
                elif ch == "a":
                    s.left()
                elif ch == "d":
                    s.right()
                elif ch == "\n":
                    print('sdsdsdsdsd')
                wiringpi.pwmWrite(18, pulse)
                print(pulse)
        except (KeyboardInterrupt, EOFError):
            pass
        



