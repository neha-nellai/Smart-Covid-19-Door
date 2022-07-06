Smart Covid 19 Door
====================
This was my submission for the 'AI FOR SOCIAL GOOD' hackathon organized by IEEE and CDAC. I have implemented a Smart Covid 19 Door.  A Small-scale prototype of a smart door using a Raspberry Pi 3B+ was built.  The MLX90614 infrared contactless temperature sensor has been used to perform contactless temperature detection. A Pi camera is used to detect the live video feed of the user and perform face-mask detection using a deep learning model. A PIR sensor has been used to detect the motion of the user, in order to decide when to open the door. A hand sanitizer system is also integrated with this assembly by employing an Ultrasonic sensor. A LCD display is used to guide the user throughout the process by displaying appropriate messages as and when needed. Temperature data is sent to ThingSpeak cloud for future analysis.

Directory Layout
====================
Smart COvid 19 Door directory structure looks as follows::

    E-commerce-Website-Django/
        |---SmartCovidDoor.py
        |--FCED EL.pptx
        |--FCSD_EL report.pdf
        |----readme.md

Extra
==========
Hardware and circuit connection details are explained in the power point presentation and report, along with block diagrams and flowcharts for better understanding of the project.
