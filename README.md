# Attendance System
## Project Description
This project aims to create a facial recognition system that allows a doctor to take attendance using an app. The system uses a retrained FaceNet architecture as a feature extraction model to understand the similarity among face features. Then, a small neural network is trained using these features. Finally, the system is deployed using Flask API and HTML pages in the templates folder. When the doctor takes an image using the app, the attendance is taken, and the attendance sheet is sent to the doctor.

## Installation
To install this project, follow these steps:

1. Clone the project repository using `git clone https://github.com/space-codes/attendance-system.git`
2. Navigate to the project directory using `cd attendance-system`
3. Create a virtual environment using `python -m venv env`
4. Activate the virtual environment using `source env/bin/activate` on macOS or Linux, or `.\env\Scripts\activate` on Windows.
5. Install the required packages using pip install -r `src/requirements.txt`

## Usage
To use this project, follow these steps:

1. Activate the virtual environment using `source env/bin/activate` on macOS or Linux, or `.\env\Scripts\activate` on Windows.
2. Navigate to the `src` directory using `cd src`
3. Run the Flask app using `python app.py`
4. Open a web browser and navigate to `http://localhost:5000` to access the app.
