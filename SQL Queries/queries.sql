use footballathletesfacerecognition;

CREATE TABLE feedbacks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(50),
  feedback TEXT,
  time_added DATETIME
);

SELECT * FROM feedbacks;

TRUNCATE TABLE feedbacks;

CREATE TABLE nopredictionsinfo (
  id INT AUTO_INCREMENT PRIMARY KEY,
  uploaded_image_url VARCHAR(255) NOT NULL,
  get_prediction VARCHAR(10) NOT NULL,
  time_added DATETIME
);

SELECT * FROM nopredictionsinfo;

TRUNCATE TABLE nopredictionsinfo;

CREATE TABLE predictionsinfo (
  id INT AUTO_INCREMENT PRIMARY KEY,
  uploaded_image_url VARCHAR(255) NOT NULL,
  predicted_football_athlete_name VARCHAR(50) NOT NULL,
  get_expected_prediction VARCHAR(10) NOT NULL,
  time_added DATETIME
);

SELECT * FROM predictionsinfo;

TRUNCATE TABLE predictionsinfo;
