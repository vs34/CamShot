drop database if exists camshot;
create database camshot;

DROP TABLE IF EXISTS StudentCount;
CREATE TABLE Studentcount (
  people_count int NOT NULL,
  entry_time datetime,
  log_id int NOT NULL auto_increment,
  
  PRIMARY KEY (log_id)
) ;

DROP TABLE IF EXISTS Studentcount;
CREATE TABLE Studentcount (
  student_id int NOT NULL,
  scan_time datetime,
  log_id int NOT NULL auto_increment,
  
  foreign key (student_id) references StudentData(student_id),
  PRIMARY KEY (log_id)
) ;

DROP TABLE IF EXISTS StudentData;
CREATE TABLE StudentData (
  student_id varchar(50),
  student_name varchar(50),
  
  PRIMARY KEY (student_id)
) ;
