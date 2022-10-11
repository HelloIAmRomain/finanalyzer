/*
 * Copyright (c) Finanalyzer 2022, 2022
 * All rights reserved.
 */

 -- Create user
CREATE USER 'finanalyzer'@'localhost' IDENTIFIED BY 'ThisIsAReallyLongPasswordForMySQLDatabase';

-- Create database
CREATE DATABASE IF NOT EXISTS `finanalyzer`;

-- Switch to the database
USE finanalyzer;

-- Add a new table
CREATE TABLE IF NOT EXISTS `newsletter` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `email` VARCHAR(64) NOT NULL,
  PRIMARY KEY (`id`));


