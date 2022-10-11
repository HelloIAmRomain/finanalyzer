<!--
  ~ Copyright (c) Finanalyzer 2022, 2022
  ~ All rights reserved.
  -->

<!doctype html>

<html>
<head>
  <meta charset="utf-8">
  <title>Finanalyzer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Form sublit">
  <meta name="author" content="Finanalyzer">
  <!-- Bootstrap CSS Link -->
  <link href="./dist/css/vendor/bootstrap.min.css" rel="stylesheet">

  <!-- Theme CSS Link -->
  <link href="./dist/css/main.css" rel="stylesheet">

  <!-- Favicon Link -->
  <link rel="shortcut icon" href="./dist/favicon.ico">
</head>

<body>
<div class="container">
  <div class="row">
    <div class="col-md-12">
      <div class="page-header">
        <h1>Finanalyzer</h1>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="col-md-12">
      <div class="panel panel-default">
        <div class="panel-heading">
          <h3 class="panel-title">Newsletter</h3>
        </div>
        <div class="panel-body">
          <p>
            Thank you for subscribing to our newsletter.
          </p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- mySQL connection -->
<?php
$servername = "127.0.0.1";
$username = "root";
$password = "DoNotUseThisPasswordItIsNotSupposedToBeUsedThankYouForYourCooperationAndHaveAGreatDay";
$dbname = "finanalyzer";
$port = 3306;
$conn = new mysqli($servername, $username, $password, $dbname);
$email = $_POST['email'];
// Check connection
if ($conn->connect_error) {
  echo "Failed to connect to MySQL: " . $conn->connect_error;
  exit();
} else {
  echo "Connected successfully";
}
// make sure the email is valid
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
  echo "Invalid email format";
  echo "<br>";
  echo $email;
  exit;
}
// insert the email into the table `newsletter` if it doesn't already exist safely (no injection)
$sql = "INSERT INTO finanalyzer.newsletter(email) VALUES ('" . $conn->real_escape_string($_POST['email']) . "')";

if ($conn->query($sql) === FALSE) {
  echo "Error: " . $sql . "<br>" . $conn->error;
}
$conn->close();

?>
