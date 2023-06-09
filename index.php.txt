<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Joanna's Personal HomePage</title>
    <link rel="stylesheet" href="index-style.css">
	
</head>

<body>

<div class="jo">
<h1>Joanna Hughes</h1>
<img src="joannahughes.jpeg" alt="joanna">

<p class="p1">Hello! <br> My name is Joanna and I am currently 
studying computer science and computer engineering.</p>
</div>

<div class="jo">
<p style="color:DodgerBlue;"><b>Graduate Classes Taken</b></p>

 <ul style="color:DodgerBlue;">
  <li>Graph Theory</li>
  <li>Computational Biology</li>
  <li>Web Applications</li>
</ul> 
</div>
​
<div class="jo">
<p class="italic"><b>Projects</b></p>

<dl>
  <dt>Automated Wheelchair</dt>
  <dd>- Developing a prototype of an autonomous wheelchair. 
  The goal is that the wheelchair will follow lines and QR codes 
  in order to reduce the number of people necessary to help as well 
  as creating an easy way to avoid collisions with bystanders</dd>
  <dt>Carbon Emmissions of Electric Vehicles</dt>
  <dd>- Research and make connections about the carbon 
  emissions during mining, shipping, and building vehicles 
  that are ‘clean energy.’</dd>
</dl>
</div>

<div class="jo">
<p style="color:Tomato;"><b>Personal Hobbies and Interests</b></p>
 <ul style="color:Tomato;">
  <li>Swimming</li>
  <li>Drawing</li>
  <li>Jewelry making</li>
  <li>Cooking</li>
  <li>Crafting</li>
</ul> 
</div>


<!DOCTYPE html>
<html>
<body>

<?php



$d=strtotime("today");
$year=date("Y", $d);
$month=date("m", $d);
$day=date("d", $d);
$dpm = 31;
$d=mktime(0,0,0,$month, 1, $year);

switch ($month) {
  case "01":
    $Nmonth= "JANUARY ";
    break;
  case "02":
    $Nmonth= "FEBRUARY ";
    $dmp=27;
    break;
  case "03":
    $Nmonth= "MARCH ";
    break;
  case "04":
    $Nmonth= "APRIL ";
    $dmp = 30;
    break;
  case "05":
    $Nmonth= "MAY ";
    break;
  case "06":
    $Nmonth= "JUNE ";
    $dmp = 30;
    break;
  case "07":
    $Nmonth= "JULY ";
    break;
  case "08":
    $Nmonth= "AUGUST ";
    break;
  case "09":
    $Nmonth= "SEPTEMBER ";
    $dmp = 30;
    break;
  case "10":
    $Nmonth= "OCTOBER ";
    break;
  case "11":
    $Nmonth= "NOVEMBER ";
    $dmp = 30;
    break;
  case "12":
    $Nmonth= "DECEMBER ";
    break;
  default:
 	$Nmonth= "ERROR ";
    break;
    }


print "<h2>" . $Nmonth .  $year . "</h2>";


echo "&ensp;S &ensp;M &ensp;T &ensp;W &ensp;T &ensp;F &ensp;S<br>";

$num = 0;
$firstDate= date("l",$d);

switch ($firstDate) {
 case "Sunday":
    $num = 0;
    break;
  case "Monday":
    $num = 1;
    break;
  case "Tuesday":
    $num = 2;
    break;
  case "Wednesday":
    $num = 3;
    break;
  case "Thursday":
    $num = 4;
    break;
  case "Friday":
    $num = 5;
    break;
  case "Saturday":
    $num = 6;
    break;
   default:
    break;
  }
  
    
    $temp =1;
    for ($i = 0; $i < 5; $i++) {
        for ($j = 0; $j < 7; $j++) {
            if ($num > 0) {
                echo "&emsp;&ensp;";
                $num--;
            }
            else if ($temp < $dpm) {
                if ($temp < 10) echo "&ensp;";
                if ($temp == $day) {
                    echo "<strong>" . $temp . "&ensp;</strong>";
                }
                else {
                    echo  $temp . "&ensp;";
                }
                $temp++;
            }
        }
        echo "<br>";
    }
    

?>

</body>
</html>




<div class="jo">
<a href="" onmouseover="document.bgColor='turquoise'">D</a>
<a href="" onmouseover="document.bgColor='pink'">r</a>
<a href="" onmouseover="document.bgColor='blue'">a</a>
<a href="" onmouseover="document.bgColor='red'">g</a>
<a href="" onmouseover="document.bgColor='yellow'">Y</a>
<a href="" onmouseover="document.bgColor='green'">o</a>
<a href="" onmouseover="document.bgColor='white'">u</a>
<a href="" onmouseover="document.bgColor='teal'">r</a>
<a href="" onmouseover="document.bgColor='olivedrab'">M</a>
<a href="" onmouseover="document.bgColor='green'">o</a>
<a href="" onmouseover="document.bgColor='seagreen'">u</a>
<a href="" onmouseover="document.bgColor='magenta'">s</a>
<a href="" onmouseover="document.bgColor='fusia'">e</a>
<a href="" onmouseover="document.bgColor='purple'">O</a>
<a href="" onmouseover="document.bgColor='navy'">v</a>
<a href="" onmouseover="document.bgColor='green'">e</a>
<a href="" onmouseover="document.bgColor='white'">r</a>
<a href="" onmouseover="document.bgColor='royalblue'">Me</a>
<a href="" onmouseover="document.bgColor='Skyblue'">To</a>
<a href="" onmouseover="document.bgColor='almond'">C</a>
<a href="" onmouseover="document.bgColor='coral'">h</a>
<a href="" onmouseover="document.bgColor='brown'">a</a>
<a href="" onmouseover="document.bgColor='almond'">n</a>
<a href="" onmouseover="document.bgColor='coral'">g</a>
<a href="" onmouseover="document.bgColor='olivedrab'">e</a>
<a href="" onmouseover="document.bgColor='teal'">B</a>
<a href="" onmouseover="document.bgColor='pink'">a</a>
<a href="" onmouseover="document.bgColor='blue'">ck</a>
<a href="" onmouseover="document.bgColor='red'">g</a>
<a href="" onmouseover="document.bgColor='yellow'">r</a>
<a href="" onmouseover="document.bgColor='green'">ou</a>
<a href="" onmouseover="document.bgColor='olivedrab'">nd</a>
<a href="" onmouseover="document.bgColor='black'">color!</a>



<p style="font-size:48px">&#128512;</p>
</div>

</body>

</html>