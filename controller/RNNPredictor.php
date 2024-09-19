<?php 

$command = escapeshellcmd('../brain/RNN_Predictor_v2.py');
$output = shell_exec($command);
echo $output;

?>