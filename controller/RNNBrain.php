<?php

// $command = escapeshellcmd('../brain/RNN_Brain_v2.py');
// $output = shell_exec($command);
// echo $output;

$command = '../brain/RNN_Brain_v2.py';
exec($command);
