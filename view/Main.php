<?php

include('../controller/MarketDataSaver.php');
include('../controller/MarketDataProcessor.php');

$_dir_ = "../model/";
$init_time = 946684801;
$query_limit = 1000;
$interval = "1m";

//update bitcoin market values
// $bit_coin_usdt = new MarketDataSaver($_dir_, "BTCUSDT", $init_time, $query_limit);
// $bit_coin_usdt->update_market_data();

$bit_coin_usdt = new MarketDataProcessor($_dir_, "BTCUSDT", $init_time, $query_limit, $interval);
$bit_coin_usdt->update_market_data();

//update bitcoin market values
$xlm_coin_usdt = new MarketDataProcessor($_dir_, "XLMUSDT", $init_time, $query_limit, $interval);
$xlm_coin_usdt->update_market_data();

?>