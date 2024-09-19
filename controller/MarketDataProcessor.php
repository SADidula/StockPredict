<?php

use PSpell\Dictionary;

class MarketDataProcessor
{
    // Define public variables
    private $_dir_, $coin_name, $init_time, $query_limit, $interval;
    private $params = array();

    // adding a construct to initialize class variables
    function __construct($_dir_, $coin_name, $init_time, $query_limit, $interval)
    {
        $this->_dir_ = $_dir_;
        $this->coin_name = $coin_name;
        $this->init_time = $init_time;
        $this->query_limit = $query_limit;
        $this->interval = $interval;
    }

    // save the market values 
    function update_market_data()
    {
        $this->params = $this->construct_params();
        $req_data = $this->call_api("GET", "https://api.binance.com/api/v3/klines", $this->params); // return json string

        $processed_data = [];
        $req_arr = json_decode($req_data, true);

        for ($i = 0; $i < sizeof($req_arr); $i++) {
            $processed_data[$i]['Timestamp'] = $req_arr[$i][0];
            $processed_data[$i]['Open'] = $req_arr[$i][1];
            $processed_data[$i]['High'] = $req_arr[$i][2];
            $processed_data[$i]['Low'] = $req_arr[$i][3];
            $processed_data[$i]['Close'] = $req_arr[$i][4];
            $processed_data[$i]['Volume_(Coin)'] = $req_arr[$i][5];
            $processed_data[$i]['Volume_(Currency)'] = $req_arr[$i][7];
            // $processed_data[$i]['Weighted_Price'] = $this->calculate_weighted_price($req_arr[$i][5], $req_arr[$i][7], $req_arr[$i][8]);
        }

        $this->save_data($this->_dir_, $this->coin_name . ".json", $processed_data); //convert to array and save
    }

    function calculate_weighted_price($high, $low, $close, $volume)
    {
        if ($volume == 0)
            return 0;

        return 0;
    }

    // construct the parameters
    function construct_params()
    {
        // use default values for the parameter initialization
        $file_name = $this->_dir_ . $this->coin_name . '.json';
        $start_time = $this->init_time;

        // if file exist then change the start time of the api request
        if (file_exists($file_name)) {

            $current_data = $this->retrieve_file($file_name);
            $start_time = end($current_data)["Timestamp"];
        }

        $params = array(
            "symbol" => $this->coin_name,
            "startTime" => $start_time,
            "limit" => $this->query_limit,
            "interval" => $this->interval
        );

        return $params;
    }

    // grab json data from an api
    function call_api($method, $url, $data = array())
    {
        $curl = curl_init();

        switch ($method) {
            case "POST":
                curl_setopt($curl, CURLOPT_POST, 1);

                if ($data)
                    curl_setopt($curl, CURLOPT_POSTFIELDS, $data);
                break;

            case "PUT":

                curl_setopt($curl, CURLOPT_PUT, 1);

                break;

            default:
                if ($data)
                    $url = sprintf("%s?%s", $url, http_build_query($data));
        }

        // Optional Authentication:
        curl_setopt($curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);

        curl_setopt($curl, CURLOPT_URL, $url);
        curl_setopt($curl, CURLOPT_RETURNTRANSFER, 1);

        $result = curl_exec($curl);

        curl_close($curl);

        return $result;
    }

    // save data on request
    function save_data($_dir_, $file_name, $raw_data)
    {
        if (file_exists($_dir_ . $file_name)) {

            $current_data = $this->retrieve_file($_dir_ . $file_name);
            $this->file_write_data($_dir_, $file_name, $current_data, $raw_data);
        } else {

            $this->file_create_data($_dir_, $file_name, $raw_data);
        }
    }

    // retrieve data from an existing file
    function retrieve_file($file_name)
    {
        $current_data = file_get_contents($file_name);
        $array_data = json_decode($current_data, true);
        return $array_data;
    }

    // write data to an existing file
    function file_write_data($_dir_, $file_name, $current_data, $raw_data)
    {
        $first_elem_time_curr_arr = $current_data[0]["Timestamp"];
        $first_elem_time_raw_arr = $raw_data[0]["Timestamp"];

        $last_elem_time_curr_arr = end($current_data)["Timestamp"];

        if ($first_elem_time_curr_arr == $first_elem_time_raw_arr)
            return;

        if ($first_elem_time_raw_arr == $last_elem_time_curr_arr) {
            array_shift($raw_data);
        }

        $merge_data = array_merge($current_data, $raw_data);

        // append only the difference
        file_put_contents($_dir_ . $file_name, json_encode($merge_data));
    }

    // write data to a new file
    function file_create_data($_dir_, $file_name, $raw_data)
    {
        $file = fopen($_dir_ . $file_name, "w+");
        fclose($file);
        file_put_contents($_dir_ . $file_name, json_encode($raw_data));
    }
}
