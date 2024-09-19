<?php

class MarketDataSaver
{
    // Define public variables
    private $_dir_, $coin_name, $init_time, $query_limit;
    private $params = array();

    // adding a construct to initialize class variables
    function __construct($_dir_, $coin_name, $init_time, $query_limit)
    {
        $this->_dir_ = $_dir_;
        $this->coin_name = $coin_name;
        $this->init_time = $init_time;
        $this->query_limit = $query_limit;
    }

    // save the market values 
    function update_market_data()
    {
        $this->params = $this->construct_params();
        $req_data = $this->call_api("GET", "https://api.binance.com/api/v3/aggTrades", $this->params); // return json string
        $this->save_data($this->_dir_, $this->coin_name . ".json", json_decode($req_data, true)); //convert to array and save
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
            $start_time = end($current_data)["T"];
        }

        $params = array(
            "symbol" => $this->coin_name,
            "startTime" => $start_time,
            "limit" => $this->query_limit
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
        if (file_exists($_dir_.$file_name)) {

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
        $first_elem_time_curr_arr = $current_data[0]["T"];
        $first_elem_time_raw_arr = $raw_data[0]["T"];

        $last_elem_time_curr_arr = end($current_data)["T"];

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
        $file = fopen($_dir_.$file_name, "w+");
        fclose($file);
        file_put_contents($_dir_ . $file_name, json_encode($raw_data));
    }
}
