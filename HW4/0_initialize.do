
********************************************************************************
* HW4
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************

* set PATH *********************************************************************
* jingyuan's path
if c(username) == "jingyuanwang" | c(username) == "Jingyuan" {
    * mac
    if regexm(c(os),"Mac") == 1 {
        global proj_folder = "/Users/jingyuanwang/Dropbox/Course/ECON/Optimization/IEMS450_HW/HW4"
        global Git = "/Users/jingyuanwang/GitHub/IEMS450_HW/HW4"
    }    
    * win
    else if regexm(c(os),"Windows") == 1 {
        ***
    } 
}


global data = "$proj_folder/data"
global results = "$proj_folder/results"

********************************************************************************
* I. install packages
********************************************************************************

