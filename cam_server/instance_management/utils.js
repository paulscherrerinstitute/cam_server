function appendTableRow(table, rowData) {
  var lastRow = $('<tr/>').appendTo(table.find('tbody:last'));
  $.each(rowData, function(colIndex, c) {
      lastRow.append($('<td/>').html(c));
  });

  return lastRow;
}

function setTableCell(value, tableName, row, col) {
    var cell = $('#' + tableName + ' tbody tr:eq(' + row  + ') td:eq(' + col  + ')')[0]
    cell.innerHTML = value
}


function getRows(table){
    var rows = table.find('tbody').find('tr').length
    return rows
}

function clear(table){
    //return $('#servers_table tr').length -1;
    table.find("tbody").empty()
}

function isDict(v) {
    return typeof v==='object' && v!==null && !(v instanceof Array) && !(v instanceof Date);
}

function isArray(v) {
    return (v instanceof Array)
}

function dictToStr(dict, indent){
        var nested = true
        var ret = ""
        if (indent == undefined ){
            indent = "";
            ret = "<font size='2'><pre>"
            nested = false
        }

        for (var key in dict) {
            var value = dict[key]
            if (isDict (value)){
                ret = ret + key + ":" + "\n"
                ret = ret + (dictToStr(value, /*"\t"*/ "    "))
            } else {
                ret = ret + indent + key + ": " + /*JSON.stringify*/(value) + "\n"
            }
        }
        if (!nested){
            ret = ret+"</pre></font>"
        }
        return ret
}

function listToStr(lst, multiline, indent) {
    if (indent == undefined) {
        indent = "";
    }
    var separator = ', '
    var ret = ""
    if (multiline) {
        ret = "<font size='2'><pre>"
        separator = "\n"
    }
    if (isArray(lst)) {
        for (var i = 0; i < lst.length; i++) {
            ret = ret + indent + lst[i]
            if (i< lst.length-1) {
                ret = ret + separator
            }
        }
    } else {
        ret = ret + indent + lst + separator
    }
    if (multiline) {
        ret = ret+"</pre></font>"
    }
    return ret
}

function dateToStr(date){
        //ret = "<pre>"
        //ret = ret + date.split(" ").join("\n")x
        //ret = ret+"</pre>"

        //return date.split(" ").join("\n\t")

    return date.split(" ").slice(0,2).join(" ")
}


function formatNum(n, force_decimals){
    if (force_decimals == undefined) {
        force_decimals = true
    }
    if (n) {
        if (n >= 1e12) {
            return (n / 1e12).toFixed(2) + "T"
        } else if (n >= 1e9) {
            return (n / 1e9).toFixed(2) + "G"
        } else if (n >= 1e6) {
            return (n / 1e6).toFixed(2) + "M"
        } else if (n >= 1e3) {
            return (n / 1e3).toFixed(2) + "K"
        } else if (!Number.isInteger(n) || force_decimals) {
            return (n).toFixed(2)
        }
    }
    return n
}
