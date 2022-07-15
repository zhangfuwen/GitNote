# qt & qml

## qml http get json

```javascript
function httpGet(keyword) {
        var http = new XMLHttpRequest();
        var url = "https://so.toutiao.com/search?source=input&keyword=" + keyword + "&format=raw_json";
        var params = "num=22&num2=333";
        http.open("GET", url, true);
        // Send the proper header information along with the request
        http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        // http.setRequestHeader("Content-length", params.length);
        http.setRequestHeader("Connection", "close");
        console.error("http get");
        http.onreadystatechange = function() {
            // Call a function when the state changes.
            if (http.readyState == 4) {
                if (http.status == 200) {
                    console.error("ok");
                    // console.error(http.responseJSON.data);
                    var jsonResponse = JSON.parse(http.responseText);
                    console.error("data length " + jsonResponse.data.length);
                    searchEngineListModel.clear();
                    for (var i = 0; i < jsonResponse.data.length; i++) {
                        let d = jsonResponse.data[i];
                        if (d.hasOwnProperty('emphasized')) {
                            console.error("emphasized " + d.emphasized);
                            if (d.emphasized.hasOwnProperty('title')) {
                                searchEngineListModel.append({
                                    "name": d.emphasized.title,
                                    "desc": d.emphasized.summary
                                });
                                console.error("title: " + d.emphasized.title);
                                console.error("summary: " + d.emphasized.summary);
                            } else {
                                console.error("data " + i + " has no field named emphasized.title");
                            }
                        } else {
                            console.error("data " + i + " has no field named emphasized");
                        }
                    }
                } else {
                    console.log("error: " + http.status);
                }
            }
        };
        http.send(params);
    }
```

## 