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

## qml singleton

```qml
pragma Singleton  
import QtQuick 2.0  
  
Item {  
    readonly property string colourBlue: "blue"  
    readonly property string colourRed: "red"  
    readonly property int fontPointSize: 16  
    
    property Action firstAction
    firstAction : Action {
        name: "xxx"
    }
}  
```

```C++
qmlRegisterSingletonType( QUrl("file:///absolute/path/MyStyleObject.qml"), "MyNameSpace", 1, 0, "MySingletonItem" );

```

使用：
```qml
import MyNameSpace 1.0

Button {
        action:MySingletonItem.firstAction
}
```

## Qt5 GDB pretty printer

```bash
 if [[ ! -d ~/.gdb/qt5printers ]]; then mkdir -p ~/.gdb/qt5printers; cd ~/.gdb; git clone https://github.com/Lekensteyn/qt5printers; cd -; fi
 
 cat > ~/.gdbinit << EOM
set auto-load local-gdbinit on
add-auto-load-safe-path /
python
import sys, os.path
sys.path.insert(0, os.path.expanduser('~/.gdb'))
import qt5printers
qt5printers.register_printers(gdb.current_objfile())
end
EOM
```


