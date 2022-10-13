---

title: qt/qml profiler
tags: ['qml', 'profiler', 'engine']

---

# 使用

```bash
qmlprofiler /usr/bin/testapp
```

or 

```bash
/usr/bin/testapp -qmljsdebugger=port:12345

# another terminal

qmlprofiler -a localhsot -p 12345

```

# diagram



```plantuml
@startuml
class QJSEnginePrivate {
    addToDebugServer(QJSEngine *q)
}

QJSEnginePrivate <|-- QQmlEnginePrivate
QQmlEnginePrivate::profiler *-> QQmlProfiler
class QQmlEnginePrivate {
    profiler : QQmlProfiler
}

class QJSEngine {
    evaluate(program, fileName, lineNumber)
    newObject()
    newArray(length)
    newQObject(QObject *o)
    newQMetaObject()
}

QJSEngine <|-- QQmlEngine
class QQmlEngine {
    rootContext();
    setImageProvider();
    setUrlInterpreter()
    createContextForObject()
    setObjectOwnersip()
}

QQmlProfilerDefinitions <|-- QQmlAbstractProfilerAdaptor
QQmlProfilerDefinitions <|-- QQmlProfiler
QQmlProfilerDefinitions <|-- QQmlProfilerHelper
QQmlProfilerHelper <|-- QQmlBindingProfiler
QQmlProfilerHelper <|-- QQmlHandlingSignalProfiler
QQmlProfilerHelper <|-- QQmlCompilingProfiler
QQmlProfilerHelper <|-- QQmlVmeProfiler
QQmlProfilerHelper <|-- QQmlObjectCreationProfiler
QQmlProfilerHelper <|-- QQmlObjectCompletionProfiler

QQmlBindingProfiler::profiler *-> QQmlProfiler
QQmlHandlingSignalProfiler::profiler *-> QQmlProfiler
QQmlCompilingProfiler::profiler *-> QQmlProfiler
QQmlVmeProfiler::profiler *-> QQmlProfiler
QQmlObjectCreationProfiler::profiler *-> QQmlProfiler
QQmlObjectCompletionProfiler::profiler *-> QQmlProfiler

class QQmlAbstractProfilerAdaptor {
    startProfiling(features)
    stopProfiling()
    reportData(bool trackLocations)
    startService(QQmlProfilerService service)
    sendMessages()
}

class QQmlProfiler {
    startBinding(binding, function)
    startCompiling()
    startHandlingSignal()
    startCreating()
    m_data
}

class QQmlMemoryProfiler
@enduml

```