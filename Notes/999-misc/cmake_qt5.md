---

title: cmake_qt5

---

# Get started with CMake

Start with find_package to locate the libraries and header files shipped with Qt. Then, you can use these libraries and header files with the target_link_libraries command to build Qt-based libraries and applications. This command automatically adds the appropriate include directories, compile definitions, the position-independent-code flag, and links to the qtmain.lib library on Windows, for example.

## Build a GUI executable

To build a helloworld GUI executable, you need the following:

```cmake

cmake_minimum_required(VERSION 3.1.0)

project(helloworld VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(helloworld
        mainwindow.ui
        mainwindow.cpp
        main.cpp
        resources.qrc
        )

target_link_libraries(helloworld Qt5::Widgets)

```

For find_package to be successful, CMake must find the Qt installation in one of the following ways:

Set your CMAKE_PREFIX_PATH environment variable to the Qt 5 installation prefix. This is the recommended way.
Set the Qt5_DIR in the CMake cache to the location of the Qt5Config.cmake file.
The CMAKE_AUTOMOC setting runs moc automatically when required. For more details, see CMake AUTOMOC documentation.

## Imported library targets
Each Qt module that is loaded defines a CMake library target. The target names start with Qt5::, followed by the module name. For example: Qt5::Core, Qt5::Gui. Pass the name of the library target to target_link_libraries to use the respective library.

Note: Since Qt 5.15, the CMake targets are also available as Qt::Core, Qt::Gui, and so on. This eases writing CMake code that can work with both Qt 5 and Qt 6.

Imported targets are created with the same configurations as when Qt was configured. That is:

If Qt was configured with the -debug switch, an imported target with the DEBUG configuration is created.
If Qt was configured with the -release switch, an imported target with the RELEASE configuration is created.
If Qt was configured with the -debug-and-release switch, then imported targets are created with both RELEASE and DEBUG configurations.
If your project has custom CMake build configurations, you have to map your custom configuration to either the debug or the release Qt configuration.

```cmake
find_package(Qt5 COMPONENTS Core REQUIRED)

set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

# set up a mapping so that the Release configuration for the Qt imported target is
# used in the COVERAGE CMake configuration.
set_target_properties(Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")

```


# qt5 command

CMake Command Reference

## Qt5::Core

### [qt5_add_big_resources](https://doc.qt.io/qt-5.15/qtcore-cmake-qt5-add-big-resources.html)

Compiles big binary resources into object code

```cmake
qt5_add_big_resources(<VAR> file1.qrc [file2.qrc ...]
                      [OPTIONS ...])
```

Creates compiled object files from Qt resource files using the Resource Compiler (rcc). Paths to the generated files are added to <VAR>.

This is similar to qt5_add_resources / qt_add_resources, but directly generates object files (.o, .obj) files instead of C++ source code. This allows to embed bigger resources, where compiling to C++ sources and then to binaries would be too time consuming or memory intensive.

```cmake
set(SOURCES main.cpp)
qt5_add_big_resources(SOURCES big_resource.qrc)
add_executable(myapp ${SOURCES})
```

For compatibility with Qt 6, the command is also available under the name qt_add_big_resources.

### [qt5_add_binary_resources]

Creates an RCC file from a list of Qt resource files

```cmake
qt5_add_binary_resources(target file1.qrc [file2.qrc ...]
                         [DESTINATION ...]
                         [OPTIONS ...])

```

Adds a custom target that compiles Qt resource files into a binary .rcc file.

Note: For compatibility with Qt 6, the command is also available under the name qt_add_binary_resources.

DESTINATION sets the path of the generated .rcc file. The default is ${CMAKE_CURRENT_BINARY_DIR}/${target}.rcc.

You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

Examples

```cmake

qt5_add_binary_resources(resources project.qrc OPTIONS -no-compress)
add_dependencies(myapp resources)
```

### [qt5_add_resources](https://doc.qt.io/qt-5.15/qtcore-cmake-qt5-add-resources.html)

Compiles binary resources into source code

Synopsis
```cmake
qt5_add_resources(<VAR> file1.qrc [file2.qrc ...]
        [OPTIONS ...])

```
Description
You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

For embedding bigger resources, see qt5_add_big_resources.

Note: For compatibility with Qt 6, the command is also available under the name qt_add_resources.

Arguments
You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

Examples

```cmake

set(SOURCES main.cpp)
qt5_add_resources(SOURCES example.qrc)
add_executable(myapp ${SOURCES})
```

### [qt5_generate_moc](https://doc.qt.io/qt-5.15/qtcore-cmake-qt5-generate-moc.html)

Calls moc on an input file

Synopsis
```cmake

qt5_generate_moc(src_file dest_file
        [TARGET target])
```

Description
Creates a rule to call the Meta-Object Compiler (moc) on src_file and store the output in dest_file.

Note: This is a low-level macro. See the CMake AUTOMOC Documentation for a more convenient way to let source files be processed with moc. qt5_wrap_cpp is also similar, but automatically generates a temporary file path for you.

Note: For compatibility with Qt 6, the command is also available under the name qt_generate_moc.

Arguments
You can set an explicit TARGET. This will make sure that the target properties INCLUDE_DIRECTORIES and COMPILE_DEFINITIONS are also used when scanning the source files with moc.

Examples
```cmake
qt5_generate_moc(main.cpp main.moc TARGET myapp)
```

### [qt5_import_plugins](https://doc.qt.io/qt-5.15/qtcore-cmake-qt5-import-plugins.html)

Specifies a custom set of plugins to import for a static Qt build

Synopsis

```cmake
qt5_import_plugins(target
        [INCLUDE plugin ...]
        [EXCLUDE plugin ...]
        [INCLUDE_BY_TYPE plugin_type plugin ...]
        [EXCLUDE_BY_TYPE plugin_type])

```

Description
Specifies a custom set of plugins to import. The optional arguments: INCLUDE, EXCLUDE, INCLUDE_BY_TYPE, and EXCLUDE_BY_TYPE, can be used more than once.

This CMake command was introduced in Qt 5.14.

INCLUDE -- can be used to specify a list of plugins to import.
EXCLUDE -- can be used to specify a list of plugins to exclude.
INCLUDE_BY_TYPE -- can be used to override the list of plugins to import for a certain plugin type.
EXCLUDE_BY_TYPE -- can be used to specify a plugin type to exclude; then no plugins of that type are imported.
Qt provides plugin types such as imageformats, platforms, and sqldrivers.

If the command isn't used the target automatically links against a sane set of default plugins, for each Qt module that the target is linked against. For more information, see target_link_libraries.

Each plugin comes with a C++ stub file that automatically initializes the plugin. Consequently, any target that links against a plugin has this C++ file added to its SOURCES.

Note: This command imports plugins from static Qt builds only. On shared builds, it does nothing.

Note: For compatibility with Qt 6, the command is also available under the name qt_import_plugins.

Examples
```cmake

add_executable(myapp main.cpp)
target_link_libraries(myapp Qt5::Gui Qt5::Sql)
qt5_import_plugins(myapp
INCLUDE Qt5::QCocoaIntegrationPlugin
EXCLUDE Qt5::QMinimalIntegrationPlugin
INCLUDE_BY_TYPE imageformats Qt5::QGifPlugin Qt5::QJpegPlugin
EXCLUDE_BY_TYPE sqldrivers
)
```

In the snippet above, the following occurs with the executable myapp:

The Qt5::QCocoaIntegrationPlugin is imported into myapp.
The Qt5::QMinimalIntegrationPlugin plugin is excluded from being automatically imported into myapp.
The default list of plugins for imageformats is overridden to only include Qt5::QGifPlugin and Qt5::QJpegPlugin.
All sqldrivers plugins are excluded from automatic importing.

### [qt5_wrap_cpp](https://doc.qt.io/qt-5.15/qtcore-cmake-qt5-wrap-cpp.html)

Creates .moc files from sources

Synopsis
```cmake

qt5_wrap_cpp(<VAR> src_file1 [src_file2 ...]
[TARGET target]
[OPTIONS ...]
[DEPENDS ...])
```

Description
Creates rules for calling the Meta-Object Compiler (moc) on the given source files. For each input file, an output file is generated in the build directory. The paths of the generated files are added to <VAR>.

Note: This is a low-level macro. See the CMake AUTOMOC Documentation for a more convenient way to let source files be processed with moc.

Note: For compatibility with Qt 6, the command is also available under the name `qt_wrap_cpp`.

Arguments
You can set an explicit TARGET. This will make sure that the target properties INCLUDE_DIRECTORIES and COMPILE_DEFINITIONS are also used when scanning the source files with moc.

You can set additional OPTIONS that should be added to the moc calls. You can find possible options in the moc documentation.

DEPENDS allows you to add additional dependencies for recreation of the generated files. This is useful when the sources have implicit dependencies, like code for a Qt plugin that includes a .json file using the Q_PLUGIN_METADATA() macro.

Examples

```cmake
set(SOURCES myapp.cpp main.cpp)
qt5_wrap_cpp(SOURCES myapp.h)
add_executable(myapp ${SOURCES})
```

### [qt_add_big_resources]

Compiles big binary resources into object code

#### Synopsis

```cmake
qt_add_big_resources(<VAR> file1.qrc [file2.qrc ...]
        [OPTIONS ...])
```

#### Description

Creates compiled object files from Qt resource files using the Resource Compiler (rcc). Paths to the generated files are added to <VAR>.

This is similar to qt5_add_resources / qt_add_resources, but directly generates object files (.o, .obj) files instead of C++ source code. This allows to embed bigger resources, where compiling to C++ sources and then to binaries would be too time consuming or memory intensive.

Note: This macro is only available if using CMake 3.9 or later.

This command was introduced in Qt 5.15. You can use qt5_add_big_resources in older versions of Qt.

#### Arguments

You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

#### Examples

```cmake
set(SOURCES main.cpp)
qt_add_big_resources(SOURCES big_resource.qrc)
add_executable(myapp ${SOURCES})
```

### [qt_add_binary_resources]

Creates an RCC file from a list of Qt resource files

### [qt_add_resources]

Compiles binary resources into source code

### [qt_generate_moc]

Calls moc on an input file

### [qt_import_plugins]

Specifies a custom set of plugins to import for a static Qt build

### [qt_wrap_cpp]

Creates .moc files from sources

## Qt5::DBus

### [qt_add_dbus_adaptor](https://doc.qt.io/qt-5.15/qtdbus-cmake-qt-add-dbus-adaptor.html)

Generates an adaptor class for a D-Bus interface

Synopsis
```cmake

# added in Qt 5.15
qt_add_dbus_adaptor(<VAR> dbus_spec header
        [parent_class]
        [basename]
        [classname]
        )

qt5_add_dbus_adaptor(<VAR> dbus_spec header
        [parent_class]
        [basename]
        [classname]
        )

```

#### Description
Generates a C++ header file implementing an adaptor for a D-Bus interface description file defined in dbus_spec. The path of the generated file is added to <VAR>. The generated adaptor class takes a pointer to parent_class as QObject parent. parent_class should be declared in header, which is included in the generated code as #include "header".

The function sets up a call to the Qt D-Bus XML compiler (qdbusxml2cpp) in adaptor mode. The default file and class name are generated from the last segment in the dbus_spec base name:

XML file	Header file	Class name
org.example.chat	chatadaptor.h	ChatAdaptor
You can change the name of the header file to be generated by passing basename as the fifth argument. The .h suffix is always added.

You can change the default class name by passing classname as the sixth argument.

#### Examples
```cmake
qt_add_dbus_adaptor(GENERATED_SOURCES org.example.chat.xml chat.h ChatMainWindow)
```

### [qt_add_dbus_interface](https://doc.qt.io/qt-5.15/qtdbus-cmake-qt-add-dbus-interface.html)

Generates C++ sources implementing an interface for a D-Bus interface description file

#### Synopsis

```cmake
qt_add_dbus_interface(<VAR> dbus_spec basename)  # added in Qt 5.15

qt5_add_dbus_interface(<VAR> dbus_spec basename)
```

#### Description

Generates C++ sources implementing an interface for a D-Bus interface description file defined in dbus_spec. The generated files are named after basename: basename.h, basename.cpp, basename.moc. The paths of the files are added to <VAR>.

The function sets up a call to the Qt D-Bus XML compiler (qdbusxml2cpp) in interface (proxy) mode. By default, qdbusxml2cpp generates a C++ class named after the interface name, with a namespaced alias:

| D-Bus  Interface Name	 | Class name               | 	Namespaced name  |
|------------------------|--------------------------|-------------------|
| org.example.chat       | 	OrgExampleChatInterface | 	org.example.chat |

#### Options

Options can be set using set_source_file_property on the dbus_spec:

| Option	       | Value        | 	Description                                               |
|---------------|--------------|------------------------------------------------------------|
| CLASSNAME     | 	class_name	 | Override the default interface class name with class_name. |
| NO_NAMESPACE	 | boolean	     | Do not generate the namespaced name if set to ON.          |
| INCLUDE	      | path	        | Add an #include "path" in the generated code.              |


### [qt_add_dbus_interfaces](https://doc.qt.io/qt-5.15/qtdbus-cmake-qt-add-dbus-interfaces.html)

Generates C++ sources implementing interfaces for D-Bus interface description files

#### Synopsis
```cmake
qt_add_dbus_interfaces(<VAR> dbus_spec1 [dbus_spec2 ...])  # added in Qt 5.15
qt5_add_dbus_interfaces(<VAR> dbus_spec1 [dbus_spec2 ...])
```

#### Description

Generates C++ sources implementing D-Bus interfaces defined in dbus_spec1, dbus_spec2, where each argument needs to be the path to a valid D-Bus interface description file. The paths of the generated files are added to `<VAR>`.

For each argument, a call to the Qt D-Bus XML compiler (qdbusxml2cpp) in interface (proxy) mode is set up.

The generated C++ source files are named after the XML file: For the file org.example.chat.xml the generated header will be named orgexamplechatinterface.h.

#### Options

Options can be set using `set_source_file_property` on each of the file arguments:

| Option        | 	Value	     | Description                                                |
|---------------|-------------|------------------------------------------------------------|
| CLASSNAME	    | class_name	 | Override the default interface class name with class_name. |
| NO_NAMESPACE	 | boolean	    | Do not generate the namespaced name if set to ON.          |
| INCLUDE	      | path	       | Add an #include "path" in the generated code.              |

### [qt_generate_dbus_interface](https://doc.qt.io/qt-5.15/qtdbus-cmake-qt-generate-dbus-interface.html)

Generates a D-Bus interface from a header file

#### Synopsis

```cmake
# added in Qt 5.15
qt_generate_dbus_interface(header
        [customName]
        [OPTIONS options]
        )

qt5_generate_dbus_interface(header
        [customName]
        [OPTIONS options]
        )

```

#### Description

Parses the C++ source or header file containing a QObject-derived class declaration and generates a file containing the D-BUS Introspection XML.

By default, the generated XML file is stored in the current binary directory, and has the same base name as the header. You can specify a different name or path by adding customName as an optional second argument.

#### Options

The function sets up a call to the qdbuscpp2xml command line tool. Further arguments to the tool can be set after OPTIONS.

## Qt5::LinguistTools

### [qt5_add_translation](https://doc.qt.io/qt-5.15/qtlinguist-cmake-qt5-add-translation.html)

Compiles Qt Linguist .ts files into .qm files

#### Synopsis

```cmake
qt5_add_translation(<VAR> file1.ts [file2.ts ...]
        [OPTIONS ...])
```

#### Description

Calls `lrelease` on each .ts file passed as an argument, generating .qm files. The paths of the generated files are added to `<VAR>`.

#### Options

You can set additional `OPTIONS` that should be passed when `lrelease` is invoked. You can find possible options in the [lrelease documentation](https://doc.qt.io/qt-5.15/linguist-manager.html#lrelease).

By default, the qm files will be placed in the root level of the build directory. To change this, you can set `OUTPUT_LOCATION` as a property of the source .ts file.

#### Examples

Generating helloworld_en.qm, helloworld_de.qm in the build directory:

```cmake
qt5_add_translation(qmFiles helloworld_en.ts helloworld_de.ts)
```

Generating helloworld_en.qm, helloworld_de.qm in a l10n sub-directory:

```cmake
set(TS_FILES helloworld_en.ts helloworld_de.ts)
set_source_files_properties(${TS_FILES} PROPERTIES OUTPUT_LOCATION "l10n")
qt5_add_translation(qmFiles ${TS_FILES})
```

### [qt5_create_translation](https://doc.qt.io/qt-5.15/qtlinguist-cmake-qt5-create-translation.html)


Sets up the Qt Linguist translation toolchain

#### Synopsis

```cmake
qt5_create_translation(<VAR> ts-file-or-sources [ts-file-or-sources2 ...]
        [OPTIONS ...])
```

#### Description

Processes given sources (directories or individual files) to generate Qt Linguist .ts files. The .ts files are in turn compiled into .qm files of the same base name that are stored in the build directory. Paths to the generated .qm files are added to `<VAR>`.

The translation files to create or update need to have a .ts suffix. If the given file path is not absolute it is resolved relative to the current source directory. If no .ts file is passed as an argument, the macro does nothing.

Any arguments that do not have a .ts suffix are passed as input to the `lupdate`. `lupdate` accepts directories and source files as input. See also the [lupdate documentation](https://doc.qt.io/qt-5.15/linguist-manager.html#lupdate) on further details.

#### Options

You can set additional OPTIONS that should be passed when lupdate is invoked. You can find possible options in the lupdate documentation.

#### Examples

Recursively look up Qt translations from source files in current directory and generate or update helloworld_en.ts and helloworld_de.ts file using lupdate. Compile said files into helloworld_en.qm and helloworld.de.qm files in the build directory:

```cmake
qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} helloworld_en.ts helloworld_de.ts)
```

## Qt5::RemoteObjects

### [qt5_generate_repc](https://doc.qt.io/qt-5.15/qtremoteobjects-cmake-qt5-generate-repc.html)

Creates a C++ type from a Qt Remote Objects .rep file

#### Synopsis

```cmake
qt5_generate_repc(<VAR> rep_file output_type)
```

#### Description

Creates rules for calling repc on rep_file. output_type must be either SOURCE or REPLICA. The paths of the generated files are added to <VAR>.

Depending on the output_type argument, the generated code will either implement a Qt Remote Objects Source or a Qt Remote Objects Replica type in C++.

#### Examples

```cmake

find_package(Qt5 COMPONENTS RemoteObjects REQUIRED)

set(SOURCES
        main.cpp
        simpleswitch.cpp
        )

qt5_generate_repc(SOURCES simpleswitch.rep SOURCE)

add_executable(directconnectserver ${SOURCES})
target_link_libraries(directconnectserver Qt5::RemoteObjects)

```

## Qt5::Widgets

### [qt5_wrap_ui](https://doc.qt.io/qt-5.15/qtwidgets-cmake-qt5-wrap-ui.html)

Creates sources for .ui files

#### Synopsis

```cmake

qt5_wrap_ui(<VAR> ui_file1 [ui_file2 ...]
        [OPTIONS ...])
```

#### Description

Creates rules for calling the User Interface Compiler (uic) on the given .ui files. For each input file, an header file is generated in the build directory. The paths of the generated header files are added to `<VAR>`.

Note: This is a low-level macro. See the CMake AUTOUIC Documentation for a more convenient way to process .ui files with uic.

Note: For compatibility with Qt 6, the command is also available under the name qt_wrap_ui.

#### Options

You can set additional OPTIONS that should be added to the uic calls. You can find possible options in the uic documentation.

#### Examples

```cmake
set(SOURCES mainwindow.cpp main.cpp)
qt5_wrap_ui(SOURCES mainwindow.ui)
add_executable(myapp ${SOURCES})
```

### [qt_wrap_ui](https://doc.qt.io/qt-5.15/qtwidgets-cmake-qt-wrap-ui.html)

Creates sources for .ui files

#### Synopsis

```cmake
qt_wrap_ui(<VAR> ui_file1 [ui_file2 ...]
        [OPTIONS ...])
```

#### Description

Creates rules for calling the User Interface Compiler (uic) on the given .ui files. For each input file, an header file is generated in the build directory. The paths of the generated header files are added to `<VAR>`.

Note: This is a low-level macro. See the CMake AUTOUIC Documentation for a more convenient way to process .ui files with uic.

This command was introduced in Qt 5.15. You can use qt5_wrap_ui in older versions of Qt.

#### Options

You can set additional OPTIONS that should be added to the uic calls. You can find possible options in the uic documentation.

#### Examples

```cmake
set(SOURCES mainwindow.cpp main.cpp)
qt_wrap_ui(SOURCES mainwindow.ui)
add_executable(myapp ${SOURCES})
```

# CMake Variable Reference

## Module variables

Qt modules loaded with find_package set various variables.

Note: You rarely need to access these variables directly. Common tasks like linking against a module should be done through the library targets each module defines.

For example, `find_package(Qt5 COMPONENTS Widgets)`, when successful, makes the following variables available:

| Variable	                            | Description                                                                                              |
|--------------------------------------|----------------------------------------------------------------------------------------------------------|
| Qt5Widgets_COMPILE_DEFINITIONS	      | A list of compile definitions to use when building against the library.                                  |
| Qt5Widgets_DEFINITIONS	              | A list of definitions to use when building against the library.                                          |
| Qt5Widgets_EXECUTABLE_COMPILE_FLAGS	 | A string of flags to use when building executables against the library.                                  |
| Qt5Widgets_FOUND	                    | A boolean that describes whether the module was found successfully.                                      |
| Qt5Widgets_INCLUDE_DIRS              | 	A list of include directories to use when building against the library.                                 |
| Qt5Widgets_LIBRARIES	                | The name of the imported target for the module: Qt5::Widgets                                             |
| Qt5Widgets_PRIVATE_INCLUDE_DIRS	     | A list of private include directories to use when building against the library and using private Qt API. |
| Qt5Widgets_VERSION_STRING	           | A string containing the module's version.                                                                |

For all packages found with find_package, equivalents of these variables are available; they are case-sensitive.

## Installation variables

Additionally, there are also variables that don't relate to a particular package, but to the Qt installation itself.

| Variable	                           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| QT_DEFAULT_MAJOR_VERSION	           | An integer that controls the Qt version that qt_ commands forward to in case of mixed Qt 5 and Qt 6 projects. It needs to be set to either 5 or 6 before the respective find_package() calls.  If set to 5, commands starting with qt_ will call their counterpart starting with qt5_. If set to 6, they will call their counterpart starting with qt6_. If not set, the first find_package call defines the default version. This functionality was added in Qt 5.15. |
| QT_LIBINFIX	                        | A string that holds the infix used in library names, when Qt is configured with -libinfix.                                                                                                                                                                                                                                                                                                                                                                             |
| QT_NO_CREATE_VERSIONLESS_FUNCTIONS	 | Since Qt 5.15, modules define not only commands that start with qt5_, but also ones with qt_. You can set QT_NO_CREATE_VERSIONLESS_FUNCTIONS before find_package to prevent this.                                                                                                                                                                                                                                                                                      |
| QT_NO_CREATE_VERSIONLESS_TARGETS	   | Since Qt 5.15, modules define not only targets that start with Qt5::, but also ones with Qt::. You can set QT_NO_CREATE_VERSIONLESS_TARGETS before find_package to prevent this.                                                                                                                                                                                                                                                                                       |
| QT_VISIBILITY_AVAILABLE	            | On Unix, a boolean that describes whether Qt libraries and plugins were compiled with -fvisibility=hidden. This means that only selected symbols are exported.                                                                                                                                                                                                                                                                                                         |

# qt6 command
