---

title: cmake_qt6

---

# CMake Command Reference

## Qt6::Core

### [qt_add_big_resources](https://doc.qt.io/Qt-6/qt-add-bigresources.html)

Compiles big binary resources into object code.

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
```

This command was introduced in Qt 5.12.

#### Synopsis

```cmake
qt_add_big_resources(<VAR> file1.qrc [file2.qrc ...]
        [OPTIONS ...])

```

If versionless commands are disabled, use qt6_add_big_resources() instead. It supports the same set of arguments as this command.

#### Description

Creates compiled object files from Qt resource files using the Resource Compiler (rcc). Paths to the generated files are added to <VAR>.

This is similar to qt_add_resources, but directly generates object files (.o, .obj) files instead of C++ source code. This allows to embed bigger resources, where compiling to C++ sources and then to binaries would be too time consuming or memory intensive.

#### Arguments

You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

#### Examples

```cmake
set(SOURCES main.cpp)
qt_add_big_resources(SOURCES big_resource.qrc)
add_executable(myapp ${SOURCES})
```

### [qt_add_binary_resources](https://doc.qt.io/Qt-6/qt-add-binary-resources.html)

Creates an RCC file from a list of Qt resource files.

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
```

This command was introduced in Qt 5.10.

#### Synopsis

```cmake
qt_add_binary_resources(target file1.qrc [file2.qrc ...]
        [DESTINATION ...]
        [OPTIONS ...])
```

If versionless commands are disabled, use qt6_add_binary_resources() instead. It supports the same set of arguments as this command.

#### Description

Adds a custom target that compiles Qt resource files into a binary .rcc file.

#### Arguments

DESTINATION sets the path of the generated .rcc file. The default is ${CMAKE_CURRENT_BINARY_DIR}/${target}.rcc.

You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

#### Examples

```cmake
qt_add_binary_resources(resources project.qrc OPTIONS -no-compress)
add_dependencies(myapp resources)
```


### [qt_add_executable](https://doc.qt.io/Qt-6/qt-add-executable.html)

Creates and finalizes an application target of a platform-specific type

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
```
This command was introduced in Qt 6.0.

#### Synopsis

```cmake
qt_add_executable(target
        [WIN32] [MACOSX_BUNDLE]
        [MANUAL_FINALIZATION]
        sources...)

```

If versionless commands are disabled, use qt6_add_executable() instead. It supports the same set of arguments as this command.

#### Description

This command performs the following tasks:

Create a CMake target of the appropriate type for the target platform.
Link the target to the Qt::Core library.
Handle finalization of the CMake target.

##### Target Creation

On all platforms except Android, an executable target will be created. All arguments will be passed through to the standard CMake add_executable() command, except MANUAL_FINALIZATION (if present). On Android, a MODULE library will be created and any WIN32 or MACOSX_BUNDLE options will be ignored. Some target properties will also be set for Android:

The SUFFIX target property will be set to give the library file name an architecture-specific suffix.
Various <lang>_VISIBILITY_PRESET target properties will be set to default to ensure that the main() function is visible in the resultant binary.

##### Linking Qt::Core

Since all Qt applications need to link to the Qt::Core library, this is done for you as a convenience.

##### Finalization
After a target is created, further processing or finalization steps are commonly needed. The steps to perform depend on the platform and on various properties of the target. The finalization processing is implemented by the qt_finalize_target() command. You might need to also call the qt_finalize_project() command at the end of top-level CMakeLists.txt to correctly resolve the dependencies between project targets.

Finalization can occur either as part of this call or be deferred to sometime after this command returns (but it should still be in the same directory scope). When using CMake 3.19 or later, finalization is automatically deferred to the end of the current directory scope. This gives the caller an opportunity to modify properties of the created target before it is finalized. When using CMake versions earlier than 3.19, automatic deferral isn't supported. In that case, finalization is performed immediately before this command returns.

Regardless of the CMake version, the MANUAL_FINALIZATION keyword can be given to indicate that you will explicitly call qt_finalize_target() yourself instead at some later time. In general, MANUAL_FINALIZATION should not be needed unless the project has to support CMake 3.18 or earlier.

#### Examples

In the following simple case, finalization is handled automatically. If using a CMake version earlier than 3.19, finalization will be performed immediately as part of the call. When using CMake 3.19 or later, finalization will occur at the end of the current directory scope.

```cmake
qt_add_executable(simpleapp main.cpp)
```

The following example shows a scenario where finalization must be deferred. The OUTPUT_NAME target property affects deployment settings on Android, but those settings are written out as part of finalizing the target. In order to support using CMake versions earlier than 3.19, we take over responsibility for finalizing the target by adding the MANUAL_FINALIZATION keyword.

```cmake
qt_add_executable(complexapp MANUAL_FINALIZATION complex.cpp)
set_target_properties(complexapp PROPERTIES OUTPUT_NAME Complexify)
qt_finalize_target(complexapp)
```

Warning: If your Android project is built using a CMake version lower than 3.19, make sure that you call qt6_finalize_project() at the end of a top-level CMakeLists.txt.

### [qt_add_library](https://doc.qt.io/Qt-6/qt-add-library.html)

Creates and finalizes a library

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
```

This command was introduced in Qt 6.2.

#### Synopsis

```cmake
qt_add_library(target
        [STATIC | SHARED | MODULE | INTERFACE | OBJECT]
        [MANUAL_FINALIZATION]
        sources...
        )
```

If versionless commands are disabled, use qt6_add_library() instead. It supports the same set of arguments as this command.

#### Description

qt_add_library() is a wrapper around CMake's built-in add_library() command. It performs the following tasks:

Create a CMake target of an appropriate library type.
Handle finalization of the CMake target.
Target Creation
The type of library created can be specified explicitly with one of the STATIC, SHARED, MODULE, INTERFACE or OBJECT keywords, just as one might for add_library(). If none of these keywords are given, the library type created depends on how Qt was built. If Qt was built statically, a static library will be created. Otherwise, a shared library will be created. Note that this is different to how CMake's add_library() command works, where the BUILD_SHARED_LIBS variable controls the type of library created. The qt_add_library() command does not consider BUILD_SHARED_LIBS when deciding the library type.

Any sources provided will be passed through to the internal call to add_library().

Finalization
After a target is created, further processing or finalization steps may be needed. The finalization processing is implemented by the qt_finalize_target() command.

Finalization can occur either as part of this call or be deferred to sometime after this command returns (but it should still be in the same directory scope). When using CMake 3.19 or later, finalization is automatically deferred to the end of the current directory scope. This gives the caller an opportunity to modify properties of the created target before it is finalized. When using CMake versions earlier than 3.19, automatic deferral isn't supported. In that case, finalization is performed immediately before this command returns.

Regardless of the CMake version, the MANUAL_FINALIZATION keyword can be given to indicate that you will explicitly call qt_finalize_target() yourself instead at some later time. In general, MANUAL_FINALIZATION should not be needed unless the project has to support CMake 3.18 or earlier.

### [qt_add_plugin](https://doc.qt.io/Qt-6/qt-add-plugin.html)

Creates a Qt plugin target

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
```

This command was introduced in Qt 6.0.

#### Synopsis

```cmake
qt_add_plugin(target
        [SHARED | STATIC]
        [CLASS_NAME class_name]
        [OUTPUT_TARGETS variable_name]
        )
```

If versionless commands are disabled, use qt6_add_plugin() instead. It supports the same set of arguments as this command.

#### Description

Qt plugin targets have additional requirements over and above an ordinary CMake library target. The qt_add_plugin() command adds the necessary handling to ensure these requirements are met. It should be called rather than the built-in CMake add_library() command when defining a Qt plugin target.

By default, the plugin will be created as a STATIC library if Qt was built statically, or as a MODULE library otherwise. You can override this default by explicitly providing the STATIC or SHARED option.

Note: Non-static plugins are meant to be loaded dynamically at runtime, not linked to at build time. CMake differentiates between these two scenarios by providing the MODULE library type for dynamically loaded libraries, and the SHARED library type for libraries that may be linked to directly. This distinction is important for some toolchains (notably Visual Studio), due to the way symbol exports are handled. It may not be possible to link to MODULE libraries, and generating a SHARED library with no exported symbols can result in build-time errors. If the SHARED option is passed to qt_add_plugin(), it will therefore create a MODULE library rather than a SHARED library.

Every Qt plugin has a class name. By default, this will be the same as the target, but it can be overridden with the CLASS_NAME option. The class name corresponds to the name of the C++ class that declares the metadata for the plugin. For static plugins, it is also the name passed to Q_IMPORT_PLUGIN, which imports the plugin into an application and ensures it is available at run time.

If the plugin is built statically, qt_add_plugin() may define additional internal targets. These facilitate automatic importing of the plugin for any executable or shared library that links to the plugin. If the project installs the plugin and intends to make it available for other projects to link to, the project should also install these internal targets. The names of these targets can be obtained by providing the OUTPUT_TARGETS option, followed by the name of a variable in which to return the target list.


#### [qt_add_resources](https://doc.qt.io/Qt-6/qt-add-resources.html)

Compiles binary resources into source code

The command is defined in the Core component of the Qt6 package, which can be loaded like so:

```cmake

find_package(Qt6 REQUIRED COMPONENTS Core)
```

#### Synopsis

````cmake
qt_add_resources(<VAR> file1.qrc [file2.qrc ...]
        [OPTIONS ...])

````

If versionless commands are disabled, use qt6_add_resources() instead. It supports the same set of arguments as this command.

Since 6.0:

```cmake
qt_add_resources(<TARGET> <RESOURCE_NAME>
        [PREFIX <PATH>]
        [LANG <LANGUAGE>]
        [BASE <PATH>]
        [OUTPUT_TARGETS <VARIABLE_NAME>]
        [FILES ...] [OPTIONS ...])
```

If versionless commands are disabled, use qt6_add_resources() instead. It supports the same set of arguments as this command.

#### Description
To add resources, you can pass either a variable name or a target as the first argument of the command.

When passing a variable name as first argument, qt_add_resources creates source code from Qt resource files using the Resource Compiler (rcc). Paths to the generated source files are added to <VAR>.

When passing a target as first argument, the function creates a resource with the name RESOURCE_NAME, containing the specified FILES. The resource is automatically linked into TARGET.

For embedding bigger resources, see qt_add_big_resources.

See The Qt Resource System for a general description of Qt resources.

Arguments of the target-based variant
PREFIX specifies a path prefix under which all files of this resource are accessible from C++ code. This corresponds to the XML attribute prefix of the .qrc file format. If PREFIX is not given, the target property QT_RESOURCE_PREFIX is used.

LANG specifies the locale of this resource. This corresponds to the XML attribute lang of the .qrc file format.

BASE is a path prefix that denotes the root point of the file's alias. For example, if BASE is "assets" and FILES is "assets/images/logo.png", then the alias of that file is "images/logo.png".

Alias settings for files need to be set via the QT_RESOURCE_ALIAS source file property.

When using this command with static libraries, one or more special targets will be generated. Should you wish to perform additional processing on these targets, pass a variable name to the OUTPUT_TARGETS parameter. The qt_add_resources function stores the names of the special targets in the specified variable.

Arguments of both variants
You can set additional OPTIONS that should be added to the rcc calls. You can find possible options in the rcc documentation.

#### Examples
Variable variant, using a .qrc file:

```cmake
set(SOURCES main.cpp)
qt_add_resources(SOURCES example.qrc)
add_executable(myapp ${SOURCES})
```

Target variant, using immediate resources:

```cmake
add_executable(myapp main.cpp)
qt_add_resources(myapp "images"
        PREFIX "/images"
        FILES image1.png image2.png)

```

#### Caveats

When adding multiple resources, RESOURCE_NAME must be unique across all resources linked into the final target.

This especially affects static builds. There, the same resource name in different static libraries conflict in the consuming target.

### qt_allow_non_utf8_sources

Prevents forcing source files to be treated as UTF-8 for Windows

### qt_android_add_apk_target

Defines a build target that runs androiddeployqt to produce an APK

### qt_android_apply_arch_suffix

Configures the target binary's name to include an architecture-specific suffix

### qt_android_generate_deployment_settings

Generates the deployment settings file needed by androiddeployqt

### qt_deploy_qt_conf

Write a qt.conf file at deployment time

### qt_deploy_runtime_dependencies

Deploy Qt plugins, Qt and non-Qt libraries needed by an executable

### qt_disable_unicode_defines

Prevents some unicode-related compiler definitions being set automatically on a target

### qt_extract_metatypes

Extracts metatypes from a Qt target and generates an associated metatypes.json file

### qt_finalize_project

Handles various common platform-specific tasks associated with Qt project

### qt_finalize_target

Handles various common platform-specific tasks associated with Qt targets

### qt_generate_deploy_app_script

Generate a deployment script for an application

### qt_generate_moc

Calls moc on an input file

### [qt_import_plugins](https://doc.qt.io/Qt-6/qt-import-plugins.html)

Specifies a custom set of plugins to import for a static Qt build

qt_set_finalizer_mode

Customizes aspects of a target's finalization

qt_standard_project_setup

Setup project-wide defaults to a standard arrangement

qt_wrap_cpp

Creates .moc files from sources

Qt6::DBus
qt_add_dbus_adaptor

Generates an adaptor class for a D-Bus interface

qt_add_dbus_interface

Generates C++ sources implementing an interface for a D-Bus interface description file

qt_add_dbus_interfaces

Generates C++ sources implementing interfaces for D-Bus interface description files

qt_generate_dbus_interface

Generates a D-Bus interface from a header file

Qt6::LinguistTools
qt_add_lrelease

Add targets to transform Qt Linguist .ts files into .qm files

qt_add_lupdate

Add targets to generate or update Qt Linguist .ts files

qt_add_translation

Compiles Qt Linguist .ts files into .qm files

qt_add_translations

Add targets to update and transform Qt Linguist .ts files into .qm files

qt_create_translation

Sets up the Qt Linguist translation toolchain

Qt6::Qml
qt_add_qml_module

Defines a QML module

qt_add_qml_plugin

Defines a plugin associated with a QML module

qt_deploy_qml_imports

Deploy the runtime components of QML modules needed by an executable

qt_generate_deploy_qml_app_script

Generate a deployment script for a QML application

qt_generate_foreign_qml_types

Registers types from one target in a QML module

qt_import_qml_plugins

Ensures QML plugins needed by a target are imported for static builds

qt_query_qml_module

Retrieve information about a QML module

qt_target_compile_qml_to_cpp

Compiles QML files (.qml) to C++ source code with qmltc

qt_target_qml_sources

Add qml files and resources to an existing QML module target

Qt6::RemoteObjects
qt_add_repc_merged

Creates C++ header files for source and replica types from the Qt Remote Objects .rep files

qt_add_repc_replicas

Creates C++ header files for replica types from the Qt Remote Objects .rep files

qt_add_repc_sources

Creates C++ header files for source types from the Qt Remote Objects .rep files

qt_reps_from_headers

Creates .rep files from the QObject header files

Qt6::Scxml
qt_add_statecharts

Qt6::Widgets
qt_wrap_ui

Creates sources for .ui files

Qt6::WebEngineCore
qt_add_webengine_dictionary

Converts the hunspell dictionary format into bdict binary format