# KHR_parallel_shader_compile

Name

    KHR_parallel_shader_compile

Name Strings

    GL_KHR_parallel_shader_compile

Contact

    Geoff Lang, (geofflang 'at' google.com)

Contributors

    Timothy Lottes, AMD
    Graham Sellers, AMD
    Eric Werness, NVIDIA
    Geoff Lang, Google
    Daniel Koch, NVIDIA

Notice

    Copyright (c) 2015 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL and OpenGL ES Working Groups. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete

Version

    Last Modified Date: 2017-04-24
    Revision: 2

Number

    ARB Extension #192
    OpenGL ES Extension #288

Dependencies

    This extension is written against OpenGL 4.5 (CoreProfile) dated
    May 28 2015.

    OpenGL ES 2.0 is required (for mobile).

Overview

    Compiling GLSL into implementation-specific code can be a time consuming
    process, so a GL implementation may wish to perform the compilation in a
    separate CPU thread. This extension provides a mechanism for the application
    to provide a hint to limit the number of threads it wants to be used to
    compile shaders, as well as a query to determine if the compilation process
    is complete.

New Procedures and Functions

    void MaxShaderCompilerThreadsKHR(uint count);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, GetFloatv, and GetDoublev:

        MAX_SHADER_COMPILER_THREADS_KHR   0x91B0

    Accepted as part of the <pname> parameter to GetShaderiv and
    accepted as part of the <pname> parameter to GetProgramiv:

        COMPLETION_STATUS_KHR             0x91B1

IP Status

    None.

Additions to Chapter 7 "Programs and Shaders",

    Append to the end of 7.1 "Shader Objects",

        Applications may use the following to hint to the driver the maximum
    number background threads it would like to be used in the process of
    compiling shaders or linking programs,

        void MaxShaderCompilerThreadsKHR(uint count);

    where <count> is the number of background threads. A <count> of zero
    specifies a request for no parallel compiling or linking and a <count> of
    0xFFFFFFFF requests an implementation-specific maximum.

    An implementation may combine the maximum compiler thread request from
    multiple contexts in a share group in an implementation-specific way.

    An application can query the current MaxShaderCompilerThreadsKHR <count>
    by calling GetIntegerv with <pname> set to MAX_SHADER_COMPILER_THREADS_KHR,
    which returns the value of the current state (Table 23.51).


    Add to 7.13 "Shader, Program, and Program Pipeline Queries" under the
    descriptions for "pname" for "GetShaderiv",

        If <pname> is COMPLETION_STATUS_KHR, TRUE is returned if the shader
        compilation has completed, FALSE otherwise.

    Add to 7.13 "Shader, Program, and Program Pipeline Queries" under the
    descriptions for "pname" for "GetProgramiv",

        If <pname> is COMPLETION_STATUS_KHR, TRUE is returned if the program
        linking has completed, FALSE otherwise.

New State

    Add to Table 23.51: Hints
    Get Value                        Type  Get Command   Initial Value  Description           Sec
    -------------------------------  ----  ------------  -------------  --------------------  ----
    MAX_SHADER_COMPILER_THREADS_KHR  Z+    GetIntegerv   0xFFFFFFFF     Max compile threads   7.13

    Add to Table 23.32: Program Object State
    Get Value               Type  Get Command   Initial Value  Description           Sec
    ----------------------  ----  ------------  -------------  --------------------  ----
    COMPLETION_STATUS_KHR   B     GetProgramiv  TRUE           Program linking has   7.13
                                                               completed

    Add to Table 23.30: Shader Object State
    Get Value               Type  Get Command   Initial Value  Description           Sec
    ---------------------   ----  ------------  -------------  --------------------  ----
    COMPLETION_STATUS_KHR   B     GetShaderiv   TRUE           Shader compilation    7.13
                                                               has completed

Interactions with OpenGL ES

    If implemented in OpenGL ES ignore all references to GetDoublev.

    If the supported ES version is less than 3.0, ignore all references to
    GetInteger64v.

Issues

    1) Where should the hint state be stored?

    UNRESOLVED: Each context has its own value which may be specified and
    queried, but an implementation may choose to combine the hints from multiple
    contexts in an implmentation-specific manner. There isn't really any
    precedent for per-share group state.

    2) Can we make the requirements more strict?

    RESOLVED: We could, but making sure all of the error behavior is correct and
    fully specified would likely take more time than we have. This spec allows
    an application to clearly request its intent even if there aren't guarantees
    that the implementation will exactly obey the request.

    3) Does glGetIntegerv(MAX_SHADER_COMPILER_THREADS_KHR) just return the
    value set by MaxShaderCompilerThreadsKHR? Or, if the state is 0xFFFFFFFF
    ("do something implementation specific"), does it return the number of
    threads the implementation has actually chosen to use?

    RESOLVED: As with other state queries, this returns the value that was last
    set, or if no value was set by the application it returns the default state
    value (0xFFFFFFFF).

Revision History

    Rev  Date        Author    Changes
    ---  ----------  --------  ---------------------------------------------
      1  2017-03-23  glang     Cast as KHR based on v6 of
                               ARB_parallel_shader_compile.
      2  2017-04-24  dgkoch    Spec clarifications, add issue 3.
