# QCOM_extended_get2

Name

    QCOM_extended_get2

Name Strings

    GL_QCOM_extended_get2

Contributors

    Jukka Liimatta
    James Ritts

Contact

    Jukka Liimatta (jukka.liimatta 'at' qualcomm.com)

Notice

    Copyright Qualcomm 2009.

IP Status

    Qualcomm Proprietary.

Status

    Complete.

Version

    Last Modified Date: October 30, 2009
    Revision: #2

Number

    OpenGL ES Extension #63

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0 specification.

    Requires extension QCOM_extended_get to be implemented.   

Overview

    This extension enables instrumenting the driver for debugging of OpenGL ES 
    applications.

New Procedures and Functions

    void ExtGetShadersQCOM(uint* shaders, int maxShaders, int* numShaders);

    void ExtGetProgramsQCOM(uint* programs, int maxPrograms, 
                            int* numPrograms);

    boolean ExtIsProgramBinaryQCOM(uint program);

    void ExtGetProgramBinarySourceQCOM(uint program, enum shadertype, 
                                       char* source, int* length)

Additions to OpenGL ES 2.0 Specification

    The command

        void ExtGetShadersQCOM(uint* shaders, int maxShaders, int* numShaders);

    returns list of shader objects in the current render context.

    The command

        void ExtGetProgramsQCOM(uint* programs, int maxPrograms, int* numPrograms);

    returns list of program objects in the current render context.

    The command

        boolean ExtIsProgramBinaryQCOM(uint program);

    returns boolean indicating if the program is created with ProgramBinaryOES.

    The command

        void ExtGetProgramBinarySourceQCOM(uint program, enum shadertype, char* source, int* length)

    returns source string, if any exists, for program created with ProgramBinaryOES.

Errors

    INVALID_VALUE error will be generated if the <program> parameter to
    ExtIsProgramBinaryQCOM does not reference to a valid program object.

    INVALID_VALUE error will be generated if the <program> parameter to
    ExtGetProgramBinarySourceQCOM does not reference to a valid program object.

    INVALID_ENUM error will be generated if the <shadertype> parameter to
    ExtGetProgramBinarySourceQCOM is not one of the allowable values.

New State

    None.

Revision History

    #01    05/14/2009    Jukka Liimatta       First draft.
    #02    10/30/2009    Jon Leech            Make ExtIsProgramBinaryQCOM
                                              return boolean (Khronos bug
                                              5705).
