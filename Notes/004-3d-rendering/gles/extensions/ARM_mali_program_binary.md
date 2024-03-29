# ARM_mali_program_binary

Name

    ARM_mali_program_binary

Name Strings

    GL_ARM_mali_program_binary

Contributors

    Jan-Harald Fredriksen, ARM
    Tom Olson, ARM

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Shipping as of August 2012.

Version

    Last Modified Date: June 26, 2015

Number

    OpenGL ES Extension #120

Dependencies

    OpenGL ES 2.0 is required.
    OES_get_program_binary is required.

    This extension is written based on the wording of the OpenGL ES 2.0.25
    specification and the OES_get_program_binary extension.

Overview

    The OES_get_program_binary extension enables applications to retrieve program
    binaries using GetProgramBinaryOES and reload them using ProgramBinaryOES.

    The mechanism for retrieval and reloading of program binaries is vendor
    agnostic, but the binary format itself is vendor specific.

    This extension adds a token to identify program binaries that are
    compatible with the ARM Mali family of GPUs.

Issues

    1. When should applications recompile and relink program binaries?

    UNRESOLVED: ProgramBinaryOES may fail after a driver update. In this
    case, it may be useful for applications to know whether all program
    binaries need to be recompiled and/or relinked. There is no language
    mechanism other than the program object info log to provide this
    information to the application. However, it is expected that if any
    binary fails to load after a driver update, then all programs binaries
    retrieved prior to the driver update will fail to load.

New Procedures and Functions

    None

New Tokens

    Accepted by the <binaryFormat> parameter of ProgramBinaryOES:

        MALI_PROGRAM_BINARY_ARM              0x8F61

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    At the end of the section called Program Binaries, add:

    "The format MALI_PROGRAM_BINARY_ARM will be present in the list of
    program binary formats returned when querying PROGRAM_BINARY_FORMATS_OES.
    This format will be returned in <binaryFormat> by GetProgramBinaryOES, and
    may be used as the <binaryFormat> in calls to ProgramBinaryOES.

    ProgramBinaryOES may reject a MALI_PROGRAM_BINARY_ARM program binary. This
    can happen if <binary> is not a valid Mali program binary, if <binary> has
    been compiled for an incompatible Mali GPU, if <binary> has been compiled
    for a different API version, or if <binary> has been produced by an
    incompatible version of the shader compiler or driver. If <binary> is
    rejected for any of these reasons, the LINK_STATUS will be set to FALSE
    and the program object's info log will be updated with a reason for the
    rejection.

    If <binary> was rejected because it was produced by an incompatible version
    of the shader compiler or driver, applications should recompile and relink
    all programs produced with the previous version of the shader compiler and
    driver."

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State Requests)

    None

Errors

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    #1  21/03/2012   Jan-Harald Fredriksen    First draft.
    #2  26/03/2012   Jan-Harald Fredriksen    Completed internal review.
    #3  24/06/2012   Jan-Harald Fredriksen    Clarified behavior for incompatible binaries.
    #4  01/09/2012   Jan-Harald Fredriksen    Updated status.
    #5  25/06/2015   Jan-Harald Fredriksen    Fixed typo.
