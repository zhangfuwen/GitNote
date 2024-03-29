# QCOM_driver_control

Name

    QCOM_driver_control

Name Strings

    GL_QCOM_driver_control

Contributors

    Maurice Ribble
    Mikko Nurmi

Contact

    Maurice Ribble (mribble 'at' qualcomm.com)

Status

    Incomplete.

Version

    Last Modified Date: May 5, 2009
    Revision: #0.3

Number

    OpenGL ES Extension #55

Dependencies

    OpenGL ES 1.0 or higher is required.

    This extension is written based on the wording of the OpenGL ES 2.0
    specification.

Overview

    This extension exposes special control features in a driver to a
    developer. A use of these controls would be to override state or
    implement special modes of operation. One common example might be an
    IFH or infinitely fast hardware mode. In this mode none of draw
    commands would be sent to the GPU so no image would be displayed,
    but all the driver software overhead would still happen thus
    enabling developers to analyze driver overhead separate from GPU
    performance. Some uses of this extension could invalidate future
    rendering and thus should only be used by developers for debugging
    and performance profiling purposes.

    The extension is general enough to allow the implementation to
    choose which controls to expose and to provide a textual description
    of those controls to developers.

Issues

    None.

New Procedures and Functions

    void GetDriverControlsQCOM(int *num, sizei size, uint *driverControls)

    void GetDriverControlStringQCOM(uint driverControl,
                                   sizei bufSize,
                                   sizei *length,
                                   char *driverControlString)

    void EnableDriverControlQCOM(uint driverControl)

    void DisableDriverControlQCOM(uint driverControl)

New Tokens

    None.

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Add a new section "Driver Control":

    The command

        void GetDriverControlsQCOM(int *num, sizei size, uint *driverControls)

    returns the number of available driver controls in <num>, if <num>
    is not NULL. If <size> is not 0 and <driverControls> is not NULL,
    then the list of available driver controls is returned. The number
    of entries that will be in <driverControls> is determined by <size>.
    If <size> is 0, no information is copied. Each driver control is
    identified by a unique unsigned int identifier.

    The command

        void GetDriverControlStringQCOM(uint driverControl,
                                       sizei bufSize,
                                       sizei *length,
                                       char *driverControlString)

    returns the string that describes the driver control identified by
    <driverControl> in <driverControlString>. The first part of this
    string is a unique identifying name for the driver control. A space
    signifies the end of this name. It is encouraged that implementers
    not change the name so that developers can key off this name for
    their implementation. The actual number of characters written to
    <driverControlString>, excluding the null terminator, is returned in
    <length>. If <length> is NULL, then no length is returned. The
    maximum number of characters that may be written into
    <driverControlString>, including the null terminator, is specified
    by <bufSize>. If <bufSize> is 0 and <driverControlString> is NULL,
    the number of characters that would be required to hold the
    driverControl string, excluding the null terminator, is returned in
    <length>. If <driverControl> does not reference a valid
    driverControl ID, an INVALID_VALUE error is generated.

    The command

        void EnableDriverControlQCOM(uint driverControl)

    enables a driver control.

        void DisableDriverControlQCOM(uint driverControl)

    disables a driver control.


GLX Protocol

    None.

Errors

    INVALID_VALUE error will be generated if the <driverControl>
    parameter to GetDriverControlStringQCOM, EnableDriverControlQCOM or
    DisableDriverControlQCOM does not reference a valid driver control
    ID.

New State

    None.


Revision History
    11/10/2008 - Maurice Ribble
       + Initial version written.
    03/19/2009 - Maurice Ribble
       + Change from AMD to Qualcomm extension for publication
    05/05/2009 - Jon Leech
       + Reflow paragraphs and assign extension number.
