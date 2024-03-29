# MESA_program_binary_formats

Name

    MESA_program_binary_formats

Name Strings

    GL_MESA_program_binary_formats

Contributors

    Ian Romanick
    Jordan Justen
    Timothy Arceri

Contact

    Jordan Justen (jordan.l.justen 'at' intel.com)

Status

    Complete.

Version

    Last Modified Date: November 10, 2017
    Revision: #2

Number

    OpenGL Extension #516
    OpenGL ES Extension #294

Dependencies

    For use with the OpenGL ARB_get_program_binary extension, or the
    OpenGL ES OES_get_program_binary extension.

Overview

    The get_program_binary exensions require a GLenum binaryFormat.
    This extension documents that format for use with Mesa.

New Procedures and Functions

    None.

New Tokens

        GL_PROGRAM_BINARY_FORMAT_MESA           0x875F

    For ARB_get_program_binary, GL_PROGRAM_BINARY_FORMAT_MESA may be
    returned from GetProgramBinary calls in the <binaryFormat>
    parameter and when retrieving the value of PROGRAM_BINARY_FORMATS.

    For OES_get_program_binary, GL_PROGRAM_BINARY_FORMAT_MESA may be
    returned from GetProgramBinaryOES calls in the <binaryFormat>
    parameter and when retrieving the value of
    PROGRAM_BINARY_FORMATS_OES.

New State

    None.

Issues

    (1) Should we have a different format for each driver?

      RESOLVED. Since Mesa supports multiple hardware drivers, having
      a single format may cause separate drivers to have to reject a
      binary for another type of hardware on the same machine. This
      could lead to an application having to invalidate and get a new
      binary more often.

      This extension, at least initially, does not to attempt to
      define a new token for each driver since systems that run
      multiple drivers are not the common case.

      Additionally, drivers in Mesa are now gaining the ability to
      transparently cache shader programs. Therefore, although they
      may need to provide the application with a new binary more
      often, they likely can retrieve the program from the cache
      rather than performing an expensive recompile.

Revision History

    #02    11/10/2017    Jordan Justen       Add Issues (1) suggested by Ian

    #01    10/28/2017    Jordan Justen       First draft.
