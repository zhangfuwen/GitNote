# INTEL_blackhole_render

Name

    INTEL_blackhole_render

Name Strings

    GL_INTEL_blackhole_render

Contact

    Lionel Landwerlin, Intel  (lionel.g.landwerlin 'at' intel.com)

Contributors

    Ben Widawsky (benjamin.widawsky 'at' intel.com)

Status

    Draft.

Version

    Last Modified Date:        03/02/2018
    INTEL Revision:            1

Number

    OpenGL Extension #521
    OpenGL ES Extension #300

Dependencies

    OpenGL dependencies:

        OpenGL 3.0 is required.

        The extension is written against the OpenGL 4.6 Specification, Core
        Profile, July 30, 2017.

    OpenGL ES dependencies:

        This extension is written against the OpenGL ES 3.2 Specification,
        November 3, 2016.

Overview

    The purpose of this extension is to allow an application to disable all
    rendering operations emitted to the GPU through the OpenGL rendering
    commands (Draw*, DispatchCompute*, BlitFramebuffer, etc...). Changes to the
    OpenGL pipeline are not affected.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_INTEL_blackhole_render 1

New Procedures and Functions

    None.

New Tokens

    Accepted by the <target> parameter of Enable, Disable, IsEnabled:

        BLACKHOLE_RENDER_INTEL  0x83FC

Additions to the OpenGL 4.6 (Core Profile) Specification

    Modify section 2.4 Rendering Commands (add new text at the end of the
    section)

    The effect of the above commands can be disabled by enabling
    BLACKHOLE_RENDER_INTEL.

Additions to Chapter 14.2.2, Shader Inputs of the OpenGL ES 3.2 Specification

    Modify section 2.4 Rendering Commands (add new text at the end of the
    section)

    The effect of the above commands can be disabled by enabling
    BLACKHOLE_RENDER_INTEL.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Errors

    None.

New State in OpenGL 4.6 Core Profile

    (add new row to the Table 23.74, Miscellaneous)

                                     Initial
    Get Value      Type  Get Command  Value  Description                 Sec.
    -------------  ----  ----------- ------- -------------------------   ------
    BLACKHOLE_
    RENDERING_     B     IsEnabled()  FALSE  Disable rendering           2.4
    INTEL

New State in OpenGL ES 3.2

    (add new row to the Table 21.57, Miscellaneous)

                                     Initial
    Get Value      Type  Get Command  Value  Description                 Sec.
    -------------  ----  ----------- ------- -------------------------   ------
    BLACKHOLE_
    RENDERING_     B     IsEnabled()  FALSE  Disable rendering           2.4
    INTEL

Issues

    None.

Revision History

    Rev.     Date       Author       Changes
    ----  ----------  ----------  -----------------------------------------
      1    3/2/2018   llandwerlin Initial revision.
