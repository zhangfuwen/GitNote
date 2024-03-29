# NV_draw_buffers

Name

    NV_draw_buffers

Name Strings

    GL_NV_draw_buffers

Contact

    Greg Roth, NVIDIA Corporation (groth 'at' nvidia.com)

Contributors

     Benj Lipchak, AMD
     Bill Licea-Kane, AMD
     Rob Mace, NVIDIA Corporation
     James Helferty, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date: July 11, 2013
    NVIDIA Revision: 4.0

Number

    OpenGL ES Extension #91

Dependencies

    Written against the OpenGL ES 2.0 Specification and the OpenGL ES
    Shader Language 1.0.14 Specification.

    This extension interacts with the OpenGL ES 3.0 Specification

Overview

    This extension extends OpenGL ES 2.0 to allow multiple output
    colors, and provides a mechanism for directing those outputs to
    multiple color buffers.

    This extension serves a similar purpose to ARB_draw_buffers except
    that this is for OpenGL ES 2.0.

    When OpenGL ES 3.0 is present, this extension relaxes the order
    restriction on color attachments to draw framebuffer objects.

IP Status

    NVIDIA Proprietary

New Procedures and Functions

    void DrawBuffersNV(sizei n, const enum *bufs);

New Tokens

    Accepted by the <pname> parameters of GetIntegerv, GetFloatv,
    and GetDoublev:

        MAX_DRAW_BUFFERS_NV                     0x8824
        DRAW_BUFFER0_NV                         0x8825
        DRAW_BUFFER1_NV                         0x8826
        DRAW_BUFFER2_NV                         0x8827
        DRAW_BUFFER3_NV                         0x8828
        DRAW_BUFFER4_NV                         0x8829
        DRAW_BUFFER5_NV                         0x882A
        DRAW_BUFFER6_NV                         0x882B
        DRAW_BUFFER7_NV                         0x882C
        DRAW_BUFFER8_NV                         0x882D
        DRAW_BUFFER9_NV                         0x882E
        DRAW_BUFFER10_NV                        0x882F
        DRAW_BUFFER11_NV                        0x8830
        DRAW_BUFFER12_NV                        0x8831
        DRAW_BUFFER13_NV                        0x8832
        DRAW_BUFFER14_NV                        0x8833
        DRAW_BUFFER15_NV                        0x8834


    Accepted by the <bufs> parameter of DrawBuffersNV:

        COLOR_ATTACHMENT0_NV                    0x8CE0
        COLOR_ATTACHMENT1_NV                    0x8CE1
        COLOR_ATTACHMENT2_NV                    0x8CE2
        COLOR_ATTACHMENT3_NV                    0x8CE3
        COLOR_ATTACHMENT4_NV                    0x8CE4
        COLOR_ATTACHMENT5_NV                    0x8CE5
        COLOR_ATTACHMENT6_NV                    0x8CE6
        COLOR_ATTACHMENT7_NV                    0x8CE7
        COLOR_ATTACHMENT8_NV                    0x8CE8
        COLOR_ATTACHMENT9_NV                    0x8CE9
        COLOR_ATTACHMENT10_NV                   0x8CEA
        COLOR_ATTACHMENT11_NV                   0x8CEB
        COLOR_ATTACHMENT12_NV                   0x8CEC
        COLOR_ATTACHMENT13_NV                   0x8CED
        COLOR_ATTACHMENT14_NV                   0x8CEE
        COLOR_ATTACHMENT15_NV                   0x8CEF


Changes to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Section 3.2, (Multisampling). Replace the second paragraph:

    An additional buffer, called the multisample buffer, is added to the
    framebuffer. Pixel sample values, including color, depth, and
    stencil values, are stored in this buffer. Samples contain separate
    color values for each fragment color. When the framebuffer includes
    a multisample buffer, it does not include depth or stencil buffers,
    even if the multisample buffer does not store depth or stencil
    values. The color buffer does coexist with the multisample buffer,
    however.

    Section 3.8.2, (Shader Execution) Replace subsection "Shader
    Outputs":

    The OpenGL ES Shading Language specification describes the values
    that may be output by a fragment shader. These are gl_FragColor and
    gl_FragData[n].  The final fragment color values or the final
    fragment data values written by a fragment shader are clamped to the
    range [0, 1] and then converted to fixed-point as described in
    section 2.1.2 for framebuffer color components.

    Writing to gl_FragColor specifies the fragment color (color number
    zero) that will be used by subsequent stages of the pipeline.
    Writing to gl_FragData[n] specifies the value of fragment color
    number n. Any colors, or color components, associated with a
    fragment that are not written by the fragment shader are undefined.
    A fragment shader may not statically assign values to both
    gl_FragColor and gl_FragData. In this case, a compile or link error
    will result. A shader statically assigns a value to a variable if,
    after preprocessing, it contains a statement that would write to the
    variable, whether or not run-time flow of control will cause that
    statement to be executed.

Changes to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Replace Section 4.2.1, "Selecting a Buffer for Writing"

    By default, color values are written into the front buffer for
    single buffered surfaces or into the back buffer for back buffered
    surfaces as determined when making the context current. To control
    the color buffer into which each of the fragment color values is
    written, DrawBuffersNV is used.

    The command

      void DrawBuffersNV(sizei n, const enum *bufs);

    defines the draw buffers to which all fragment colors are written.
    <n> specifies the number of buffers in <bufs>. <bufs> is a pointer
    to an array of symbolic constants specifying the buffer to which
    each fragment color is written.

    Each buffer listed in <bufs> must be NONE, COLOR_ATTACHMENT0, or
    COLOR_ATTACHMENTi_NV, where <i> is the index of the color attachment
    point. Otherwise, an INVALID_ENUM error is generated. DrawBuffersNV
    may only be called when the GL is bound to a framebuffer object. If
    called when the GL is bound to the default framebuffer, an INVALID_-
    OPERATION error is generated.

    The draw buffers being defined correspond in order to the respective
    fragment colors. The draw buffer for fragment colors beyond <n> is
    set to NONE.

    The maximum number of draw buffers is implementation dependent and
    must be at least 1. The number of draw buffers supported can be
    queried by calling GetIntegerv with the symbolic constant
    MAX_DRAW_BUFFERS_NV. An INVALID_VALUE error is generated if <n> is
    less than one or greater than MAX_DRAW_BUFFERS_NV.

    Except for NONE, a buffer may not appear more then once in the array
    pointed to by <bufs>. Specifying a buffer more then once will result
    in the error INVALID_OPERATION.

    If a fragment shader writes to "gl_FragColor", DrawBuffersNV
    specifies a set of draw buffers into which the color written to
    "gl_FragColor" is written. If a fragment shader writes to
    gl_FragData, DrawBuffers specifies a set of draw buffers into which
    each of the multiple output colors defined by these variables are
    separately written. If a fragment shader writes to neither
    gl_FragColor nor gl_FragData the values of the fragment colors
    following shader execution are undefined, and may differ for each
    fragment color.

    If DrawBuffersNV is supplied with a constant COLOR_ATTACHMENT<m>
    where <m> is greater than or equal to the value of
    MAX_COLOR_ATTACHMENTS_NV, then the error INVALID_OPERATION results.

    Indicating a buffer or buffers using DrawBuffersNV causes subsequent
    pixel color value writes to affect the indicated buffers. If the GL is
    bound to a draw framebuffer object and a draw buffer selects an attachment
    that has no image attached, then that fragment color is not written.

    Specifying NONE as the draw buffer for a fragment color will inhibit
    that fragment color from being written to any buffer.

    The state required to handle color buffer selection is an integer
    for each supported fragment color.  For each framebuffer object, the
    initial state of the draw buffer for fragment color zero is COLOR_-
    ATTACHMENT0 and the initial state of draw buffers for fragment
    colors other than zero is NONE.

    The value of the draw buffer selected for fragment color <i> can be
    queried by calling GetIntegerv with the symbolic constant
    DRAW_BUFFER<i>_NV.

Changes to Chapter 3 of the OpenGL Shading Language 1.0 Specification (Basics)

    Add a new section:

    3.3.1 GL_NV_draw_buffers Extension

    To use the GL_NV_draw_buffers extension in a shader it must be
    enabled using the #extension directive.

    The shading language preprocessor #define GL_NV_draw_buffers will be
    defined to 1, if the GL_NV_draw_buffers extension is supported.

Interactions with OpenGL ES 3.0

    Section 4.2.1 of OpenGL ES 3.0, (Selecting a Buffer for Writing). Replace
    the eighth and ninth paragraph:

    If the GL is bound to a draw framebuffer object, the ith buffer listed in
    bufs must be COLOR_ATTACHMENTm or NONE, where m is less than the value of
    MAX_COLOR_ATTACHMENTS. Specifying BACK or COLOR_ATTACHMENTm where m is
    greater than or equal to MAX_COLOR_ATTACHMENTS will generate the error
    INVALID_OPERATION.

    If an OpenGL ES Shading Language 1.00 or 3.00 fragment shader writes a
    user-defined varying out variable, DrawBuffers specifies a set of draw
    buffers into which each of the multiple output colors defined by these
    variables are separately written. If a fragment shader writes to none of
    gl_FragColor, gl_FragData, nor any user-defined output variables, the
    values of the fragment colors following shader execution are undefined, and
    may differ for each fragment color. If some, but not all user-defined
    output variables are written, the values of fragment colors corresponding
    to unwritten variables are similarly undefined.

New State

    Add Table 6.X Framebuffer (State per framebuffer object):

        State           Type  Get Command Initial Value Description 
        --------------- ---- ------------ ------------- -----------
        DRAW_BUFFERi_NV Z10* GetIntegerv  see 4.2.1     Draw buffer selected 
                                                        for fragment color i

    Add the new Table 6.X "Framebuffer Dependent Values" :

        State               Type Get Command Min Value Description
        ------------------- ---- ----------- --------- -----------
        MAX_DRAW_BUFFERS_NV  Z+  GetIntegerv 1         Maximum number of
                                                       active draw buffers

Issues

    1. What behavior should be expected if a draw buffer selects an attachment
       for a draw framebuffer that has no image attached?

      Early drivers considered this an INVALID_OPERATION and the DrawBuffersNV
      operation did not succeed. (Following precedent set in Issue 55 of
      EXT_framebuffer_object)

      OpenGL ES 3.0 and Desktop GL 4.3 consider this legal, and the DrawBuffers
      call succeeds. Missing attachments are simply not written to.

      RESOLVED: Behavior should match OpenGL ES 3.0. Application developers are
      cautioned that early Tegra drivers may exhibit the previous behavior.

    See ARB_draw_buffers for additional relevant issues.

Revision History

    Rev.    Date      Author       Changes
    ----   --------   ---------    ------------------------------------
     4     07/11/13   jhelferty    Updated error behavior for missing
                                   attachments to match ES 3.0.
                                   Clarified ES 3.0 interactions.
     3     06/07/11   groth        Clarified default behavior, state tables.
     2     04/26/11   groth        Filled in many missing elements.
     1     03/03/08   kashida      First revision based on
                                   ARB_draw_buffers.
