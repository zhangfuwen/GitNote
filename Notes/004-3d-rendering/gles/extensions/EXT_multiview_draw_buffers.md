# EXT_multiview_draw_buffers

Name

    EXT_multiview_draw_buffers

Name Strings

    GL_EXT_multiview_draw_buffers

Contributors

    Acorn Pooley, NVIDIA
    Greg Roth, NVIDIA
    Maurice Ribble, Qualcomm

Contact

    Greg Roth (groth 'at' nvidia.com)

Version

    Version 3, Sept 03, 2011

Number

    OpenGL ES Extension #125

Status
    
    Complete

Dependencies

    Written against the OpenGL ES 2.0 Specification

    NV_draw_buffers affects the definition of this extension.

    OpenGL ES 3.0 affects the definition of this extension.

Overview

    This extension allows selecting among draw buffers as the
    rendering target. This may be among multiple primary buffers
    pertaining to platform-specific stereoscopic or multiview displays
    or among offscreen framebuffer object color attachments.

    To remove any artificial limitations imposed on the number of
    possible buffers, draw buffers are identified not as individual
    enums, but as pairs of values consisting of an enum representing
    buffer locations such as COLOR_ATTACHMENT_EXT or MULTIVIEW_EXT,
    and an integer representing an identifying index of buffers of this
    location. These (location, index) pairs are used to specify draw
    buffer targets using a new DrawBuffersIndexedEXT call.

    Rendering to buffers of location MULTIVIEW_EXT associated with the
    context allows rendering to multiview buffers created by EGL using
    EGL_EXT_multiview_window for stereoscopic displays.

    Rendering to COLOR_ATTACHMENT_EXT buffers allows implementations to
    increase the number of potential color attachments indefinitely to
    renderbuffers and textures.

    This extension allows the traditional quad buffer stereoscopic
    rendering method that has proven effective by indicating a left or
    right draw buffer and rendering to each accordingly, but is also
    dynamic enough to handle an arbitrary number of color buffer targets
    all using the same shader. This grants the user maximum flexibility
    as well as a familiar interface.

New Procedures and Functions

    void ReadBufferIndexedEXT(enum src, int index);
    void DrawBuffersIndexedEXT(int n, const enum *location,
                               const int *indices);
    void GetIntegeri_vEXT(enum target, uint index, int *data);

New Tokens

    Accepted by the <location> parameter of DrawBuffersIndexedEXT:

        COLOR_ATTACHMENT_EXT                0x90F0
        MULTIVIEW_EXT                       0x90F1

    Accepted by the <target> parameter of GetIntegeri_EXT:

        DRAW_BUFFER_EXT                     0x0C01
        READ_BUFFER_EXT                     0x0C02

    Accepted by the <target> parameter of GetInteger:

        MAX_MULTIVIEW_BUFFERS_EXT           0x90F2


Changes to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify section 4.2.1, "Selecting a Buffer for Writing"

    Change first paragraph to:

    By default, color values are written into the front buffer for
    single buffered contexts or into the back buffer for back buffered
    contexts as determined when creating the GL context. To control
    the color buffer into which each of the fragment color values is
    written, DrawBuffersNV or DrawBuffersIndexedEXT is used.

    Add to the end of 4.2.1:

    The command

        void DrawBuffersIndexedEXT(sizei n, const enum *locations,
                                   const int *indices);

    defines the draw buffers to which all fragment colors are written.
    <n> specifies the number of values in <locations> and <indices>.
    <locations> is a pointer to an array of symbolic constants
    specifying the location of the draw buffer. <indices> is a pointer
    to an array of integer values specifying the index of the draw
    buffer. Together <locations> and <indices> specify the draw buffer
    to which each fragment color is written.

    Each constant in <locations> must be MULTIVIEW_EXT, COLOR_-
    ATTACHMENT_EXT, or NONE. Otherwise, an INVALID_ENUM error is
    generated. Further, acceptable values for the constants in
    <locations> depend on whether the GL is using the default
    framebuffer (i.e. DRAW_FRAMEBUFFER_BINDING is non-zero). For more
    information about framebuffer objects, see section 4.4.
    
    If the GL is bound to the default framebuffer, then each of the
    location constants must be MULTIVIEW_EXT or NONE.
    
    If the GL is bound to a framebuffer object, then each of the
    location constants must be COLOR_ATTACHMENT_EXT or NONE.

    Where the constant in <locations> is MULTIVIEW_EXT, the
    corresponding value in <indices> must be a value from 0 through
    MAX_MULTIVIEW_BUFFERS_EXT. Where the constant in <locations> is
    COLOR_ATTACHMENT_EXT, the value in <indices> must be a value from 0
    through MAX_COLOR_ATTACHMENTS_NV. Otherwise, an INVALID_OPERATION
    error is generated. Where the constant in <locations> is NONE, the
    value in <indices> is ignored.

    For monoscopic rendering, the only available view is index 0. For
    stereoscopic rendering, view index 0 corresponds to the left buffer
    and view index 1 corresponds to the right buffer.

    The draw buffers being defined correspond, in order, to the
    respective fragment colors. The draw buffer for fragment colors
    beyond <n> is set to NONE.

    Except for where the constant in <locations> is NONE, a buffer may
    not be specified more than once by the arrays pointed to by
    <locations> and <indices>. Specifying a buffer more than once will
    result in the error INVALID_OPERATION.

    If a fragment shader writes to "gl_FragColor", DrawBuffersIndexedEXT
    specifies a set of draw buffers into which the color written to
    "gl_FragColor" is written. If a fragment shader writes to
    gl_FragData, DrawBuffersIndexedEXT specifies a set of draw buffers
    into which each of the multiple output colors defined by these
    variables are separately written. If a fragment shader writes to
    neither gl_FragColor nor gl_FragData, the values of the fragment
    colors following shader execution are undefined, and may differ
    for each fragment color.
    
    Indicating a buffer or buffers using DrawBuffersIndexedEXT causes
    subsequent pixel color value writes to affect the indicated
    buffers.  If more than one color buffer is selected for drawing,
    blending is computed and applied independently for each buffer.

    Specifying NONE in the <locations> array for a fragment color will
    inhibit that fragment color from being written to any buffer.

    Monoscopic surfaces include only left buffers, while stereoscopic
    surfaces include a left and a right buffer, and multiview surfaces
    include more than 1 buffer (a stereoscopic surface is a multiview
    surface with 2 buffers).  The type of surface is selected at EGL
    surface initialization.

    The state required to handle color buffer selection is two integers
    for each supported fragment color for each framebuffer or
    framebuffer object. For the default framebuffer, the initial state
    of the draw buffer location for fragment color zero is MULTIVIEW_EXT
    and the index is 0. For framebuffer objects, the initial state of
    the draw buffer location for fragment color zero is COLOR_-
    ATTACHMENT_EXT and the index is 0. The initial state of draw buffers
    for fragment colors other than zero is NONE.
    
    The color buffer location and index to which fragment colors are
    written for an output color index <i> can be queried by calling
    GetIntegeri_vEXT with <target> DRAW_BUFFER_EXT and index
    <i>. This returns a pair of values representing the draw buffer
    location and index. The number of multiview buffers available to a
    GL context can be queried by calling GetIntegerv with <target>
    MAX_MULTIVIEW_BUFFERS_EXT.

    Section 4.3.1 (Reading Pixels), subsection "Obtaining Pixels from
    the Framebuffer" add:

    For color formats, the read buffer from which values are obtained is
    one of the color buffers; the selection of color buffer is
    controlled with ReadBufferIndexedEXT.

    The command

        void ReadBufferIndexedEXT(enum location, int index);

    takes a symbolic constant and integer pair to select the color
    buffer from which color values are obtained. <location> must be one
    of MULTIVIEW_EXT, COLOR_ATTACHMENT_EXT, or NONE. Otherwise, an
    INVALID_ENUM error is generated. If <location> is MULTIVIEW_EXT,
    <index> must be a value from 0 through MAX_MULTIVIEW_BUFFERS_EXT.
    If <location> is COLOR_ATTACHMENT_EXT, <index> must be a value from
    0 through MAX_COLOR_ATTACHMENTS_NV. Otherwise, an INVALID_OPERATION
    error is generated. If <location> is NONE, <index> is ignored.

    The acceptable values for <location> depend on whether the GL is
    using the default framebuffer (i.e. FRAMEBUFFER_BINDING is zero), or
    a framebuffer object (i.e. FRAMEBUFFER_BINDING is non-zero). For
    more information about framebuffer objects, see section 4.4.

    If the object bound to FRAMEBUFFER_BINDING is not framebuffer
    complete (as defined in section 4.4.5), then ReadPixels generates
    the error INVALID_FRAMEBUFFER_OPERATION. If <location> is a constant
    that is neither legal for the default framebuffer, nor legal for a
    framebuffer object, then the error INVALID_ENUM results.

    When FRAMEBUFFER_BINDING is zero, i.e. the default framebuffer,
    <location> must be MULTIVIEW_EXT or NONE. If the buffer indicated by
    <index> is missing, the error INVALID_OPERATION is generated. For
    the default framebuffer, the initial setting for READ_BUFFER_EXT is
    <location> of MULTIVIEW_EXT and <index> of zero.

    When the GL is using a framebuffer object, <location> must be NONE
    or COLOR_ATTACHMENT_EXT. Specifying COLOR_ATTACHMENT_EXT enables
    reading from the image attached to the framebuffer at COLOR_-
    ATTACHMENT<index>_NV. For framebuffer objects, the initial setting
    for READ_BUFFER_EXT is <location> of COLOR_ATTACHMENT_EXT and
    <index> of zero.

    ReadPixels generates an INVALID_OPERATION error if it attempts to
    select a color buffer while READ_BUFFER_EXT is none.

Changes to chapter 6

    Add to section 6.1.1, "Simple Queries" before description of
    IsEnabled:

    Indexed simple state variables are queried with the command

        void GetIntegeri_vEXT(enum target, uint index, int* data);

    <target> is the name of the indexed state and <index> is the
    index of the particular element being queried. <data> is a
    pointer to a scalar or array of the indicated type in which
    to place the returned data. An INVALID_VALUE error is generated
    if <index> is outside the valid range for the indexed state
    <target>.

Changes to Chapter 3 of the OpenGL Shading Language 1.0 Specification
(Basics)

    Add a new section:

    3.4.1 GL_EXT_multiview_draw_buffers Extension

    To use the GL_EXT_multiview_draw_buffers extension in a shader it
    must be enabled using the #extension directive.

    The shading language preprocessor #define GL_EXT_multiview_draw_-
    buffers will be defined to 1, if the GL_EXT_multiview_draw_buffers
    extension is supported.

Dependencies on NV_draw_buffers:

    If NV_draw_buffers is not supported and OpenGL ES 3 is, add to the
    description of DrawBuffersIndexedEXT:

        The <i>th index listed in <indices> must be <i> or NONE.
        Specifying a buffer out of order will generate the error
        INVALID_OPERATION.

    If neither NV_draw_buffers nor OpenGL ES 3 is supported, all
    references to DrawBuffersNV and color attachments are removed. The
    following is substituted for the above changes to Chapter 4 (Per-
    Fragment Operations and the Frame Buffer):

    Change section 4.2.1, "Selecting a Buffer for Writing"

    Change first paragraph to:

    By default, color values are written into the front buffer for
    single buffered contexts or into the back buffer for back buffered
    contexts as determined when creating the GL context. To control
    the color buffer into which each of the fragment color values is
    written, DrawBuffersIndexedEXT is used.

    Add to the end of 4.2.1:

    The command

        void DrawBuffersIndexedEXT(sizei n, const enum *locations,
                                   const int *indices);

    defines the draw buffers to which all fragment colors are written.
    <n> specifies the number of values in <locations> and <indices>.
    <locations> is a pointer to an array of symbolic constants
    specifying the location of the draw buffer. <indices> is a pointer
    to an array of integer values specifying the index of the draw
    buffer. Together <locations> and <indices> specify the draw buffer
    to which each fragment color is written.

    Each constant in <locations> must be MULTIVIEW_EXT or NONE.
    Otherwise, an INVALID_ENUM error is generated.

    DrawBuffersIndexedEXT generates an INVALID_OPERATION error if the GL
    is bound to a framebuffer object.
    
    Where the constant in <locations> is MULTIVIEW_EXT, the
    corresponding value in <indices> must be a value from 0 through
    MAX_MULTIVIEW_BUFFERS_EXT. Otherwise, an INVALID_OPERATION error is
    generated. Where the constant in <locations> is NONE, the value
    in <indices> is ignored.

    An INVALID_VALUE error is generated if <n> is not 1.

    For monoscopic rendering, the only available view is index 0. For
    stereoscopic rendering, view index 0 is left and view index 1 is
    right.

    If a fragment shader writes to "gl_FragColor" or "gl_FragData[0]",
    DrawBuffersIndexedEXT specifies a set of draw buffers into which the
    output color is written. If a fragment shader writes to neither
    gl_FragColor nor gl_FragData[0], the values of the fragment colors
    following shader execution are undefined, and may differ for each
    fragment color.
    
    Indicating a buffer using DrawBuffersIndexedEXT causes subsequent
    pixel color value writes to affect the indicated buffer.

    Specifying NONE in the <locations> array for a fragment color will
    inhibit that fragment color from being written to any buffer.

    Monoscopic surfaces include only left buffers, while stereoscopic
    surfaces include a left and a right buffer, and multiview surfaces
    include more than 1 buffer (a stereoscopic surface is a multiview
    surface with 2 buffers).  The type of surface is selected at EGL
    surface initialization.

    The state required to handle color buffer selection is two integers
    for the fragment color output for each framebuffer or framebuffer
    object. For the default framebuffer, the initial state of the draw
    buffer location for fragment color zero is MULTIVIEW_EXT and the
    index is 0. For framebuffer objects, the initial state of the draw
    buffer location for fragment color zero is COLOR_ATTACHMENT_EXT and
    the index is 0.

    The color buffer location and index to which fragment colors are
    written can be queried by calling GetIntegeri_vEXT with <target>
    DRAW_BUFFER_EXT and index 0. This returns a pair of values
    representing the draw buffer location and index. The number of
    multiview buffers available to a GL context can be queried by
    calling GetIntegerv with <target> MAX_MULTIVIEW_BUFFERS_EXT.

    The command

        void ReadBufferIndexedEXT(enum location, int index);

    takes a symbolic constant and integer pair to select the color
    buffer from which color values are obtained. <location> must be
    MULTIVIEW_EXT or NONE. Otherwise, an INVALID_ENUM error is
    generated. If <location> is MULTIVIEW_EXT, <index> must be a value
    from 0 through MAX_MULTIVIEW_BUFFERS_EXT. Otherwise, an
    INVALID_OPERATION error is generated. If <location> is NONE, the
    <index> is ignored.

    When FRAMEBUFFER_BINDING is zero, i.e. the default framebuffer,
    <location> must be MULTIVIEW_EXT or NONE. If the buffer indicated by
    <index> is missing, the error INVALID_OPERATION is generated. For
    the default framebuffer, the initial setting for READ_BUFFER_EXT is
    <location> of MULTIVIEW_EXT and <index> of zero.

    ReadBufferIndexedEXT generates an INVALID_OPERATION error if the GL
    is bound to a framebuffer object.

    ReadPixels generates an INVALID_OPERATION error if it attempts to
    select a color buffer while READ_BUFFER_EXT is none.

New State

    Add the new Table 6.X "Framebuffer (State per framebuffer object)" :

        State                     Type    Get Command      Initial    Description
        ---------------           -----   ---------------  -------    -----------
        DRAW_BUFFER_EXT           nxZ+    GetIntegeri_vEXT See 4.2.1  Multiview draw buffer
                                                                      location and index
                                                                      selected for the
                                                                      specified color output
        READ_BUFFER_EXT           nxZ+    GetInteger       See 4.3.1  Read source buffer
                                                                      location and index

    Add the new Table 6.X "Framebuffer Dependent Values" :

        State                     Type    Get Command     Min Value  Description
        ---------------           -----   --------------- ---------  -----------
        MAX_MULTIVIEW_BUFFERS_EXT Z+      GetIntegerv     1          Number of multiview
                                                                     draw buffers

Issues

    1. What should this extension be called?

    RESOLVED: multiview_draw_buffers. Multiview has come to be the
    standard term to refer to stereoscopic and beyond buffer rendering
    and this approach centers around the traditional usage of
    a drawbuffers call to specify the buffer(s) to render to.

    2. How should draw buffer bindings be queried?

    RESOLVED: A new indexed integer query function called
    glGetIntegeri_vEXT. This extension adds an indexed binding for draw
    buffers so it follows that an indexed query should be used to
    retrieve the state that it sets. The name glGetIntegeri_vEXT is
    chosen as it was in desktop GL to clarify the 'i' suffix
    indicating an indexed call as opposed to 'i' indicating an integer
    variant of a call accepting parameters of various types.

    3. Should the <location> parameter of DrawBuffersIndexedEXT be removed?

    RESOLVED: No. It is useful when draw_buffers is supported.

Revision History
    Version 4, 25 Sept 2012 Clean up overview. Fix a few typographical
                            errors.
    Version 3, 03 Sept 2011 EXTify. Remove ALL broadcast. 
                            Add interactions for ES3 and non-
                            draw_buffers cases
    Version 2, 02 Aug 2011 Responses to feedback.
    Version 1, 14 April 2011 First draft.

